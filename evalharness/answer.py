"""Proxy end-to-end scoring for a runner output file.

The official AML pipeline answers with gpt-4o-mini and judges with the same
model, using the templates copied verbatim below (`OPEN_ENDED_ANSWER_TEMPLATE`
and `ACCURACY_PROMPT` from `data/locomo-refined/pipeline.py` in the AML repo).
gpt-4o-mini is not served by the MaaS endpoint we have, so this script runs the
*same* prompts against a different hosted model.

Consequence: the number it prints is a RELATIVE signal for A/B comparison of
retrieval configurations. It is never an official score and must not be quoted
as one.

Usage:
    python -m evalharness.answer --run runs/final_default.json --per-category 60
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

OPEN_ENDED_ANSWER_TEMPLATE = """You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Use only the provided memories. Prefer the memory that answers the question most directly.
2. Your memories are episodic raw observations. Reason about what they imply. Do not refuse just because the answer is not stated verbatim.
3. The question may contain typos. Match it to the most relevant memory even if the wording differs.
4. When multiple answers are possible, list all supported answers, not just the first.
5. For counts or time intervals, enumerate carefully before answering.
6. Preserve specific names, titles, places, and labels from the memories. Use "Rob" not "a colleague", "Sweden" not "home country".
7. Convert relative times like "yesterday", "last month", and "last year" into dates, months, or years when the memory timestamp makes it clear. Keep week-based expressions relative.
8. If memories conflict, prefer the most recent supported memory.
9. For list questions, include all required items and no extras.
10. Keep the final answer minimal. Do not add explanation, background, or extra dates unless needed for correctness.
</instructions>

<memories>
Memories for user {{speaker_1_name}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_name}}:

{{speaker_2_memories}}
</memories>

Question: {{question}}
Answer with the shortest correct phrase or sentence. No preamble, no fluff:"""

ACCURACY_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```"""


def render_answer(question: str, contents: list[str], names: tuple[str, str] = ("speaker 1", "speaker 2")) -> str:
    memories = "\n\n".join(contents)
    return (
        OPEN_ENDED_ANSWER_TEMPLATE.replace("{{speaker_1_name}}", names[0])
        .replace("{{speaker_1_memories}}", memories)
        .replace("{{speaker_2_name}}", names[1])
        .replace("{{speaker_2_memories}}", "")
        .replace("{{question}}", question)
    )


_LABEL = re.compile(r'"label"\s*:\s*"\s*(CORRECT|WRONG)\s*"', re.I)


def parse_label(text: str) -> str:
    match = _LABEL.search(text or "")
    if match:
        return match.group(1).upper()
    tail = (text or "")[-400:].upper()
    if "WRONG" in tail and "CORRECT" not in tail:
        return "WRONG"
    if "CORRECT" in tail and "WRONG" not in tail:
        return "CORRECT"
    return "UNPARSED"


class Chat:
    def __init__(self, base: str, key: str, model: str, timeout: float = 180.0, retries: int = 3):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.failures = 0

    def __call__(self, system: str, user: str, max_tokens: int = 256) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        last = ""
        for attempt in range(self.retries):
            try:
                with httpx.Client(timeout=self.timeout) as cli:
                    response = cli.post(f"{self.base}/chat/completions", headers={
                        "Authorization": f"Bearer {self.key}",
                        "Content-Type": "application/json",
                    }, json=payload)
                if response.status_code >= 500 or response.status_code == 429:
                    last = f"{response.status_code} {response.text[:200]}"
                    continue
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                self.calls += 1
                return str((body.get("choices") or [{}])[0].get("message", {}).get("content") or "")
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
        self.failures += 1
        return f"__ERROR__ {last}"


def score_one(chat_answer: Chat, chat_judge: Chat, row: dict) -> dict:
    contents = row.get("contents") or []
    prompt = render_answer(row["question"], contents)
    generated = chat_answer("You answer strictly from the provided memories.", prompt, max_tokens=200)
    if generated.startswith("__ERROR__"):
        return {"label": "ERROR", "generated": generated, "explanation": ""}
    judge_user = ACCURACY_PROMPT.format(question=row["question"], gold_answer=row["answer"],
                                        generated_answer=generated.strip())
    verdict = chat_judge("You are a strict answer grader.", judge_user, max_tokens=600)
    return {"label": parse_label(verdict), "generated": generated.strip(), "explanation": verdict.strip()[:600]}


_RATE_YUAN_PER_M = {  # rough list price, yuan per million *input* tokens
    "qwen-flash": 0.15,
    "qwen-turbo": 0.3,
    "qwen-plus": 0.8,
    "qwen-max": 2.4,
    "qwen3-max": 6.0,
    "deepseek": 2.0,
    "glm": 4.0,
}


def _rate(model: str) -> float:
    """Approximate price so a run can print what it just cost.

    These are list prices, not invoices: the point is to make the expensive step
    visible *before* the account runs dry, not to reconcile the bill.  Override with
    e.g. PROXY_RATE_QWEN_MAX=20 if the deployment prices differently.
    """
    env_key = "PROXY_RATE_" + model.upper().replace("-", "_").replace(".", "_")
    if os.environ.get(env_key):
        try:
            return float(os.environ[env_key])
        except ValueError:
            pass
    for name, rate in _RATE_YUAN_PER_M.items():
        if model.startswith(name):
            return rate
    return 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="runner output json with --keep-content")
    parser.add_argument("--per-category", type=int, default=0, help="0 = all questions")
    parser.add_argument("--only-categories", default="", help="comma-separated subset, e.g. multihop,open_domain")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--answer-model", default=os.environ.get("PROXY_ANSWER_MODEL", "qwen-flash"))
    parser.add_argument("--judge-model", default=os.environ.get("PROXY_JUDGE_MODEL", "qwen-flash"))
    # The judge is the expensive half of this harness: a full 1 382-question suite judged
    # by a max-tier model is real money, and this is only ever used to compare
    # configurations against each other.  Cheap judge + a hard token budget by default;
    # asking for more is an explicit decision.
    parser.add_argument("--budget-tokens", type=int,
                        default=int(os.environ.get("PROXY_BUDGET_TOKENS", "1500000")),
                        help="stop starting new questions once this many tokens are spent (0 = no cap)")
    parser.add_argument("--sample", type=int, default=0,
                        help="stratified sample of N questions across categories (0 = all)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from aml_memory.config import load_settings

    settings = load_settings()
    base = settings.embed_base_url
    key = settings.embed_api_key
    if not (base and key):
        raise SystemExit("no EMBED_BASE_URL / EMBED_API_KEY configured")

    data = json.loads(Path(args.run).read_text(encoding="utf-8"))
    rows = [r for r in data["results"] if r.get("contents")]
    if not rows:
        raise SystemExit(f"{args.run} has no stored contents — rerun the runner with --keep-content")

    if args.per_category:
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(row.get("category") or "unknown", []).append(row)
        picked: list[dict] = []
        for category, bucket in sorted(buckets.items()):
            step = max(1, len(bucket) // args.per_category)
            picked.extend(bucket[::step][: args.per_category])
        rows = picked
    if args.only_categories:
        wanted = {name.strip() for name in args.only_categories.split(",") if name.strip()}
        rows = [row for row in rows if (row.get("category") or "unknown") in wanted]
    if args.sample and args.sample < len(rows):
        # Stratified by category so a 300-question run still says something about every
        # slice; deterministic for a given seed so two configs are compared on the same
        # questions.
        import random

        rng = random.Random(args.seed)
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(row.get("category") or "unknown", []).append(row)
        picked: list[dict] = []
        remaining = args.sample
        for category, bucket in sorted(buckets.items()):
            share = max(1, round(args.sample * len(bucket) / len(rows)))
            share = min(share, len(bucket), remaining)
            picked.extend(rng.sample(bucket, share))
            remaining -= share
            if remaining <= 0:
                break
        rows = picked
    if args.limit:
        rows = rows[: args.limit]
    print(f"scoring {len(rows)} questions with answer={args.answer_model} judge={args.judge_model}", flush=True)

    chat_answer = Chat(base, key, args.answer_model)
    chat_judge = Chat(base, key, args.judge_model)

    out: list[dict] = [None] * len(rows)  # type: ignore[list-item]

    skipped = 0

    def work(index: int) -> None:
        nonlocal skipped
        if args.budget_tokens and (
            chat_answer.prompt_tokens + chat_judge.prompt_tokens >= args.budget_tokens
        ):
            skipped += 1
            return
        row = rows[index]
        verdict = score_one(chat_answer, chat_judge, row)
        out[index] = {"category": row.get("category"), "sample_id": row.get("sample_id"),
                      "question": row.get("question"), "gold": row.get("answer"),
                      "label": verdict["label"], "generated": verdict["generated"],
                      "first_rank": row.get("first_rank"), "ans_str": row.get("ans_str"),
                      "explanation": verdict["explanation"]}
        done = sum(1 for item in out if item is not None)
        if done % 10 == 0:
            labels = [item["label"] for item in out if item is not None]
            acc = sum(1 for label in labels if label == "CORRECT") / max(len(labels), 1)
            print(f"  {done}/{len(rows)} acc={acc:.3f}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        list(pool.map(work, range(len(rows))))

    by_category: dict[str, list[str]] = {}
    for item in out:
        if item:
            by_category.setdefault(item["category"] or "unknown", []).append(item["label"])
    print(f"\n{'slice':14} {'n':>5} {'acc':>7}")
    overall = [label for labels in by_category.values() for label in labels]
    for category, labels in sorted(by_category.items()):
        correct = sum(1 for label in labels if label == "CORRECT")
        print(f"{category:14} {len(labels):5d} {correct / max(len(labels), 1):7.3f}")
    correct_all = sum(1 for label in overall if label == "CORRECT")
    print(f"{'overall':14} {len(overall):5d} {correct_all / max(len(overall), 1):7.3f}")
    print(f"prompt_tokens={chat_answer.prompt_tokens + chat_judge.prompt_tokens} "
          f"completion_tokens={chat_answer.completion_tokens + chat_judge.completion_tokens} "
          f"calls={chat_answer.calls + chat_judge.calls} failures={chat_answer.failures + chat_judge.failures}")
    if skipped:
        print(f"budget stop: {skipped} question(s) skipped after reaching "
              f"--budget-tokens={args.budget_tokens:,}")
    spent = (chat_answer.prompt_tokens + chat_judge.prompt_tokens) / 1e6
    print(f"~{spent:.2f}M prompt tokens ≈ ¥{spent * _rate(args.answer_model) + spent * _rate(args.judge_model):.2f} "
          f"at list price (answer {args.answer_model} + judge {args.judge_model})")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "run": args.run, "answer_model": args.answer_model, "judge_model": args.judge_model,
            "summary": {
                "overall": {"n": len(overall), "acc": correct_all / max(len(overall), 1)},
                "by_category": {c: {"n": len(l), "acc": sum(1 for x in l if x == "CORRECT") / max(len(l), 1)}
                                for c, l in by_category.items()},
                "prompt_tokens": chat_answer.prompt_tokens + chat_judge.prompt_tokens,
                "completion_tokens": chat_answer.completion_tokens + chat_judge.completion_tokens,
            },
            "items": out,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
