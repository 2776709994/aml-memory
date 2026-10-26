"""Multiple-choice scoring for PersonaMem-shaped questions.

The open-ended judge in `answer.py` is the wrong instrument here: the platform compares
a single option letter, so the only thing that matters is whether the reader, given our
returned memories, picks the gold option.  This harness asks exactly that and scores it
exactly the way the official pipeline does (`<final_answer>` marker, first letter).

It exists to answer one retrieval question offline: does searching with the question
text alone, or with the question plus its option texts, deliver better evidence?  The
absolute accuracy is a proxy — the official reader is a different model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aml_memory.config import load_settings

from .answer import Chat

OFFICIAL_INSTRUCTION = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token <final_answer>."
)

TEMPLATE = """You are given a list of memories retrieved from this user's history.
Use them to decide which response is the most appropriate one for the user.

<memories>
{memories}
</memories>

{question}

{instruction}

{options}
"""


def render(question: str, options: list[str], contents: list[str], budget_chars: int) -> str:
    body = []
    spent = 0
    for content in contents:
        if spent + len(content) > budget_chars and body:
            break
        body.append(content)
        spent += len(content)
    option_text = "\n".join(options)
    return TEMPLATE.format(
        memories="\n\n".join(body),
        question=question,
        instruction=OFFICIAL_INSTRUCTION,
        options=option_text,
    )


def parse_choice(text: str) -> str:
    body = str(text or "")
    if "<final_answer>" in body:
        body = body.split("<final_answer>")[-1]
    body = body.replace("</final_answer>", " ")
    in_parens = re.findall(r"\(([a-dA-D])\)", body)
    if in_parens:
        return in_parens[0].lower()
    standalone = re.findall(r"\b([a-dA-D])\b", body)
    return standalone[0].lower() if standalone else ""


def gold_letter(answer: str) -> str:
    return str(answer).strip().strip("() ").lower()[:1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score PersonaMem MCQ answers from a runner dump.")
    parser.add_argument("--run", required=True, help="runner JSON produced with --keep-content")
    parser.add_argument("--sample", type=int, default=0, help="score at most N questions (0 = all)")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--budget-chars", type=int, default=320_000)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--out", default="runs/mcq.json")
    args = parser.parse_args()

    settings = load_settings()
    base_url = os.environ.get("PROXY_BASE_URL") or settings.embed_base_url
    api_key = os.environ.get("PROXY_API_KEY") or settings.embed_api_key
    model = args.answer_model or os.environ.get("PROXY_ANSWER_MODEL") or "deepseek-v4.1-flash"
    chat = Chat(base=base_url, key=api_key, model=model, retries=6)

    payload = json.loads(Path(args.run).read_text(encoding="utf-8"))
    rows = payload.get("results") or payload.get("items") or payload.get("rows") or []
    if args.sample:
        rows = rows[: args.sample]
    print(f"scoring {len(rows)} questions with {model} @ {base_url}", flush=True)

    def work(row: dict) -> dict:
        prompt = render(
            row.get("question", ""),
            row.get("options") or [],
            row.get("contents") or [],
            args.budget_chars,
        )
        try:
            # 512, not 64: reasoning-style models spend their first tokens thinking and
            # return an empty `content` if the cap cuts them off before the letter.
            reply = chat("Answer the multiple-choice question.", prompt, max_tokens=512)
            choice = parse_choice(reply)
            error = None
        except Exception as exc:  # noqa: BLE001
            reply, choice, error = "", "", str(exc)
        return {
            "sample_id": row.get("sample_id"),
            "question": row.get("question"),
            "category": row.get("category"),
            "gold": gold_letter(row.get("answer") or row.get("gold") or ""),
            "choice": choice,
            "reply": reply[:200],
            "correct": bool(choice) and choice == gold_letter(row.get("answer") or row.get("gold") or ""),
            "error": error,
            "prompt_chars": len(prompt),
        }

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        scored = list(pool.map(work, rows))

    good = [row for row in scored if not row["error"]]
    correct = sum(1 for row in good if row["correct"])
    print(f"\nslice              n  acc")
    per_category: dict[str, list[bool]] = {}
    for row in good:
        per_category.setdefault(row["category"] or "-", []).append(row["correct"])
    for name, hits in sorted(per_category.items(), key=lambda item: -len(item[1])):
        print(f"{name:<18}{len(hits):>4}  {sum(hits) / len(hits):.3f}")
    if good:
        print(f"{'overall':<18}{len(good):>4}  {correct / len(good):.3f}")
    failures = len(scored) - len(good)
    print(f"failures={failures} prompt_chars={sum(r['prompt_chars'] for r in scored):,}")
    if failures:
        print("first failure:", next(r["error"] for r in scored if r["error"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"run": args.run, "answer_model": model, "items": scored},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
