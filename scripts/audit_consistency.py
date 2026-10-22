"""Consistency audit: config env names vs docs, docs numbers vs run artefacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # aml-memory/
issues: list[str] = []


def env_names_in_config() -> set[str]:
    text = (ROOT / "aml_memory" / "config.py").read_text(encoding="utf-8")
    return set(re.findall(r'_env(?:_int|_float|_bool)?\("([A-Z0-9_]+)"', text))


def keys_in_env_file(path: Path) -> set[str]:
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0].strip())
    return keys


def documented_keys(*paths: Path) -> set[str]:
    found: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # any UPPER_CASE token that matches the config/dotenv naming space
        found |= set(re.findall(r"\b((?:AML_|EMBED_|RERANK_|LLM_)[A-Z0-9_]+)\b", text))
    return found


cfg = env_names_in_config()
example = keys_in_env_file(ROOT / ".env.example")
env = keys_in_env_file(ROOT / ".env") if (ROOT / ".env").exists() else set()
docs = documented_keys(ROOT / "README.md", ROOT / "DEPLOY.md", ROOT / "docs" / "METHOD.md")

# Consumed by the container stack, not by the Python settings object.
DEPLOY_ONLY = {"AML_DOMAIN", "AML_MEMORY_LIMIT"}

print(f"config.py env names : {len(cfg)}")
print(f".env.example keys   : {len(example)}")
print(f".env keys (real)    : {len(env)}")

missing_in_example = sorted(cfg - example)
if missing_in_example:
    issues.append(f"env names read by config.py but absent from .env.example: {missing_in_example}")
stale_in_example = sorted(example - cfg - DEPLOY_ONLY)
if stale_in_example:
    issues.append(f".env.example declares keys config.py never reads: {stale_in_example}")
unknown_in_env = sorted(k for k in env - cfg - DEPLOY_ONLY)
if unknown_in_env:
    issues.append(f".env sets keys config.py never reads: {unknown_in_env}")
missing_in_docs = sorted(cfg - (docs | example))
if missing_in_docs:
    issues.append(f"settings never mentioned in README/DEPLOY/METHOD: {missing_in_docs}")

# --- docker wiring --------------------------------------------------------
compose = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
caddy = (ROOT / "deploy" / "Caddyfile").read_text(encoding="utf-8")
for rel in re.findall(r"^COPY\s+(\S+)", dockerfile, re.M):
    if not (ROOT / rel).exists():
        issues.append(f"Dockerfile COPY source missing: {rel}")
if "context: .." not in compose:
    issues.append("compose build context is not the repo root")
if "env_file:\n      - ../.env" not in compose:
    issues.append("compose does not load ../.env")
if "$AML_DOMAIN" not in caddy and "{$AML_DOMAIN}" not in caddy:
    issues.append("Caddyfile does not use AML_DOMAIN")
if "/health" not in compose:
    issues.append("compose healthcheck does not hit /health")

# --- numbers in the docs vs the run artefacts -----------------------------
RUN_DIRS = [ROOT / "runs", ROOT.parent / "runs"]


def metric(name: str) -> float | None:
    """Top-line number of a run artefact: R@1 for retrieval, accuracy for a proxy run."""
    for directory in RUN_DIRS:
        path = directory / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary") or data
        overall = summary.get("overall") if isinstance(summary, dict) else None
        if not isinstance(overall, dict):
            overall = summary if isinstance(summary, dict) else {}
        for key in ("R@1", "acc", "accuracy"):
            if key in overall:
                return float(overall[key])
    return None


checks = [
    ("final_default.json", ".748"),
    ("answer_widen_full.json", ".675"),
    ("answer_nowiden_full.json", ".640"),
    ("lme_oracle.json", ".464"),
]
readme = (ROOT / "README.md").read_text(encoding="utf-8")
for fname, claimed in checks:
    value = metric(fname)
    if value is None:
        issues.append(f"runs/{fname} missing")
        continue
    if abs(value - float(claimed)) > 0.0011:
        issues.append(f"runs/{fname} top-line is {value:.3f}, docs claim {claimed}")
    if claimed not in readme:
        issues.append(f"{claimed} (from runs/{fname}) no longer appears in README.md")

print("\n".join(f"  - {item}" for item in issues) if issues else "no inconsistencies found")
print(f"\n{len(issues)} issue(s)")
