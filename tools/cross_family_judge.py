#!/usr/bin/env python3
"""Cross-family judge check: run strict-judge v3 with a NON-Claude model in the LLM branches and measure agreement
with the human-proxy labels and with the Sonnet judge.

    python -m tools.cross_family_judge --judges claude:sonnet or:openai/gpt-6.1-terra or:google/gemini-3.8-flash

Backends: `claude:<alias>` runs `claude -p` (the repo's subscription path); `or:<model>` calls OpenRouter with the key
in $openrouter_api (or $OPENROUTER_API_KEY). Temperature 0, JSON reply. Writes per-item scores and a summary to
results/judge_calibration/cross_family/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from src.evaluation.strict_judge import strict_judge_v3
from tools.kappa import kappa_binary, kappa_weighted

CAL = Path("results/judge_calibration")
OUT = CAL / "cross_family"
SYS = "You are a strict, careful answer judge. Reply with JSON only."
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def _parse(t: str) -> dict:
    t = _FENCE.sub("", (t or "").strip())
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0) if m else t)
    except ValueError:
        return {"score": 0.0, "reasoning": f"unparseable: {t[:200]}"}


def make_judge(spec: str, ledger: list):
    kind, model = spec.split(":", 1)

    def claude(prompt: str) -> dict:
        r = subprocess.run(["claude", "-p", "--model", model, "--max-turns", "1", "--output-format", "json",
                            "--append-system-prompt", SYS, prompt], capture_output=True, text=True, timeout=300)
        d = json.loads(r.stdout)
        ledger.append({"cost_usd": d.get("total_cost_usd")})
        return _parse(d.get("result", ""))

    def openrouter(prompt: str) -> dict:
        key = os.environ.get("openrouter_api") or os.environ["OPENROUTER_API_KEY"]
        body = {"model": model, "temperature": 0, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}, "usage": {"include": True}}
        for attempt in range(4):
            try:
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                d = json.loads(urllib.request.urlopen(req, timeout=180).read())
                ledger.append({"cost_usd": (d.get("usage") or {}).get("cost")})
                return _parse(d["choices"][0]["message"]["content"])
            except Exception as ex:  # noqa: BLE001
                if attempt == 3:
                    return {"score": 0.0, "reasoning": f"error: {ex}"[:200], "error": True}
                time.sleep(2 * (attempt + 1))

    return claude if kind == "claude" else openrouter


def run(spec: str, items: list[dict], workers: int) -> tuple[dict, float]:
    ledger: list = []
    fn = make_judge(spec, ledger)

    def one(x):
        s, why, mode = strict_judge_v3(x["question"], x["answer"], x["reference"], fn)
        return x["key"], {"score": s, "mode": mode, "reasoning": why[:300]}

    with cf.ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, items))
    return res, round(sum(r["cost_usd"] or 0 for r in ledger), 4)


def boot_ci(a, b, n=2000, seed=0):
    import random
    r, k, v = random.Random(seed), len(a), []
    for _ in range(n):
        ix = [r.randrange(k) for _ in range(k)]
        v.append(kappa_binary([a[i] for i in ix], [b[i] for i in ix]))
    v.sort()
    return [round(v[int(0.025 * n)], 3), round(v[int(0.975 * n) - 1], 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    # question ids repeat across architectures (100 items, 71 ids): key by position, labels are aligned by position
    items = json.load(open(CAL / "calibration_sample.json"))
    l1 = json.load(open(CAL / "proxy_human_labels.json")); l2 = json.load(open(CAL / "proxy_human_labels_pass2.json"))
    assert all(a["id"] == b["id"] == c["id"] for a, b, c in zip(items, l1, l2))
    for n, x in enumerate(items):
        x["key"] = f"{n:03d}:{x['source']}:{x['id']}"
    ids = [x["key"] for x in items]
    avg = [(a["proxy_score"] + b["proxy_score_2"]) / 2 for a, b in zip(l1, l2)]
    OUT.mkdir(parents=True, exist_ok=True)
    scores, summary = {}, []
    for spec in a.judges:
        f = OUT / f"{spec.replace(':', '__').replace('/', '_')}.json"
        if f.exists():
            d = json.loads(f.read_text()); res, usd = d["items"], d["cost_usd"]
        else:
            res, usd = run(spec, items, a.workers)
            f.write_text(json.dumps({"judge": spec, "cost_usd": usd, "items": res}, indent=1))
        s = [res[i]["score"] for i in ids]
        scores[spec] = s
        summary.append({"comparison": f"{spec} (strict v3) vs proxy_avg", "kappa_binary": round(kappa_binary(s, avg), 3),
                        "kappa_weighted": round(kappa_weighted(s, avg), 3), "mean_judge": round(sum(s) / len(s), 3),
                        "mean_proxy": round(sum(avg) / len(avg), 3), "n_llm_calls": sum(res[i]["mode"] in ("normal_judge", "ref_abstention_judge") for i in ids),
                        "n_errors": sum("error:" in res[i]["reasoning"] for i in ids), "cost_usd": usd})
    specs = list(scores)
    for i, x in enumerate(specs):
        for y in specs[i + 1:]:
            summary.append({"comparison": f"{x} vs {y}", "kappa_binary": round(kappa_binary(scores[x], scores[y]), 3),
                            "kappa_weighted": round(kappa_weighted(scores[x], scores[y]), 3)})
    # agreement on the items that actually reach an LLM branch (deterministic branches agree by construction)
    first = json.loads((OUT / f"{specs[0].replace(':', '__').replace('/', '_')}.json").read_text())["items"]
    llm = [n for n, i in enumerate(ids) if first[i]["mode"] in ("normal_judge", "ref_abstention_judge")]
    pick = lambda xs: [xs[n] for n in llm]  # noqa: E731
    for x in specs:
        summary.append({"comparison": f"[LLM-branch only, n={len(llm)}] {x} vs proxy_avg", "kappa_binary": round(kappa_binary(pick(scores[x]), pick(avg)), 3),
                        "kappa_binary_ci95": boot_ci(pick(scores[x]), pick(avg)),
                        "judge_pass_proxy_fail": sum(scores[x][n] >= .5 > avg[n] for n in llm),
                        "judge_fail_proxy_pass": sum(avg[n] >= .5 > scores[x][n] for n in llm)})
    for i, x in enumerate(specs):
        for y in specs[i + 1:]:
            summary.append({"comparison": f"[LLM-branch only, n={len(llm)}] {x} vs {y}", "kappa_binary": round(kappa_binary(pick(scores[x]), pick(scores[y])), 3),
                            "kappa_binary_ci95": boot_ci(pick(scores[x]), pick(scores[y])),
                            "disagreements": [ids[n] for n in llm if (scores[x][n] >= .5) != (scores[y][n] >= .5)]})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
