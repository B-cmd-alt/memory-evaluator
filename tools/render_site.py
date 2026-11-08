#!/usr/bin/env python3
"""Render the Memory System Evaluator site from results/feasibility/*.json (and later round-2 results).

One self-contained HTML file, no framework, no CDN; a few lines of inline JS only for filtering.
Pages (sections): System profiles · Lineage · Summary table · Case explorer (every question, every
system side by side: retrieved context → answer → judge verdict + reasoning → cost/latency).

    python tools/render_site.py            # -> site/index.html
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.systems.registry import REGISTRY  # noqa: E402

RES = Path("results/feasibility")
OUT = Path("site/index.html")
TASKS = {"T1": "single fact", "T2": "multi-session", "T3": "temporal", "T4": "conflict/update", "T5": "abstention"}

CSS = """
:root{--bg:#fbfbfa;--fg:#1c1c1c;--muted:#666;--card:#fff;--line:#e4e2dd;--ok:#1b7f3b;--bad:#b3261e;--mid:#9a6700;--code:#f3f2ee;--acc:#2a5db0}
@media(prefers-color-scheme:dark){:root{--bg:#121212;--fg:#ececec;--muted:#9a9a9a;--card:#1b1b1b;--line:#2c2c2c;--ok:#5fd38d;--bad:#ff6b63;--mid:#ffc53d;--code:#202020;--acc:#7aa7ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:28px 16px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}h3{margin:0 0 4px;font-size:16px}
.sub{color:var(--muted);margin:0 0 18px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.card small{color:var(--muted)}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;border:1px solid var(--line);color:var(--muted);margin-right:4px}
.big{font-size:30px;font-weight:700;line-height:1.1}.ok{color:var(--ok)}.bad{color:var(--bad)}.mid{color:var(--mid)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--bg)}
.bars{display:flex;gap:3px;margin:6px 0}.bar{flex:1;height:26px;border-radius:4px;background:var(--code);position:relative;overflow:hidden}.bar i{position:absolute;bottom:0;left:0;right:0;background:var(--acc);opacity:.75}.bar b{position:absolute;left:0;right:0;top:4px;text-align:center;font-size:10px;color:var(--fg)}
details{margin-top:6px}summary{cursor:pointer;color:var(--muted);font-size:13px}pre{background:var(--code);border-radius:8px;padding:10px;overflow:auto;font-size:12px;line-height:1.45;margin:6px 0;white-space:pre-wrap}
.q{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}.q .ref{color:var(--muted)}
.cols{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin-top:10px}.sys{border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.sys b.name{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}.ans{margin:4px 0;font-size:14px}
.chip{display:inline-block;font-weight:700;padding:0 8px;border-radius:999px;font-size:12px;border:1px solid}.chip.ok{border-color:var(--ok)}.chip.bad{border-color:var(--bad)}.chip.mid{border-color:var(--mid)}
.lineage{display:flex;gap:8px;overflow-x:auto;padding:6px 0}.node{min-width:220px;border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:var(--card)}.node .yr{font-size:12px;color:var(--muted)}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0;font-size:13px}.filters label{border:1px solid var(--line);border-radius:999px;padding:2px 10px;cursor:pointer}
footer{color:var(--muted);font-size:13px;margin-top:48px;border-top:1px solid var(--line);padding-top:16px}
"""

JS = """
function applyFilters(){
  const task=document.querySelector('input[name=task]:checked').value;
  const onlyDis=document.getElementById('f-dis').checked;
  document.querySelectorAll('.q').forEach(q=>{
    let show=(task==='all'||q.dataset.task===task);
    if(show&&onlyDis){show=q.dataset.disagree==='1';}
    q.style.display=show?'':'none';
  });
}
document.addEventListener('change',e=>{if(e.target.closest('.filters'))applyFilters();});
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def load():
    rows = {}
    for p in sorted(RES.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(d, dict) and "system" in d:
            rows[d["system"]] = d
    return rows


def chip(score):
    if score is None:
        return "<span class='chip mid'>—</span>"
    cls = "ok" if score >= 0.99 else ("mid" if score > 0 else "bad")
    return f"<span class='chip {cls}'>{'✓' if cls == 'ok' else ('½' if cls == 'mid' else '✗')} {score:.2f}</span>"


def profile_card(e, r) -> str:
    tags = f"<span class='tag'>{esc(e.kind)}</span><span class='tag'>{esc(e.year)}</span><span class='tag'>{esc(e.provenance)}</span>"
    head = f"<h3>{esc(e.label)}</h3><small>{esc(e.family)} · lineage: {esc(e.lineage)}</small><div style='margin:6px 0'>{tags}</div>"
    if not r:
        return f"<div class='card'>{head}<small>not run yet</small></div>"
    if r["status"] in ("BLOCKED", "FAILED_INGEST"):
        return f"<div class='card'>{head}<div class='big bad'>{esc(r['status'])}</div><small>{esc(r.get('error'))}</small></div>"
    q, ing = r["query"], r["ingest"]
    bars = "".join(f"<div class='bar' title='{TASKS[t]}'><i style='height:{int(100 * q['by_task'].get(t, 0))}%'></i><b>{t} {q['by_task'].get(t, 0):.1f}</b></div>" for t in TASKS)
    qcost = sum(x["read_path"]["cost_usd"] for x in r["questions"])
    return (f"<div class='card'>{head}<div class='big'>{q['accuracy']:.2f}</div><small>accuracy, S1, strict-judge v3 · {esc(r['status'])}</small>"
            f"<div class='bars'>{bars}</div>"
            f"<small>write path: <b>{ing['llm_calls_per_memory']}</b> LLM calls/memory, {ing['seconds']}s, ${ing['cost_usd']}<br>"
            f"read path: <b>{q['read_llm_calls_per_query']}</b> calls/q, <b>{q['read_prompt_tokens_per_query']}</b> prompt tok/q, "
            f"p50 {q['latency_p50_s']}s · p95 {q['latency_p95_s']}s · ${qcost:.3f}/25 q</small></div>")


def lineage_section(rows) -> str:
    lines = {}
    for e in REGISTRY:
        lines.setdefault(e.lineage, []).append(e)
    out = []
    for name, es in lines.items():
        es = sorted(es, key=lambda e: e.year)
        nodes = []
        for e in es:
            r = rows.get(e.key)
            acc = f"{r['query']['accuracy']:.2f}" if r and r.get("query") else (r["status"] if r else "—")
            nodes.append(f"<div class='node'><div class='yr'>{e.year}</div><b>{esc(e.label)}</b><div><small>{esc(e.family)}</small></div><div class='big' style='font-size:22px'>{esc(acc)}</div></div>")
        out.append(f"<h3 style='margin-top:14px'>{esc(name)}</h3><div class='lineage'>{'<div style=&quot;align-self:center;color:var(--muted)&quot;>→</div>'.join(nodes)}</div>")
    return "".join(out)


def summary_table(rows) -> str:
    L = ["<table><tr><th>system</th><th>kind</th><th>status</th><th>acc</th>" + "".join(f"<th>{t}</th>" for t in TASKS) +
         "<th>ingest calls/mem</th><th>ingest s</th><th>read calls/q</th><th>prompt tok/q</th><th>p50 s</th><th>p95 s</th><th>$ ingest</th><th>$ 25 q</th></tr>"]
    for e in REGISTRY:
        r = rows.get(e.key)
        if not r:
            L.append(f"<tr><td>{esc(e.label)}</td><td>{esc(e.kind)}</td><td>not run</td><td colspan=13></td></tr>"); continue
        if r["status"] in ("BLOCKED", "FAILED_INGEST"):
            L.append(f"<tr><td>{esc(e.label)}</td><td>{esc(e.kind)}</td><td class='bad'>{esc(r['status'])}</td><td colspan=13><small>{esc(r.get('error'))}</small></td></tr>"); continue
        q, ing = r["query"], r["ingest"]
        qcost = sum(x["read_path"]["cost_usd"] for x in r["questions"])
        L.append(f"<tr><td>{esc(e.label)}</td><td>{esc(e.kind)}</td><td>{esc(r['status'])}</td><td><b>{q['accuracy']:.2f}</b></td>" +
                 "".join(f"<td>{q['by_task'].get(t, 0):.1f}</td>" for t in TASKS) +
                 f"<td>{ing['llm_calls_per_memory']}</td><td>{ing['seconds']}</td><td>{q['read_llm_calls_per_query']}</td><td>{q['read_prompt_tokens_per_query']}</td>"
                 f"<td>{q['latency_p50_s']}</td><td>{q['latency_p95_s']}</td><td>{ing['cost_usd']}</td><td>{qcost:.3f}</td></tr>")
    return "".join(L) + "</table>"


def case_explorer(rows) -> str:
    ran = [e for e in REGISTRY if e.key in rows and rows[e.key].get("questions")]
    if not ran:
        return "<p class='sub'>no per-question records yet</p>"
    qids = [q["id"] for q in rows[ran[0].key]["questions"]]
    byq = {e.key: {q["id"]: q for q in rows[e.key]["questions"]} for e in ran}
    filters = ("<div class='filters'><label><input type=radio name=task value=all checked> all tasks</label>" +
               "".join(f"<label><input type=radio name=task value={t}> {t} {TASKS[t]}</label>" for t in TASKS) +
               "<label><input type=checkbox id=f-dis> only where systems disagree</label></div>")
    out = [filters]
    for qid in qids:
        q0 = next(byq[e.key][qid] for e in ran if qid in byq[e.key])
        scores = [byq[e.key][qid]["score"] for e in ran if qid in byq[e.key]]
        disagree = "1" if len({round(s) for s in scores}) > 1 else "0"
        cols = []
        for e in ran:
            r = byq[e.key].get(qid)
            if not r:
                continue
            retrieved = "\n".join(f"[{esc(it['source'])}] {esc(it['content'])}" for it in r["retrieved"]) or "(nothing retrieved)"
            rp = r["read_path"]
            cols.append(f"<div class='sys'><b class='name'>{esc(e.label)}</b>{chip(r['score'])} <small>{esc(r['tier'])} · {r['latency_s']}s · {rp['llm_calls']} calls · {rp['prompt_tokens']} tok</small>"
                        f"<div class='ans'>{esc(r['answer'][:600])}</div>"
                        f"<details><summary>retrieved context ({len(r['retrieved'])})</summary><pre>{retrieved}</pre></details>"
                        f"<details><summary>judge: {esc(r['judge_mode'])}</summary><pre>{esc(r['judge_reasoning'])}</pre></details></div>")
        out.append(f"<section class='q' data-task='{esc(q0['task'])}' data-disagree='{disagree}'><b>{esc(qid)} · {esc(q0['task'])} {TASKS.get(q0['task'], '')}</b>"
                   f"<div>{esc(q0['question'])}</div><div class='ref'>reference: {esc(q0['reference'])}</div><div class='cols'>{''.join(cols)}</div></section>")
    return "".join(out)


def render() -> Path:
    rows = load()
    n_ok = sum(1 for r in rows.values() if r["status"] in ("OK", "PARTIAL"))
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Memory System Evaluator</title><style>{CSS}</style></head><body><main>",
             "<h1>Memory System Evaluator</h1><p class='sub'>Ten memory systems and two baselines on one model, one embedder, one answer prompt, one calibrated judge. "
             "Every number below links to the question, the retrieved context, the answer and the judge's reasoning that produced it. "
             f"Round 1 (feasibility, S1: 90 memories, 25 questions) — {n_ok}/{len(REGISTRY)} systems ran.</p>",
             "<h2>System profiles</h2><div class='grid'>" + "".join(profile_card(e, rows.get(e.key)) for e in REGISTRY) + "</div>",
             "<h2>Lineage — 2023 → 2026, same paradigm, same questions</h2>" + lineage_section(rows),
             "<h2>Summary</h2>" + summary_table(rows) +
             "<p class='sub'>S1 has a ~7pp inter-run noise envelope (RESULTS_SUMMARY §1): differences inside it are ties. "
             "Round 1 decides feasibility and cost class, not ranking; round 2 (SixAxis, 3 replications, 2 model tiers) ranks.</p>",
             "<h2>Case explorer</h2><p class='sub'>Same question, every system side by side. Open the context and judge panels to see why a verdict was given.</p>" + case_explorer(rows),
             "<footer>Generator: Claude Sonnet via <code>tools/openai_shim.py</code> · embedder bge-small-en-v1.5 · judge strict-v3 (METHODOLOGY.md). "
             "Limitations: human-proxy judge calibration, Claude-family judge only, Letta blocked in this environment, Mastra OM and Generative Agents are faithful re-implementations.</footer>",
             f"</main><script>{JS}</script></body></html>"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts), encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(render())
