#!/usr/bin/env bash
# Round 3 (experiments/prereg_round3.md): one R3 world, Sonnet tier, one replication. Four runner groups run CONCURRENTLY,
# each against its own shim (own port, own ledger -> exclusive cost windows); all shims share one SQLite prompt cache.
# GraphRAG runs A1+A2 only. Graphiti and GraphRAG are in different groups so their heavy phases do not overlap in RAM.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:-results/sixaxis_R3_seed42.json}"; OUT="${OUT:-results/round3}"; mkdir -p "$OUT/logs"
export SHIM_CACHE="$OUT/logs/shim_cache.sqlite" SHIM_DEFAULT_MODEL=sonnet SHIM_CONCURRENCY="${SHIM_CONCURRENCY:-4}"
declare -A GROUPS_=(
  [g1]="bm25 vector_rag hybrid_rerank full_context|ms_graphrag"
  [g2]="mastra_om generative_agents|"
  [g3]="mem0 anthropic_memory_tool|"
  [g4]="zep_graphiti|"
)
port=8030; SHIMS=(); RUNS=()
for g in g1 g2 g3 g4; do
  port=$((port + 1))
  SHIM_LOG="$OUT/logs/shim_$g.jsonl" python3 -m tools.openai_shim --port "$port" > "$OUT/logs/shim_$g.log" 2>&1 &
  SHIMS+=($!)
done
sleep 15
port=8030
for g in g1 g2 g3 g4; do
  port=$((port + 1)); all="${GROUPS_[$g]%%|*}"; a12="${GROUPS_[$g]#*|}"
  env="TIER=sonnet OPENAI_BASE_URL=http://127.0.0.1:${port}/v1 OPENAI_API_KEY=shim SHIM_LOG=$OUT/logs/shim_$g.jsonl"
  ( set +e
    [ -n "$all" ] && env $env python3 run_sixaxis.py --data "$DATA" --systems $all --out "$OUT" --skip-existing
    [ -n "$a12" ] && env $env AXES=A1A2 python3 run_sixaxis.py --data "$DATA" --systems $a12 --out "$OUT" --skip-existing
  ) > "$OUT/logs/run_$g.log" 2>&1 &
  RUNS+=($!)
done
echo "[round3] runners=${RUNS[*]} shims=${SHIMS[*]}"
for p in "${RUNS[@]}"; do wait "$p" || true; done
kill "${SHIMS[@]}" 2>/dev/null || true
echo "[round3] done"
