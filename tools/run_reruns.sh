#!/usr/bin/env bash
# Re-run the affected axes of round 2 (see tools/rerun_axes.py docstring) for one tier: A4 for every system on
# the regenerated poison cases, A3+A4 for Graphiti (native deletion). Light systems (no write cost) and heavy
# systems (Mem0 / A-MEM / GraphRAG / Graphiti, one ingest each) run in two processes against two shims so that
# at most two graph indexes are in memory per tier (six at once OOM-killed runners on the 4-core box).
#   tools/run_reruns.sh sonnet            # PORT_BASE 8010 (shims 8011/8012)
#   PORT_BASE=8020 tools/run_reruns.sh haiku
set -euo pipefail
TIER="${1:-sonnet}"
PORT_BASE="${PORT_BASE:-8010}"
OUT="${OUT:-results/sixaxis}"; mkdir -p "$OUT/logs"
cd "$(dirname "$0")/.."
LIGHT="full_context bm25 vector_rag hybrid_rerank mastra_om generative_agents anthropic_memory_tool"
HEAVY="mem0 a_mem ms_graphrag"
SHIMS=(); RUNS=()
for i in 1 2; do
  port=$((PORT_BASE + i))
  SHIM_DEFAULT_MODEL="$TIER" SHIM_LOG="$OUT/logs/shim_rerun_${TIER}_$i.jsonl" \
    python3 -m tools.openai_shim --port "$port" > "$OUT/logs/shim_rerun_${TIER}_$i.log" 2>&1 &
  SHIMS+=($!)
done
sleep 10
TIER="$TIER" OPENAI_BASE_URL="http://127.0.0.1:$((PORT_BASE + 1))/v1" OPENAI_API_KEY=shim SHIM_LOG="$OUT/logs/shim_rerun_${TIER}_1.jsonl" \
  python3 tools/rerun_axes.py --axes A4 --systems $LIGHT --tier "$TIER" > "$OUT/logs/rerun_${TIER}_light.log" 2>&1 &
RUNS+=($!)
( TIER="$TIER" OPENAI_BASE_URL="http://127.0.0.1:$((PORT_BASE + 2))/v1" OPENAI_API_KEY=shim SHIM_LOG="$OUT/logs/shim_rerun_${TIER}_2.jsonl" \
    python3 tools/rerun_axes.py --axes A4 --systems $HEAVY --tier "$TIER" && \
  TIER="$TIER" OPENAI_BASE_URL="http://127.0.0.1:$((PORT_BASE + 2))/v1" OPENAI_API_KEY=shim SHIM_LOG="$OUT/logs/shim_rerun_${TIER}_2.jsonl" \
    python3 tools/rerun_axes.py --axes A3 A4 --systems zep_graphiti --tier "$TIER" ) > "$OUT/logs/rerun_${TIER}_heavy.log" 2>&1 &
RUNS+=($!)
echo "[rerun] tier=$TIER runners=${RUNS[*]} shims=${SHIMS[*]}"
rc=0; for p in "${RUNS[@]}"; do wait "$p" || rc=1; done
kill "${SHIMS[@]}" 2>/dev/null || true
echo "[rerun] tier=$TIER done rc=$rc"; exit $rc
