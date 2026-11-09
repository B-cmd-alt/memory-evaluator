#!/usr/bin/env bash
# SixAxis round 2 at S1 for one generator tier: the three pre-registered seeds run CONCURRENTLY, each against
# its own shim (own port, own ledger) so the per-run cost windows in run_sixaxis.py stay exclusive.
#   tools/run_round2.sh sonnet      # tier 1 (prereg: Sonnet)      tools/run_round2.sh haiku   # tier 2
# The judge is pinned to Sonnet inside run_sixaxis.py regardless of tier.
set -euo pipefail
TIER="${1:-sonnet}"
SYSTEMS="${SYSTEMS:-full_context bm25 vector_rag hybrid_rerank mastra_om generative_agents anthropic_memory_tool mem0 a_mem ms_graphrag zep_graphiti}"
OUT="${OUT:-results/sixaxis}"; mkdir -p "$OUT/logs"
PORT_BASE="${PORT_BASE:-8010}"   # a second tier run concurrently takes another port block, e.g. PORT_BASE=8020
cd "$(dirname "$0")/.."
SHIMS=(); RUNS=(); i=0
for seed in 42 123 456; do
  i=$((i + 1)); port=$((PORT_BASE + i))
  SHIM_DEFAULT_MODEL="$TIER" SHIM_LOG="$OUT/logs/shim_${TIER}_seed${seed}.jsonl" \
    python3 -m tools.openai_shim --port "$port" > "$OUT/logs/shim_${TIER}_seed${seed}.log" 2>&1 &
  SHIMS+=($!)
done
sleep 10
i=0
for seed in 42 123 456; do
  i=$((i + 1)); port=$((PORT_BASE + i))
  TIER="$TIER" OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" OPENAI_API_KEY=shim SHIM_LOG="$OUT/logs/shim_${TIER}_seed${seed}.jsonl" \
    python3 run_sixaxis.py --data "results/sixaxis_S1_seed${seed}.json" --systems $SYSTEMS --rep "$i" --out "$OUT" --skip-existing \
    > "$OUT/logs/run_${TIER}_seed${seed}.log" 2>&1 &
  RUNS+=($!)
done
echo "[round2] tier=$TIER runners=${RUNS[*]} shims=${SHIMS[*]}"
rc=0; for p in "${RUNS[@]}"; do wait "$p" || rc=1; done
kill "${SHIMS[@]}" 2>/dev/null || true
echo "[round2] tier=$TIER done rc=$rc"; exit $rc
