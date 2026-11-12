#!/usr/bin/env bash
# SixAxis at S2 (results/sixaxis_S2_seed42.json, 720 trusted memories), lean and staged. One shim, cells run in
# sequence so the shim-ledger cost windows stay exclusive and the decoy cache is written by one process at a time.
#
#   tools/run_s2.sh 1        # stage 1: baselines x3 reps, cheap-write and native-delete systems x1   (~4.5K calls)
#   tools/run_s2.sh 2        # stage 2: Graphiti and GraphRAG x1, flagged                             (~5.4K calls)
#   TIER=haiku tools/run_s2.sh 1   # the same on the second generator tier (the judge stays pinned to Sonnet)
#
# Call counts are estimates from round-2 S1 rates, not measurements; the shim ledger records the real ones.
# What is left out, and why (declare these in the write-up — they deviate from experiments/prereg_sixaxis.md):
#   * A3 for systems without a native delete (anthropic_memory_tool, mastra_om, generative_agents, ms_graphrag):
#     each deletion case rebuilds the whole index, ~13 full re-ingests per cell.
#   * A4 for ms_graphrag: every poison ingest is a full re-index.
#   * Reps 2-3 for systems that call the model at ingest: only one S2 seed exists, so extra reps repeat the same
#     data at full ingest cost. Zero-write baselines keep 3 reps (their reps are nearly free).
# Every cell resumes with --skip-existing, so a stage can be re-run after an interruption.
set -euo pipefail
STAGE="${1:?usage: tools/run_s2.sh 1|2}"
TIER="${TIER:-sonnet}"
OUT="${OUT:-results/sixaxis}"; mkdir -p "$OUT/logs"
PORT="${PORT:-8030}"
DATA="results/sixaxis_S2_seed42.json"
cd "$(dirname "$0")/.."

export TIER OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" OPENAI_API_KEY=shim SHIM_LOG="$OUT/logs/shim_${TIER}_S2.jsonl"
SHIM_DEFAULT_MODEL="$TIER" python3 -m tools.openai_shim --port "$PORT" > "$OUT/logs/shim_${TIER}_S2.log" 2>&1 &
SHIM=$!; trap 'kill "$SHIM" 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -fsS "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1 && break; sleep 1; done

cell() {  # cell <rep> <skip-axes...> -- <systems...>
  local rep="$1"; shift; local skip=()
  while [ "$1" != "--" ]; do skip+=("$1"); shift; done; shift
  echo "[s2] tier=$TIER rep=$rep skip=${skip[*]:-none} systems=$*"
  python3 run_sixaxis.py --data "$DATA" --systems "$@" --rep "$rep" --out "$OUT" --skip-existing \
    ${skip:+--skip-axes "${skip[@]}"} 2>&1 | tee -a "$OUT/logs/run_${TIER}_S2.log"
}

case "$STAGE" in
  1)
    for rep in 1 2 3; do cell "$rep" -- full_context bm25 vector_rag hybrid_rerank; done
    cell 1 -- mem0 a_mem                                              # native delete: all axes
    cell 1 A3 -- mastra_om anthropic_memory_tool generative_agents   # rebuild-on-delete: no A3
    ;;
  2)
    cell 1 -- zep_graphiti                                            # native delete (remove_episode): all axes
    cell 1 A3 A4 -- ms_graphrag                                       # re-index on delete and on poison ingest
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "[s2] stage $STAGE done; aggregate with: python3 tools/aggregate_sixaxis.py"
