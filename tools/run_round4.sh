#!/usr/bin/env bash
# Round 4 launcher (experiments/prereg_round4.md). Each runner process gets its own shim (own port, own ledger, so cost
# windows stay exclusive); all shims share one SQLite prompt cache + spend ledger with a hard cap.
#   tools/run_round4.sh <tag> <port> <seed> "<rungs>" "<systems>" [extra env...]
# e.g. tools/run_round4.sh zw42 8061 42 "0 50000 200000 800000 3200000" "bm25 vector_rag hybrid_rerank minmem"
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="$1"; PORT="$2"; SEED="$3"; RUNGS="$4"; SYSTEMS="$5"; shift 5
OUT="${OUT:-results/round4}"; mkdir -p "$OUT/logs"
export SHIM_BACKEND=openrouter SHIM_DEFAULT_MODEL="${WRITE_MODEL:-haiku}" SHIM_CONCURRENCY="${SHIM_CONCURRENCY:-16}"
export SHIM_CACHE="$OUT/logs/shim_cache.sqlite" SHIM_SPEND_CAP_USD="${SHIM_SPEND_CAP_USD:-1000}" SHIM_LOG="$OUT/logs/shim_$TAG.jsonl"
export MASTRA_REFLECT_TOKENS=40000 GRAPHITI_READ_ARM=xenc GA_FREEZE_RECENCY=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" TOKENIZERS_PARALLELISM=false
for kv in "$@"; do export "$kv"; done
python3 -m tools.openai_shim --port "$PORT" > "$OUT/logs/shim_$TAG.log" 2>&1 &
SHIM=$!; trap 'kill $SHIM 2>/dev/null || true' EXIT
sleep 12
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" OPENAI_API_KEY=shim
for rung in $RUNGS; do
  # full context cannot be run beyond the readers' window
  sys="$SYSTEMS"; [ "$rung" -gt 900000 ] && sys="${SYSTEMS//full_context/}"
  [ -z "${sys// /}" ] && continue
  python3 run_ladder.py ${LIMIT:+--limit $LIMIT} --seed "$SEED" --rung "$rung" --systems $sys --workers "${WORKERS:-8}" --out "$OUT" ${READERS:+--readers $READERS} || echo "[round4] $TAG rung $rung exited non-zero"
done
echo "[round4] $TAG done"
