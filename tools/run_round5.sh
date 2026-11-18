#!/usr/bin/env bash
# Round 5 launcher (experiments/prereg_round5.md, Part A). Own shim, own ledger; shares round 4's prompt cache and spend cap.
#   tools/run_round5.sh <tag> <port> <seed> "<arms>" [extra env...]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="$1"; PORT="$2"; SEED="$3"; ARMS="$4"; shift 4
OUT="${OUT:-results/round5}"; mkdir -p "$OUT/logs"
export SHIM_BACKEND=openrouter SHIM_DEFAULT_MODEL="${WRITE_MODEL:-haiku}" SHIM_CONCURRENCY="${SHIM_CONCURRENCY:-16}"
export SHIM_CACHE="results/round4/logs/shim_cache.sqlite" SHIM_SPEND_CAP_USD="${SHIM_SPEND_CAP_USD:-1000}" SHIM_LOG="$OUT/logs/shim_$TAG.jsonl"
export MASTRA_REFLECT_TOKENS=40000 GRAPHITI_READ_ARM=xenc GA_FREEZE_RECENCY=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" TOKENIZERS_PARALLELISM=false
for kv in "$@"; do export "$kv"; done
python3 -m tools.openai_shim --port "$PORT" > "$OUT/logs/shim_$TAG.log" 2>&1 &
SHIM=$!; trap 'kill $SHIM 2>/dev/null || true' EXIT
sleep 12
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" OPENAI_API_KEY=shim
python3 run_residue.py --seed "$SEED" --arms $ARMS --out "$OUT" || echo "[round5] $TAG exited non-zero"
echo "[round5] $TAG done"
