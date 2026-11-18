#!/usr/bin/env bash
# Round 4, H15: LongMemEval-S held-out run. Own shim and ledger; shares the round-4 cache and spend cap.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/round4; mkdir -p "$OUT/logs" "$OUT/lme"
export SHIM_BACKEND=openrouter SHIM_DEFAULT_MODEL=haiku SHIM_CONCURRENCY=16 SHIM_CACHE="$OUT/logs/shim_cache.sqlite" SHIM_SPEND_CAP_USD=1000
export SHIM_LOG="$OUT/logs/shim_lme.jsonl" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
python3 -m tools.openai_shim --port 8091 > "$OUT/logs/shim_lme.log" 2>&1 &
SHIM=$!; trap 'kill $SHIM 2>/dev/null || true' EXIT
sleep 12
export OPENAI_BASE_URL="http://127.0.0.1:8091/v1" OPENAI_API_KEY=shim
python3 run_lme.py --n 120 --workers 4 --systems ${SYSTEMS:-hybrid_k20 minmem full_context full_context_hdr bm25_k20 vector_k20 hybrid_rerank minmem_nohdr}
echo "[lme] done"
