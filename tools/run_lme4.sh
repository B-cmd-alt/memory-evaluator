#!/usr/bin/env bash
# Round 4, H15: LongMemEval-S held-out run. Own shim and ledger.
#   tools/run_lme4.sh                      # OpenRouter path -> results/round4/lme (shares the round-4 cache and spend cap)
#   OUT=results/round4_cli tools/run_lme4.sh SHIM_BACKEND=claude_cli SHIM_CONCURRENCY=4 WORKERS=4 \
#       READERS="anthropic/claude-sonnet-5.5 anthropic/claude-haiku-5.5" SYSTEMS="hybrid_k20 minmem ..."   # amendment 3
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-results/round4}"; mkdir -p "$OUT/logs" "$OUT/lme"
export SHIM_BACKEND=openrouter SHIM_DEFAULT_MODEL=haiku SHIM_CONCURRENCY=16 SHIM_CACHE="$OUT/logs/shim_cache.sqlite" SHIM_SPEND_CAP_USD=1000
export SHIM_LOG="$OUT/logs/shim_lme.jsonl" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
for kv in "$@"; do export "$kv"; done
python3 -m tools.openai_shim --port "${PORT:-8091}" > "$OUT/logs/shim_lme.log" 2>&1 &
SHIM=$!; trap 'kill $SHIM 2>/dev/null || true' EXIT
sleep 12
export OPENAI_BASE_URL="http://127.0.0.1:${PORT:-8091}/v1" OPENAI_API_KEY=shim
python3 run_lme.py --n 120 --workers "${WORKERS:-4}" --out "$OUT/lme" ${READERS:+--readers $READERS} \
  --systems ${SYSTEMS:-hybrid_k20 minmem full_context full_context_hdr bm25_k20 vector_k20 hybrid_rerank minmem_nohdr}
echo "[lme] done"
