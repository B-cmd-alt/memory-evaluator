#!/usr/bin/env bash
# Round 4, amendment 3: the Claude-reader cells on the `claude -p` subscription path, one cell group at a time.
#   tools/queue_round4_cli.sh <tag> <port> "<rungs>" "<seeds>" "<systems>" [KEY=VAL...]
# Records go to results/round4_cli/ and are never pooled with results/round4/ (the OpenRouter path).
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="$1"; PORT="$2"; RUNGS="$3"; SEEDS="$4"; SYSTEMS="$5"; shift 5
export OUT=results/round4_cli
for rung in $RUNGS; do for seed in $SEEDS; do
  tools/run_round4.sh "${TAG}_s${seed}_t${rung}" "$PORT" "$seed" "$rung" "$SYSTEMS" SHIM_BACKEND=claude_cli \
    SHIM_CACHE=results/round4_cli/logs/shim_cache.sqlite SHIM_CONCURRENCY="${CONC:-4}" WORKERS="${CONC:-4}" \
    READERS="anthropic/claude-sonnet-5.5 anthropic/claude-haiku-5.5" "$@"
done; done
echo "[round4_cli] queue $TAG done"
