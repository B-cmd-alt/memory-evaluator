#!/usr/bin/env bash
# Deploy the interactive evaluator as ONE Cloud Run service (scale-to-zero, max 2 instances).
#
# Required env (no values live in this tree):
#   PROJECT   GCP project id          REGION   e.g. us-central1       SA   runtime service account
#   BUCKET    GCS bucket for votes + the daily budget ledger
#   LLM_SECRET  Secret Manager secret holding the OpenRouter key (mounted as OPENROUTER_API_KEY)
# Optional: LLM_MODEL (OpenRouter model id), DAILY_BUDGET_USD, PER_IP_DAILY_LIVE, AR_REPO
set -euo pipefail
: "${PROJECT:?}" "${REGION:?}" "${SA:?}" "${BUCKET:?}" "${LLM_SECRET:?}"
SERVICE="${SERVICE:-memory-evaluator}"
HERE="$(cd "$(dirname "$0")" && pwd)"
GIT_COMMIT="$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO:-cloud-run-source-deploy}/${SERVICE}:${GIT_COMMIT:0:10}"

echo "[build] ${IMAGE}"
TMP_CFG="$(mktemp)"; trap 'rm -f "$TMP_CFG"' EXIT
sed "s#^options:#serviceAccount: projects/${PROJECT}/serviceAccounts/${SA}\noptions:#" "${HERE}/cloudbuild.yaml" > "$TMP_CFG"
gcloud builds submit "$HERE" --project="$PROJECT" --region="$REGION" --config="$TMP_CFG" --substitutions="_IMAGE=${IMAGE}" --quiet

ENV_VARS="OPENAI_BASE_URL=https://openrouter.ai/api/v1,LLM_MODEL=${LLM_MODEL:-anthropic/claude-sonnet-5.5},EVALUATOR_BUCKET=${BUCKET}"
ENV_VARS="${ENV_VARS},DAILY_BUDGET_USD=${DAILY_BUDGET_USD:-1.00},PER_IP_DAILY_LIVE=${PER_IP_DAILY_LIVE:-12},GIT_COMMIT=${GIT_COMMIT}"
echo "[deploy] service ${SERVICE}"
# --max-instances/--cpu/--memory are mandatory per project policy; no --no-cpu-throttling.
gcloud run deploy "$SERVICE" --image="$IMAGE" --project="$PROJECT" --region="$REGION" --service-account="$SA" \
  --allow-unauthenticated --max-instances=2 --min-instances=0 --cpu=1 --memory=2Gi --concurrency=4 --timeout=180 \
  --update-env-vars="$ENV_VARS" --set-secrets="OPENROUTER_API_KEY=${LLM_SECRET}:latest" --quiet

URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT" --region="$REGION" --format='value(status.url)')"
echo "[verify] ${URL}"
curl -fsS "${URL}/health"; echo
echo "  image digest: $(gcloud artifacts docker images describe "$IMAGE" --project="$PROJECT" --format='value(image_summary.digest)')"
