#!/usr/bin/env bash
set -euo pipefail

AUDIO_DIR="${AUDIO_DIR:-audio}"

load_env_key() {
  local key="$1"
  if [[ -n "${!key:-}" || ! -f .env ]]; then
    return
  fi

  local value
  value="$(awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' .env)"
  if [[ -n "$value" ]]; then
    export "$key=$value"
  fi
}

load_env_key R2_BUCKET
load_env_key R2_ACCOUNT_ID
load_env_key R2_ENDPOINT_URL
load_env_key AWS_PROFILE

R2_BUCKET="${R2_BUCKET:?Set R2_BUCKET to your Cloudflare R2 bucket name}"

if [[ -z "${R2_ENDPOINT_URL:-}" ]]; then
  R2_ACCOUNT_ID="${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID or R2_ENDPOINT_URL for Cloudflare R2}"
  R2_ENDPOINT_URL="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required. Install it with: brew install awscli" >&2
  exit 1
fi

profile_args=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  profile_args+=(--profile "$AWS_PROFILE")
fi

aws s3 sync "${AUDIO_DIR}/" "s3://${R2_BUCKET}/" \
  --endpoint-url "$R2_ENDPOINT_URL" \
  "${profile_args[@]}" \
  --exclude "*" \
  --include "*.mp3"
