#!/usr/bin/env bash
# Run on the server by the CD workflow (through AWS SSM).
#   bash scripts/server_deploy.sh v1.0.0 us-east-1
#
# 1. Downloads the production settings (written by the workflow from GitHub secrets) from
#    AWS Systems Manager Parameter Store into .env.prod, keeping the IMAGE_TAG that deploy.sh manages.
# 2. Runs scripts/deploy.sh, which pulls the image, checks health and rolls back on failure.
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."

TAG="${1:?usage: server_deploy.sh <image-tag> <aws-region>}"
REGION="${2:?usage: server_deploy.sh <image-tag> <aws-region>}"
PARAM="${ENV_PARAMETER:-/nexora/env-prod}"

aws ssm get-parameter --name "$PARAM" --with-decryption --region "$REGION" \
  --query Parameter.Value --output text > .env.prod.new

KEEP_TAG=""
if [ -f .env.prod ]; then
  KEEP_TAG="$(grep -E '^IMAGE_TAG=' .env.prod | head -1 || true)"
fi

grep -v '^IMAGE_TAG=' .env.prod.new > .env.prod
if [ -n "$KEEP_TAG" ]; then
  echo "$KEEP_TAG" >> .env.prod
fi
rm -f .env.prod.new
chmod 600 .env.prod

exec bash scripts/deploy.sh "$TAG"
