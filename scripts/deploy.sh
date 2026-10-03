#!/usr/bin/env bash
# Deploy a released image tag, or roll back to an older one. Run on the server.
#   ./scripts/deploy.sh v1.0.0      deploy (or roll back to) tag v1.0.0
#
# If the new API does not become healthy, the script switches back to the previous tag.
set -euo pipefail
cd "$(dirname "$0")/.."

NEW_TAG="${1:?usage: ./scripts/deploy.sh <image-tag>   (for example v1.0.0)}"
COMPOSE="docker compose --env-file .env.prod -f docker-compose.prod.yml"

current_tag() { grep -E '^IMAGE_TAG=' .env.prod | head -1 | cut -d= -f2- | tr -d '"' || true; }
set_tag() {
  if grep -qE '^IMAGE_TAG=' .env.prod; then
    sed -i "s|^IMAGE_TAG=.*|IMAGE_TAG=$1|" .env.prod
  else
    echo "IMAGE_TAG=$1" >> .env.prod
  fi
}
api_healthy() {
  local id; id="$($COMPOSE ps -q api)"
  [ -n "$id" ] && [ "$(docker inspect --format '{{.State.Health.Status}}' "$id" 2>/dev/null)" = "healthy" ]
}
wait_healthy() {
  # The first start downloads the models, so allow up to 10 minutes.
  for _ in $(seq 1 120); do
    if api_healthy; then return 0; fi
    sleep 5
  done
  return 1
}

PREVIOUS_TAG="$(current_tag)"
echo "Current tag: ${PREVIOUS_TAG:-none}   New tag: $NEW_TAG"

set_tag "$NEW_TAG"
# Log in to Amazon ECR when the image comes from there (uses the EC2 instance role, no keys needed).
IMAGE_REPO="$(grep -E '^IMAGE=' .env.prod | head -1 | cut -d= -f2- | tr -d '"' || true)"
if [[ "$IMAGE_REPO" == *.amazonaws.com/* ]]; then
  ECR_HOST="${IMAGE_REPO%%/*}"
  ECR_REGION="$(echo "$ECR_HOST" | cut -d. -f4)"
  aws ecr get-login-password --region "$ECR_REGION" | docker login --username AWS --password-stdin "$ECR_HOST"
fi
$COMPOSE pull api
# "|| true": a slow first start must not stop the script before the health check and rollback below.
$COMPOSE up -d || echo "[warn] 'compose up' reported a problem, checking the API health anyway"

if wait_healthy; then
  echo "[ok] $NEW_TAG is healthy"
  $COMPOSE ps
  exit 0
fi

echo "[error] $NEW_TAG did not become healthy. Last API logs:"
$COMPOSE logs --tail 40 api || true

if [ -n "$PREVIOUS_TAG" ] && [ "$PREVIOUS_TAG" != "$NEW_TAG" ]; then
  echo "Rolling back to $PREVIOUS_TAG"
  set_tag "$PREVIOUS_TAG"
  $COMPOSE up -d || true
  if wait_healthy; then echo "[ok] rolled back to $PREVIOUS_TAG"; else echo "[error] rollback also unhealthy, check the logs"; fi
fi
exit 1
