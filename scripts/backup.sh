#!/usr/bin/env bash
# Backup Postgres (dump) and the Qdrant collection (snapshot). Run on the server, for example nightly:
#   0 3 * * *  cd /opt/nexora-rag && ./scripts/backup.sh >> backups/backup.log 2>&1
#
# The vector index can always be rebuilt from the repository (data/processed + ingestion scripts),
# so the Postgres dump (users, chats, feedback) is the important one.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKUP_DIR="${BACKUP_DIR:-./backups}"
KEEP_DAYS="${KEEP_DAYS:-7}"
COMPOSE="docker compose --env-file .env.prod -f docker-compose.prod.yml"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Read one value from .env.prod without sourcing the whole file.
env_get() { grep -E "^$1=" .env.prod | head -1 | cut -d= -f2- | tr -d '"' || true; }

# 1. Postgres
$COMPOSE exec -T postgres pg_dump -U postgres nexora | gzip > "$BACKUP_DIR/postgres-$STAMP.sql.gz"
echo "[ok] postgres dump: $BACKUP_DIR/postgres-$STAMP.sql.gz"

# 2. Qdrant snapshot (works for a Qdrant server; may be refused by a free cloud cluster, which is not fatal)
QDRANT_URL="$(env_get RAG_QDRANT_URL)"
QDRANT_KEY="$(env_get RAG_QDRANT_API_KEY)"
COLLECTION="$(env_get RAG_COLLECTION)"; COLLECTION="${COLLECTION:-nexora_kb}"

if [ -n "$QDRANT_URL" ]; then
  if SNAP_JSON="$(curl -fsS -X POST -H "api-key: $QDRANT_KEY" "${QDRANT_URL%/}/collections/$COLLECTION/snapshots")"; then
    SNAP_NAME="$(printf '%s' "$SNAP_JSON" | python3 -c 'import sys, json; print(json.load(sys.stdin)["result"]["name"])')"
    curl -fsS -H "api-key: $QDRANT_KEY" -o "$BACKUP_DIR/qdrant-$SNAP_NAME" \
      "${QDRANT_URL%/}/collections/$COLLECTION/snapshots/$SNAP_NAME"
    echo "[ok] qdrant snapshot: $BACKUP_DIR/qdrant-$SNAP_NAME"
  else
    echo "[warn] qdrant snapshot failed (use the Qdrant Cloud backup feature, or re-ingest to rebuild)"
  fi
fi

# 3. Optional: copy to S3 (set S3_BACKUP_BUCKET in .env.prod; the EC2 instance role needs s3:PutObject)
S3_BUCKET="$(env_get S3_BACKUP_BUCKET)"
if [ -n "$S3_BUCKET" ]; then
  aws s3 cp "$BACKUP_DIR/postgres-$STAMP.sql.gz" "s3://$S3_BUCKET/postgres/" && echo "[ok] copied postgres dump to S3" || echo "[warn] S3 upload failed"
  for f in "$BACKUP_DIR"/qdrant-*; do
    [ -f "$f" ] && [ "$f" -nt "$BACKUP_DIR/postgres-$STAMP.sql.gz" ] && aws s3 cp "$f" "s3://$S3_BUCKET/qdrant/" || true
  done
fi

# 4. Keep only the last KEEP_DAYS days
find "$BACKUP_DIR" -type f \( -name 'postgres-*.sql.gz' -o -name 'qdrant-*' \) -mtime +"$KEEP_DAYS" -delete

# Optional: copy off the server (object storage), for example with rclone:
#   rclone copy "$BACKUP_DIR" remote:nexora-backups
