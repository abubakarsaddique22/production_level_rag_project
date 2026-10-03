# ADR 0006: One EC2 VM with Docker Compose, deployed through SSM

- **Status:** Accepted
- **Date:** 2026-10

## Context

First AWS deployment, one developer, a demo-scale workload. The stack is heavy in RAM (torch, reranker, Presidio) but light in traffic.
The PRD states "no Kubernetes" as a non-goal.

## Decision

- **Compute:** one EC2 instance (Ubuntu 24.04, `t3.medium` or larger, 30 GB gp3, Elastic IP) running `docker-compose.prod.yml`:
  Caddy, API, Postgres, Redis, Prometheus, Grafana.
- **Images:** Amazon ECR (private repo, lifecycle rule keeps the last 5 images).
- **HTTPS:** Caddy obtains and renews the certificate itself.
- **Deploy:** pushing a git tag `v*` triggers GitHub Actions: build, push to ECR, then `aws ssm send-command` runs `scripts/server_deploy.sh` on the instance.
  **Port 22 stays closed.** GitHub runners have changing IPs, and Session Manager gives a shell when needed.
- **Secrets:** stored as GitHub secrets, written by the workflow to SSM Parameter Store (`SecureString`), downloaded by the server into `.env.prod` (mode 600). Never in the image or the repo.
- **Rollback:** `deploy.sh` checks the API health and returns to the previous tag if the new one is unhealthy. A manual rollback is a workflow run with the old tag.
- **Least privilege:** the EC2 role can read ECR, run SSM, read one parameter and write to one S3 backup bucket. The CI user can push to one ECR repository and send commands to one instance.
- **Backups:** `scripts/backup.sh` by cron to S3. Qdrant data is rebuilt by re-ingesting.
- **Network:** only 80 and 443 are public. Postgres, Redis, Prometheus (127.0.0.1:9090) and Grafana (127.0.0.1:3000) are private. `/metrics` returns 404 through Caddy.

## Consequences

- (+) Cheap, simple, and the whole path (tag to live) is automated with a safe rollback.
- (-) Single point of failure and no autoscaling. A reboot or a bad instance means downtime.
- (-) CI uses long-lived IAM access keys. Switching to GitHub OIDC with a role removes stored keys (planned improvement).
- (-) Latency and uptime on AWS have not been load-tested or measured over time.
