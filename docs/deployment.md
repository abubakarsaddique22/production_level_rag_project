# Deployment on AWS (Step Y)

Architecture: one EC2 instance runs the Docker Compose stack (`docker-compose.prod.yml`): Caddy (HTTPS), API,
Postgres, Redis, Prometheus and Grafana. Images live in Amazon ECR. Vectors live in Qdrant Cloud.
Tracing is done with LangSmith (no Langfuse).

```
git tag v1.0.0 -> GitHub Actions (cd.yml, AWS access keys from GitHub secrets) -> build image -> push to ECR
                                          -> AWS Systems Manager (SSM) -> EC2 runs scripts/deploy.sh vX.Y.Z
                                                                           |-> healthy: done
                                                                           `-> unhealthy: automatic rollback
```

Deploys go through SSM, so **port 22 does not need to be open** (GitHub runners have changing IPs, so an
SSH rule limited to your own IP would block the deploy).

## 1. One-time AWS setup (console)

1. **ECR:** create a private repository named `nexora-rag`. Add a lifecycle rule that keeps only the last 5 images (images are large).
2. **IAM role for the EC2 instance** (name it `nexora-ec2-role`, trusted entity: EC2). Attach:
   - `AmazonSSMManagedInstanceCore` (lets SSM run the deploy)
   - `AmazonEC2ContainerRegistryReadOnly` (lets the server pull images)
   - an inline policy allowing `ssm:GetParameter` on `arn:aws:ssm:<region>:<account-id>:parameter/nexora/env-prod` (the server downloads its settings from there)
   - an inline policy allowing `s3:PutObject` on `arn:aws:s3:::<your-backup-bucket>/*` (for backups)
3. **S3 bucket** for backups: block all public access, enable default encryption, optional lifecycle rule to expire files after 30 days.
4. **EC2:** Ubuntu 24.04, `t3.medium` or larger (4 GB RAM minimum: torch, reranker and Presidio are heavy), 30 GB gp3 disk,
   instance profile = `nexora-ec2-role`. Allocate an **Elastic IP** and attach it.
5. **Security group:** inbound 80 and 443 from anywhere. No port 22. Nothing else (Postgres, Redis, Grafana stay private).
   (If you ever need a shell, use Session Manager in the AWS console instead of SSH.)
6. **IAM user for GitHub Actions** (name it `nexora-github-actions`, no console access, create an access key for "application running outside AWS").
   Attach an inline policy with:
   - ECR push on the `nexora-rag` repository (`ecr:GetAuthorizationToken` plus the standard push actions)
   - `ssm:SendCommand` on your instance and on the `AWS-RunShellScript` document
   - `ssm:GetCommandInvocation`
   - `ssm:PutParameter` on `arn:aws:ssm:<region>:<account-id>:parameter/nexora/env-prod` (the workflow stores the production settings there)
   - `sts:GetCallerIdentity` needs no permission
7. **DNS:** point your domain (Route 53 or any DNS) to the Elastic IP. Caddy needs a real domain for HTTPS.

## 2. One-time server setup

Open Session Manager (EC2 console, Connect) and run:

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo snap install aws-cli --classic          # Ubuntu 24.04 has no apt package for the AWS CLI
sudo usermod -aG docker ubuntu
sudo mkdir -p /opt/nexora-rag && sudo chown ubuntu /opt/nexora-rag
sudo -u ubuntu git clone https://github.com/abubakarsaddique22/production_level_rag_project.git /opt/nexora-rag
```

You do **not** create `.env.prod` by hand: the CD workflow writes it from your GitHub secrets on every deploy
(`.env.prod.example` only shows which values exist).

If the repository is private, the server needs read access (a read-only GitHub deploy key, or clone with a token).

## 3. GitHub settings

- Secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `RAG_QDRANT_URL`, `RAG_QDRANT_API_KEY`, `RAG_GROQ_API_KEY`,
  `RAG_JWT_SECRET`, `PG_PASSWORD`, `GRAFANA_PASSWORD`, `LANGSMITH_API_KEY`
- Variables: `AWS_REGION`, `ECR_REPOSITORY` (= `nexora-rag`), `EC2_INSTANCE_ID` (looks like `i-0123456789abcdef0`),
  `DOMAIN`, `S3_BACKUP_BUCKET`
- Use letters and digits only in generated passwords and secrets (no `$`, no quotes), because they are written into a `.env` file.

## 4. Release

```bash
git tag v1.0.0 && git push origin v1.0.0
```

The CD workflow builds, pushes and deploys. First start downloads the models (up to about 5 minutes).
Check: `https://<DOMAIN>/health` and `https://<DOMAIN>/docs`.

## 5. Create users (first deploy only)

`scripts/seed_users.py` creates 5 test accounts (employee, engineer, finance, hr, admin) that share one password.
On production the script refuses to run with the default password, so set your own:

```bash
cd /opt/nexora-rag
docker compose --env-file .env.prod -f docker-compose.prod.yml exec -e SEED_PASSWORD='<strong-password>' api python scripts/seed_users.py
```

## 6. Rollback

- Automatic: `deploy.sh` switches back to the previous tag if the new API is not healthy.
- Manual: GitHub, Actions, CD, Run workflow, enter the old tag (for example `v1.0.0`). Or on the server: `./scripts/deploy.sh v1.0.0`.

## 7. Backups

```bash
crontab -e
0 3 * * *  cd /opt/nexora-rag && ./scripts/backup.sh >> backups/backup.log 2>&1
```

Set `S3_BACKUP_BUCKET` in `.env.prod` to copy every backup to S3. The Qdrant index can be rebuilt by re-ingesting.

## 8. Monitoring

Prometheus (127.0.0.1:9090) and Grafana (127.0.0.1:3000) are not public. With SSH closed, use an SSM port-forward:
`aws ssm start-session --target <instance-id> --document-name AWS-StartPortForwardingSession --parameters portNumber=3000,localPortNumber=3000`
(needs the Session Manager plugin on your laptop). If RAM is tight, stop them: `docker compose ... stop prometheus grafana`.
