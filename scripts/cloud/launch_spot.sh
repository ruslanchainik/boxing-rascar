#!/usr/bin/env bash
# Launch E2E-Spot job on DataSphere. Run from C:/hack.
set -e
KEY="C:/Users/Руслан/Downloads/authorized_key.json"
PROJECT="bt12vk1slp4dtfgh46vk"
JOB_DIR="C:/hack/cloud_spot"
DATASPHERE="C:/Users/Руслан/AppData/Roaming/Python/Python314/Scripts/datasphere.exe"

echo "[launch-spot] refreshing IAM token..."
export YC_IAM_TOKEN=$(python C:/hack/scripts/cloud/get_iam_token.py "$KEY")
echo "[launch-spot] token len ${#YC_IAM_TOKEN}"

cd "$JOB_DIR"
echo "[launch-spot] starting job..."
"$DATASPHERE" project job execute -p $PROJECT -c datasphere_config.yaml
