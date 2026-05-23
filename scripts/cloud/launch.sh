#!/usr/bin/env bash
# Launch the DataSphere job. Run from C:/hack root.
set -e
KEY="C:/Users/Руслан/Downloads/authorized_key.json"
PROJECT="bt12vk1slp4dtfgh46vk"
JOB_DIR="C:/hack/cloud_job"
DATASPHERE="C:/Users/Руслан/AppData/Roaming/Python/Python314/Scripts/datasphere.exe"

echo "[launch] refreshing IAM token..."
export YC_IAM_TOKEN=$(python C:/hack/scripts/cloud/get_iam_token.py "$KEY")
echo "[launch] token len ${#YC_IAM_TOKEN}"

cd "$JOB_DIR"
echo "[launch] starting job..."
"$DATASPHERE" project job execute -p $PROJECT -c datasphere_config.yaml
