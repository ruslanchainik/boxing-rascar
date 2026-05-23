"""Generate Yandex Cloud IAM token from authorized_key.json.

Usage:
    python scripts/cloud/get_iam_token.py path/to/authorized_key.json
    # or
    python scripts/cloud/get_iam_token.py path/to/authorized_key.json > token.txt
"""
from __future__ import annotations
import json, sys, time
import jwt  # pip install pyjwt[crypto]
import requests


def main(key_path: str) -> None:
    with open(key_path, "r", encoding="utf-8") as f:
        key = json.load(f)
    now = int(time.time())
    payload = {
        "iss": key["service_account_id"],
        "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
        "iat": now,
        "exp": now + 3600,
    }
    encoded = jwt.encode(payload, key["private_key"], algorithm="PS256",
                         headers={"kid": key["id"]})
    r = requests.post("https://iam.api.cloud.yandex.net/iam/v1/tokens",
                      json={"jwt": encoded}, timeout=20)
    r.raise_for_status()
    print(r.json()["iamToken"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: get_iam_token.py <authorized_key.json>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
