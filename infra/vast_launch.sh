#!/usr/bin/env bash
# インスタンスを作成する。 usage: infra/vast_launch.sh <OFFER_ID>
set -euo pipefail
cd "$(dirname "$0")/.."

OFFER_ID="${1:?usage: vast_launch.sh <OFFER_ID>}"
export VAST_API_KEY="$(grep VAST_API_KEY .env | cut -d= -f2)"
vastai set api-key "$VAST_API_KEY" >/dev/null

# ★Python は uv で管理する。onstart で uv を入れておく
vastai create instance "$OFFER_ID" \
  --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
  --disk 50 \
  --ssh \
  --onstart-cmd 'curl -LsSf https://astral.sh/uv/install.sh | sh && touch /workspace/.ready'

echo
echo "起動状況: vastai show instances"
echo "★学習が終わったら必ず infra/vast_teardown.sh <INSTANCE_ID> を実行すること"
