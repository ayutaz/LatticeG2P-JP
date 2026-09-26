#!/usr/bin/env bash
# ★インスタンスを破棄する。起動している限り課金され続ける（docs/pitfalls.md R-14）。
# usage: infra/vast_teardown.sh <INSTANCE_ID>
set -euo pipefail
cd "$(dirname "$0")/.."

INSTANCE="${1:?usage: vast_teardown.sh <INSTANCE_ID>}"
export VAST_API_KEY="$(grep VAST_API_KEY .env | cut -d= -f2)"
vastai set api-key "$VAST_API_KEY" >/dev/null

echo "破棄前に checkpoint を回収します"
./infra/vast_sync.sh pull "$INSTANCE" || echo "回収に失敗（checkpoint がない可能性）"

vastai destroy instance "$INSTANCE"
echo "破棄しました。残っているインスタンス:"
vastai show instances
