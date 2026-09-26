#!/usr/bin/env bash
# データ送信 / checkpoint 回収。
#   infra/vast_sync.sh push <INSTANCE_ID>
#   infra/vast_sync.sh pull <INSTANCE_ID>
#
# ★vastai CLI から scp サブコマンドが無くなったので、scp-url で接続先を取って
#   素の scp を使う（2026-09 時点。`vastai scp` はもう存在しない）。
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: vast_sync.sh push|pull <INSTANCE_ID>}"
INSTANCE="${2:?usage: vast_sync.sh push|pull <INSTANCE_ID>}"
export VAST_API_KEY="$(grep VAST_API_KEY .env | cut -d= -f2)"
vastai set api-key "$VAST_API_KEY" >/dev/null

URL="$(vastai scp-url "$INSTANCE")"   # scp://root@sshN.vast.ai:PORT
HOSTPORT="${URL#scp://}"
USERHOST="${HOSTPORT%:*}"
PORT="${HOSTPORT##*:}"
SCP=(scp -q -P "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)

case "$MODE" in
  push)
    "${SSH[@]}" "$USERHOST" 'mkdir -p /workspace/data /workspace/checkpoints'
    # 辞書は BSD 3-Clause なので転送してよい（docs/pitfalls.md R-01）
    # ★.python-version も送る。uv がインスタンス側で Python 3.13.5 を入れて固定する
    "${SCP[@]}" -r ./src ./configs ./scripts ./pyproject.toml ./uv.lock ./.python-version \
      "$USERHOST":/workspace/
    # ★現行の分割（M5 以降）。data/train_split.jsonl・data/dev.jsonl は M3d 以前の古い分割
    "${SCP[@]}" ./data/dic_cache.pkl ./data/train_m5_split.jsonl ./data/dev_r21.jsonl \
      "$USERHOST":/workspace/data/
    # 青空文庫のルビ由来データ（学習データにだけ足す）
    if [ -f ./data/ruby.jsonl ]; then
      "${SCP[@]}" ./data/ruby.jsonl "$USERHOST":/workspace/data/
    fi
    echo "送信しました"
    ;;
  pull)
    DEST="${3:-checkpoints}"
    mkdir -p "$DEST"
    for f in best.pt latest.pt text_vocab.json kana_vocab.json metrics.json; do
      "${SCP[@]}" "$USERHOST":/workspace/checkpoints/"$f" "$DEST"/ 2>/dev/null || true
    done
    echo "回収しました -> $DEST"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 1
    ;;
esac
