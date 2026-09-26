#!/usr/bin/env bash
# accuracy-latency カーブの計測（M5 Task 1 Step 3〜4）。
#   infra/curve_measure.sh <INSTANCE_ID> [サイズ...]
#
# ★レイテンシは必ずローカルで測る（R-13）。この スクリプトは
#   「vast.ai から checkpoint を回収 -> ローカルで ONNX 化 -> ローカル CPU で計測」
#   の順を強制する。逆順にしないこと。
set -euo pipefail
cd "$(dirname "$0")/.."

INSTANCE="${1:?usage: curve_measure.sh <INSTANCE_ID> [xs s m l]}"
shift || true
SIZES=("${@:-}")
[ -z "${SIZES[0]:-}" ] && SIZES=(xs s m l)

export VAST_API_KEY="$(grep VAST_API_KEY .env | cut -d= -f2)"
vastai set api-key "$VAST_API_KEY" >/dev/null
URL="$(vastai scp-url "$INSTANCE")"
HOSTPORT="${URL#scp://}"
USERHOST="${HOSTPORT%:*}"
PORT="${HOSTPORT##*:}"

for size in "${SIZES[@]}"; do
  echo "=== size-$size: 回収 ==="
  mkdir -p "checkpoints/size-$size"
  scp -q -P "$PORT" -o StrictHostKeyChecking=accept-new \
    "$USERHOST:/workspace/checkpoints/size-$size/{best.pt,text_vocab.json,kana_vocab.json}" \
    "checkpoints/size-$size/"
  echo "=== size-$size: ONNX 化 + INT8 ==="
  uv run python scripts/export_model.py \
    --checkpoint "checkpoints/size-$size" \
    --out "models/size-$size-fp32" \
    --quantize-out "models/size-$size-int8"
done

echo
echo "=== 精度（benchmark）==="
uv run python scripts/compare_systems.py \
  --onnx $(printf 'models/size-%s-int8 ' "${SIZES[@]}")

echo
echo "=== ★ローカル CPU レイテンシ ==="
uv run python scripts/run_benchmark.py --threads 1 --openjtalk \
  --model $(printf 'models/size-%s-int8 ' "${SIZES[@]}")
