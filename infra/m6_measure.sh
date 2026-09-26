#!/usr/bin/env bash
# M6 の 3 条件を回収して測る。
#   infra/m6_measure.sh <INSTANCE_ID>
#
# ★レイテンシはローカルで測る（R-13）。この順序を崩さないこと。
set -euo pipefail
cd "$(dirname "$0")/.."

INSTANCE="${1:?usage: m6_measure.sh <INSTANCE_ID>}"
RUNS=(m6-size-m m6-trans16 m6-trans32)

export VAST_API_KEY="$(grep VAST_API_KEY .env | cut -d= -f2)"
vastai set api-key "$VAST_API_KEY" >/dev/null
URL="$(vastai scp-url "$INSTANCE")"
HOSTPORT="${URL#scp://}"
USERHOST="${HOSTPORT%:*}"
PORT="${HOSTPORT##*:}"

for run in "${RUNS[@]}"; do
  echo "=== $run: 回収 ==="
  mkdir -p "checkpoints/$run"
  for f in best.pt text_vocab.json kana_vocab.json metrics.json; do
    scp -q -P "$PORT" -o StrictHostKeyChecking=accept-new \
      "$USERHOST:/workspace/checkpoints/$run/$f" "checkpoints/$run/$f" || true
  done
  echo "=== $run: ONNX 化 + INT8 ==="
  uv run python scripts/export_model.py \
    --checkpoint "checkpoints/$run" \
    --out "models/$run-fp32" --quantize-out "models/$run-int8"
done

echo
echo "=== benchmark ==="
uv run python scripts/compare_systems.py --onnx $(printf 'models/%s-int8 ' "${RUNS[@]}")

echo
echo "=== Hard Set ==="
uv run python scripts/run_hard_set.py --onnx $(printf 'models/%s-int8 ' "${RUNS[@]}")

echo
echo "=== ★ローカル CPU レイテンシ ==="
uv run python scripts/run_benchmark.py --threads 1 --openjtalk \
  --model $(printf 'models/%s-int8 ' "${RUNS[@]}")
