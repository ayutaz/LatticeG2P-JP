#!/usr/bin/env bash
# 学習に適したインスタンスを探す。
#
# ★本プロジェクトの学習規模（3〜10M params × 数万文）は小さい。H100 は不要。
#   論文の 4×H100 は 234 万文・100M モデルの構成であって、桁が 1〜2 つ違う。
set -euo pipefail
cd "$(dirname "$0")/.."

export VAST_API_KEY="$(grep VAST_API_KEY .env | cut -d= -f2)"
vastai set api-key "$VAST_API_KEY" >/dev/null

# ★cuda_vers >= 13.0 は必須。古いドライバのマシンだと torch (cu130) から
#   GPU が見えず、起動してから気づくことになる（実際に 1 台無駄にした）。
# ★inet_down / inet_up も効く。辞書 53 MB と checkpoint 84 MB を往復させるため。
# ★クエリは 1 行で書くこと。\ で折ると改行がそのまま渡って構文エラーになる
QUERY='reliability > 0.98 num_gpus=1 gpu_ram >= 16 disk_space >= 50 dph < 0.40 inet_down > 900 inet_up > 500 cuda_vers >= 13.0'
vastai search offers "$QUERY" -o 'dph' | head -20

echo
echo "★起動が数分を超えたら破棄して別のオファーにすること"
echo "★ラベル付きのインスタンス（別プロジェクト）には触らないこと"
