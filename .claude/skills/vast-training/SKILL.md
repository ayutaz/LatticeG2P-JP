---
name: vast-training
description: Use when running GPU training for this project on vast.ai - launching or destroying instances, syncing code and data, choosing how many runs to parallelize, or fetching checkpoints. Covers the hard rule that CPU latency must be measured locally (never on vast.ai), instance safety around other projects on the same account, and the measured CPU/memory limits that determine parallelism.
---

# vast.ai での学習

**GPU が要る処理はすべて vast.ai。ローカルで GPU 学習をしない。**
API キーは `.env` の `VAST_API_KEY`（gitignore 済み）。

## ★★ 破ってはいけない 2 つ

### 1. CPU レイテンシはローカルでしか測らない（R-13）

```
✓ vast.ai で学習 → checkpoint 回収 → ローカルで ONNX 化 → ローカル CPU で計測
✗ vast.ai 上で CPU レイテンシを測る
```

vast.ai は CPU を他テナントと共有し、型番も vCPU 割り当ても不定。
**そこで測った値に再現性は無い。** Go/No-Go の半分がレイテンシなので、
測定場所を間違えると判定そのものが無意味になる。
★**数値は「それらしく」出るので気づきにくい。**

`infra/curve_measure.sh` / `infra/m6_measure.sh` はこの順序を強制する。

### 2. ★自分が起動したインスタンスだけを破棄する

**同じアカウントで他のプロジェクトのインスタンスが動いていることがある。**
ラベルで判断して他プロジェクトのインスタンスを破棄しかけたことがある。

```bash
echo <INSTANCE_ID> > /tmp/my_instance     # 起動したら必ず記録する
cat /tmp/my_instance                      # 破棄前に必ず確認する
```

**`vastai show instances` に出るものを無条件に破棄しない。**

## 手順

```bash
bash infra/vast_search.sh                 # cuda_vers >= 13.0 / 回線速度で絞る
bash infra/vast_launch.sh <OFFER_ID>      # uv も入る
echo <INSTANCE_ID> > /tmp/my_instance

# 起動待ち（★数分を超えたら捨てて別のオファーにする）
vastai show instance <ID> --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['actual_status'])"

# 転送
tar czf - src configs scripts pyproject.toml uv.lock .python-version | ssh -p <PORT> root@<HOST> 'cd /workspace && tar xzf -'
scp -P <PORT> data/{dic_cache.pkl,train_m5_split.jsonl,dev_r21.jsonl} root@<HOST>:/workspace/data/
ssh -p <PORT> root@<HOST> 'cd /workspace && /root/.local/bin/uv sync --extra dev'   # ★Python 3.13.5 に固定（無ければ uv が入れる）

# 学習（★nohup ... & を必ず付ける。付け忘れると逐次実行になる）
ssh -p <PORT> root@<HOST> 'cd /workspace && nohup /root/.local/bin/uv run python -u scripts/train.py ... > log 2>&1 &'

# 回収 → ローカルで測る → ★破棄
vastai destroy instance $(cat /tmp/my_instance)
```

## ★何本並列に回せるか

**律速は GPU ではない。vCPU 数とメモリである。**

| 実測 | |
|---|---|
| RTX A4000 の利用率 | 1 本で **11%** / 5 本でようやく 99% |
| 理由 | ラティス構築が Python で、CPU バウンド |
| メモリ | 1 本あたり約 **4.5 GB**（辞書 885,203 エントリをプロセスごとに持つ） |

| インスタンス | 同時実行数 | 1 本あたり |
|---|---|---|
| 6 vCPU / 31 GB | **4 本まで**（6 本で OOM 寸前: 残り 2 GB） | 0.8〜1.2 s/step |
| 16 vCPU / 125 GB | 6 本以上 | 単独なら **0.11 s/step** |

★**オファーは vCPU とメモリで選ぶ。** GPU の型番より効く。

## 待ちループの落とし穴

★**`pgrep -f "..."` は待ちループ自身にマッチする。**

```bash
✗ until [ "$(pgrep -fc "checkpoints/m5")" -eq 0 ]; do sleep 60; done   # 永遠に終わらない
✓ PID=$(pgrep -f "out-dir checkpoints/m5" | head -1)
  while kill -0 $PID 2>/dev/null; do sleep 60; done
```

同じ理由で、**config 名での判定も別の実行に引っかかる**
（`pgrep -fc "configs/size-"` が `size-m.yaml` を使う別の run に当たった）。
**`--out-dir` のような一意な文字列を使うこと。**

## その他

- ★**`uv` を使う。** インスタンス側でも素の `python` / `pip` を使わない
- ★**`--extra hf` はデフォルトで入らない。** 事前学習エンコーダを使うときは明示する
- 学習規模（3〜10M params × 数千文）に H100 は不要。A4000（$0.096/h）で 30 分・約 $0.05
