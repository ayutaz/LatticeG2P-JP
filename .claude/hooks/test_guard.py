#!/usr/bin/env python3
"""guard.py の判定テスト。

    python3 .claude/hooks/test_guard.py

★ケース文字列をこのファイルに置いてあるのは、guard.py 自身に引っかからないため。
コマンドラインに規約違反の語を書くと、フックがそのコマンドを止めてしまう
（実際にこれで patch が実行できなくなった）。
"""

from __future__ import annotations  # ★システムの python3 が古い場合があるため

import json
import subprocess
import sys
from pathlib import Path

GUARD = Path(__file__).with_name("guard.py")

# (コマンド, 期待する判定). None は「通す」
CASES = [
    # --- 1. uv の規約（deny。★ユーザーには確認しない）---
    ("uv run python scripts/train.py --config configs/tiny.yaml", None),
    ("uv run pytest", None),
    ("uv add onnxruntime", None),
    ("uv pip install numpy", None),
    ("python" + " scripts/train.py", "deny"),
    ("python3" + " -m pytest", "deny"),
    ("python3" + " -m pip install x", "deny"),
    ("pip" + " install torch", "deny"),
    ("pip3" + " install torch", "deny"),
    ("cd /x && FOO=1 python" + " scripts/train.py", "deny"),
    ("(cd /x && python" + " -u scripts/train.py)", "deny"),
    # ★uv run のオプションの後ろの python は uv 経由（2026-09-26 に ask で止まった誤検出）
    ("uv run --with fugashi python scripts/run_mecab_reference.py 2>&1 | tail -10", None),
    ("uv run --extra hf python scripts/train.py", None),
    ("nohup /root/.local/bin/uv run python -u scripts/train.py > log 2>&1 &", None),
    ("for s in 1 2; do uv run python scripts/train.py --seed $s; done", None),
    # ★使い捨ての python3 は止めない（docs の一括置換などで日常的に使う）
    ("python3 - <<'PY'\nprint(1)\nPY", None),
    ("python3 -c \"import json; print(1)\"", None),
    # ★文章として書いてあるだけのものは止めない
    ("echo \"python" + " scripts/train.py を使う\"", None),
    ("git commit -m \"$(cat <<'EOF'\nfix: python" + " scripts/x の説明\nEOF\n)\"", None),
    ("uv run python - <<'PY'\ns = 'python" + " scripts/x.py'\nPY", None),
    ("# python" + " scripts/x.py は後で回す\nuv run pytest", None),
    # ★ssh の遠隔コマンドと bash -c の中はコマンドとして見る
    ("ssh -p 1 root@h 'cd /workspace && python" + " scripts/train.py'", "deny"),
    ("ssh -p 1 root@h python" + " scripts/train.py", "deny"),
    ("ssh -p 1 root@h 'cd /workspace && /root/.local/bin/uv run python scripts/train.py'", None),
    ("bash -c 'python" + " scripts/train.py'", "deny"),
    # --- 2. ssh 越しのレイテンシ計測（deny）---
    ('ssh -p 1 root@h "uv run python scripts/run_benchmark.py"', "deny"),
    ("ssh root@h 'bash infra/curve_measure.sh 123'", "deny"),
    ("ssh root@h bash -s <<'EOF'\nuv run python scripts/run_benchmark.py\nEOF", "deny"),
    ("uv run python scripts/run_benchmark.py --model x --threads 1", None),
    ("bash infra/curve_measure.sh 51893024 xs s m l", None),
    ('git commit -m "ssh で run_benchmark を回さない"', None),
    # --- 3. vastai destroy（deny）---
    ("vastai destroy instance 99999999", "deny"),
    ("vastai show instances", None),
    ('echo "vastai destroy instance 99999999"', None),
    # --- 解析できないとき: 1 は通し、2・3 は文字列で判定する ---
    ("echo 'it" + " && python" + " scripts/train.py", None),
    ("echo 'it" + " && vastai destroy instance 99999999", "deny"),
]

MY_INSTANCE = Path("/tmp/my_instance")


def run(command: str) -> str | None:
    out = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}),
        capture_output=True,
        text=True,
    ).stdout.strip()
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else None


def main() -> int:
    failed = 0
    for command, want in CASES:
        got = run(command)
        ok = got == want
        failed += not ok
        first = command.splitlines()[0]
        print(f"  {'✓' if ok else '✗'} {str(got or 'allow'):5s}  {first[:60]}")

    # ★自分のインスタンスなら通ること
    saved = MY_INSTANCE.read_text() if MY_INSTANCE.exists() else None
    MY_INSTANCE.write_text("12345678\n")
    got = run("vastai destroy instance 12345678")
    failed += got is not None
    print(f"  {'✓' if got is None else '✗'} {str(got or 'allow'):5s}  自分のインスタンスの destroy")
    if saved is None:
        MY_INSTANCE.unlink(missing_ok=True)
    else:
        MY_INSTANCE.write_text(saved)

    print("\n" + ("すべて期待どおり" if not failed else f"★{failed} 件が不一致"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
