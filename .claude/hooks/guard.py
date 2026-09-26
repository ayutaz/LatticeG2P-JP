#!/usr/bin/env python3
"""PreToolUse(Bash) のガード。このプロジェクトで実際に起きた失敗を機械的に止める。

★ここは素の python3 で動かす（uv run は起動が遅く、フックは毎回走るため）。
プロジェクトのコードではなくインフラなので、uv の規約の対象外。

止めるもの（★すべて deny。ユーザーには確認を求めない）:
  1. 素の python / pip でプロジェクトのコードや依存を扱う  規約（CLAUDE.md）
  2. ssh 越しの CPU レイテンシ計測                        R-13。再現性が無い
  3. 自分が起動していないインスタンスの vastai destroy     他プロジェクトを壊す

★ask を使わない。確認待ちで自動化が止まるほうが
損が大きい。deny なら理由が Claude に返り、Claude が自分でコマンドを直して続行できる。

★1 は以前 ask だった。正規表現でコマンド文字列全体を見ていたので、
「規約違反の語が**書いてあるだけ**」の行（ヒアドキュメントの中の文章など）も
引っかかったため。さらに後読み `(?<!uv run )` で判定していたので、
`uv run --with fugashi python ...` を素の python と誤認した。
**いまはコマンドを字句解析し、実際に実行されるコマンドの先頭だけを見る:**

  - ヒアドキュメントの本文は見ない（Python の stdin・コミットメッセージは文章）。
    ただし ssh / bash / sh に流し込むものはコマンドとして見る
  - 引用符の中の文章は見ない（echo "..." や -m "..."）。
    ただし ssh の遠隔コマンドと bash -c / sh -c の引数は中を解析する
  - 行頭（語頭）の # 以降はコメント
  - uv run の後ろのオプション（--with / --extra など）は問題にならない（先頭が uv）

★解析できないとき（引用符の不整合など）は、1 は通し、2・3 は文字列で判定する（安全側）。
"""

from __future__ import annotations  # ★システムの python3 が古い場合があるため

import json
import re
import sys
from pathlib import Path

MY_INSTANCE = Path("/tmp/my_instance")

# --- 1. 素の python / pip ---
# ★テキスト処理の使い捨て python3 は止めない（docs の一括置換などで日常的に使う）。
#   止めるのは「プロジェクトのコードを動かす」「依存を入れる」ケースだけ。
PYTHON = re.compile(r"python3?(\.\d+)?")
PROJECT_MODULES = ("pytest", "pip", "lattice_g2p")

# --- 2. ssh 越しのレイテンシ計測 ---
LATENCY_SCRIPTS = ("run_benchmark", "curve_measure.sh", "m6_measure.sh")

# --- 3. vastai destroy ---
DESTROY = re.compile(r"vastai\s+destroy\s+instance\s+(\d+)")

# --- 字句解析 ---
WRAPPERS = {
    "nohup", "time", "exec", "env", "sudo", "command", "nice",
    "then", "do", "else", "elif", "if", "while", "until", "!", "{", "}",
}
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
SHELLS = {"ssh", "bash", "sh", "zsh"}
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
SSH_OPTS_WITH_ARG = set("bcDEeFIiJLlmOopQRSWw")


def split_segments(text: str) -> list[list[str]]:
    """コマンド列を、実行単位（セグメント）ごとの語の列に分ける.

    引用符の外の ; & | ( ) < > ` 改行 $( で区切る。引用符は取り除き、中身は 1 語にする。

    Raises:
        ValueError: 引用符が閉じていない
    """
    segments: list[list[str]] = []
    words: list[str] = []
    word = ""
    in_word = False
    quote = ""
    i, n = 0, len(text)

    def end_word() -> None:
        nonlocal word, in_word
        if in_word:
            words.append(word)
        word, in_word = "", False

    def end_segment() -> None:
        nonlocal words
        end_word()
        if words:
            segments.append(words)
        words = []

    while i < n:
        c = text[i]
        if quote:
            if c == quote:
                quote = ""
            elif c == "\\" and quote == '"' and i + 1 < n:
                i += 1
                word += text[i]
            else:
                word += c
        elif c in "'\"":
            quote = c
            in_word = True
        elif c == "\\" and i + 1 < n:
            i += 1
            if text[i] != "\n":
                word += text[i]
                in_word = True
        elif c == "#" and not in_word:
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif c in " \t":
            end_word()
        elif c == "$" and i + 1 < n and text[i + 1] == "(":
            end_segment()  # $( ... ) の中は別のコマンド
            i += 1
        elif c in ";&|()<>`\n":
            end_segment()
        else:
            word += c
            in_word = True
        i += 1
    if quote:
        raise ValueError("引用符が閉じていない")
    end_segment()
    return segments


def command_words(seg: list[str]) -> list[str]:
    """先頭の環境変数の代入と nohup などを飛ばした、実際に実行されるコマンド."""
    i = 0
    while i < len(seg) and (seg[i] in WRAPPERS or ASSIGNMENT.match(seg[i])):
        i += 1
    return seg[i:]


def basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def strip_heredocs(text: str) -> tuple[str, list[tuple[str, str]]]:
    """ヒアドキュメントの本文を取り除く.

    Returns:
        (本文を除いたコマンド列, [(受け取るシェル, 本文)]). 後者は ssh / bash / sh に
        流し込むものだけ（それはコマンドなので解析する）.
    """
    lines = text.split("\n")
    kept: list[str] = []
    shell_bodies: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        kept.append(line)
        i += 1
        m = HEREDOC.search(line)
        if not m:
            continue
        body: list[str] = []
        while i < len(lines) and lines[i].strip() != m.group(2):
            body.append(lines[i])
            i += 1
        i += 1  # 終端の行
        receiver = ""
        try:
            before = split_segments(line[: m.start()])
            if before:
                words = command_words(before[-1])
                receiver = basename(words[0]) if words else ""
        except ValueError:
            pass
        if receiver in SHELLS:
            shell_bodies.append((receiver, "\n".join(body)))
    return "\n".join(kept), shell_bodies


def ssh_remote(words: list[str]) -> str:
    """ssh の遠隔コマンド（ホストより後ろ）."""
    i = 1
    while i < len(words) and words[i].startswith("-"):
        opt = words[i]
        i += 2 if len(opt) == 2 and opt[1] in SSH_OPTS_WITH_ARG else 1
    return " ".join(words[i + 1 :])


def commands(text: str, depth: int = 0) -> list[tuple[list[str], str]]:
    """実行されるコマンドをすべて (語の列, どこで実行されるか) で返す.

    ssh の遠隔コマンドと bash -c / sh -c の中も再帰的に展開する. 場所は "local" / "ssh".
    """
    stripped, bodies = strip_heredocs(text)
    out: list[tuple[list[str], str]] = []
    for seg in split_segments(stripped):
        words = command_words(seg)
        if not words:
            continue
        out.append((words, "local"))
        exe = basename(words[0])
        if depth >= 3:
            continue
        if exe == "ssh":
            out += [(w, "ssh") for w, _ in commands(ssh_remote(words), depth + 1)]
        elif exe in SHELLS and "-c" in words:
            k = words.index("-c")
            out += commands(" ".join(words[k + 1 : k + 2]), depth + 1)
    for receiver, body in bodies:
        where = "ssh" if receiver == "ssh" else "local"
        out += [(w, where if where == "ssh" else loc) for w, loc in commands(body, depth + 1)]
    return out


def is_bare_python(words: list[str]) -> bool:
    exe = words[0]
    if "/" in exe:  # パス指定（.venv/bin/python など）は対象外
        return False
    if PYTHON.fullmatch(exe):
        args = words[1:]
        for j, a in enumerate(args):
            if a.startswith(("scripts/", "./scripts/")):
                return True
            if a == "-m" and j + 1 < len(args) and args[j + 1].split(".")[0] in PROJECT_MODULES:
                return True
        return False
    if exe in ("pip", "pip3"):
        return len(words) > 1 and words[1] in ("install", "uninstall", "add")
    return False


def decide(reason: str) -> None:
    """★deny だけを使う。理由は Claude に返り、Claude が自分で直す."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            },
            ensure_ascii=False,
        )
    )
    sys.exit(0)


UV_REASON = (
    "★このプロジェクトの Python は uv で管理します（CLAUDE.md）。"
    "コマンドを直して再実行してください。\n"
    "  python scripts/... -> uv run python scripts/...\n"
    "  python -m pytest   -> uv run pytest\n"
    "  依存の追加         -> uv add <package>（一時的なら uv run --with <package> ...）\n"
    "ローカルでも vast.ai インスタンス側でも同じです（インスタンスでは /root/.local/bin/uv）。"
)
LATENCY_REASON = (
    "★CPU レイテンシを vast.ai 上で測ってはいけません（docs/pitfalls.md R-13）。\n"
    "vast.ai は CPU を他テナントと共有し、型番も vCPU 割り当ても不定なので、\n"
    "そこで測った値に再現性はありません。**数値はそれらしく出るので気づきません。**\n"
    "  ✓ vast.ai で学習 -> checkpoint 回収 -> ローカルで ONNX 化 -> ローカルで計測\n"
    "  infra/curve_measure.sh がこの順序を強制します。"
)


def destroy_reason(target: str, mine: str) -> str:
    return (
        f"★インスタンス {target} は自分が起動したものとして記録されていません"
        f"（/tmp/my_instance = {mine or '未記録'}）。\n"
        "**同じアカウントで他のプロジェクトのインスタンスが動いていることがあります。**\n"
        "他プロジェクトのインスタンスを破棄しかけたことがあります。\n"
        "自分のものなら `echo <ID> > /tmp/my_instance` を先に実行してください。"
    )


def check_destroy(target: str) -> None:
    mine = MY_INSTANCE.read_text().strip() if MY_INSTANCE.exists() else ""
    if target != mine:
        decide(destroy_reason(target, mine))


def main() -> None:
    try:
        command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    except Exception:
        sys.exit(0)  # ★判定できないときは通す。フックで作業を止めない
    if not command:
        sys.exit(0)

    try:
        found = commands(command)
    except ValueError:
        # ★解析できない. 1 は通し、取り返しのつかない 2・3 だけ文字列で判定する
        text, _ = strip_heredocs(command)
        if "ssh" in text and any(s in text for s in LATENCY_SCRIPTS):
            decide(LATENCY_REASON)
        m = DESTROY.search(text)
        if m:
            check_destroy(m.group(1))
        sys.exit(0)

    for words, where in found:
        if is_bare_python(words):
            decide(UV_REASON)
        if where == "ssh" and any(s in w for w in words for s in LATENCY_SCRIPTS):
            decide(LATENCY_REASON)
        if basename(words[0]) == "vastai":
            m = DESTROY.search(" ".join(words))
            if m:
                check_destroy(m.group(1))

    sys.exit(0)


if __name__ == "__main__":
    main()
