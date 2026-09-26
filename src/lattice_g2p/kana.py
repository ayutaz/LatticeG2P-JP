"""カナ正規化.

★このモジュールがカナ正規化の唯一の実装である.
辞書ロード・学習データ生成・評価のすべてがここを通る.
別の場所で正規化を書いてはいけない (docs/pitfalls.md R-10).

方針 (docs/evaluation.md §2. 実データから確定):
  - NFKC で半角カナ・濁点を合成し, ひらがなをカタカナに寄せる
  - 句読点・括弧・感嘆符・空白を除去する
  - 長音符「ー」は保持する (除去すると「コーヒー」が「コヒ」になる)
  - 助詞の「は→ワ」「へ→エ」は変換しない.
    UniDic の pron が既に解決しているため, 正規化側で文脈判断をしてはいけない
  - ヲ / ヂ / ヅ は変換しない
"""

import unicodedata

KATAKANA_START = "ァ"
"""ァ"""
KATAKANA_END = "ヺ"
"""ヺ"""
PROLONGED = "ー"
"""ー"""

_HIRAGANA_START = "ぁ"
"""ぁ"""
_HIRAGANA_END = "ゖ"
"""ゖ"""
_HIRA_TO_KATA_OFFSET = ord(KATAKANA_START) - ord(_HIRAGANA_START)

# 除去する文字 (読み列に含めない).
# ★長音符「ー」を含めてはいけない.
_REMOVED = frozenset(
    [" ", "　", "\t", "\n", "\r"]  # 半角空白・全角空白・タブ・改行
    + ["。", "、", "，", "．", ",", "."]  # 句読点
    + ["!", "?", "！", "？"]  # 感嘆符・疑問符
    + ["「", "」", "『", "』", "（", "）", "(", ")", "［", "］", "[", "]"]  # 括弧類
    + ["・", "…", "―", "─", "〜", "~"]  # 中黒・三点リーダ・ダッシュ (長音符とは別)
    + ["*"]  # UniDic の「値なし」マーカ
)


def normalize_kana(s: str) -> str:
    """読み文字列を正規化されたカタカナにする.

    処理順:
      1. NFKC 正規化 (半角カナ → 全角カナ, 濁点の合成)
      2. ひらがな → カタカナ
      3. 除去対象文字の削除
    """
    s = unicodedata.normalize("NFKC", s)

    out: list[str] = []
    for ch in s:
        if ch in _REMOVED:
            continue
        if _HIRAGANA_START <= ch <= _HIRAGANA_END:
            out.append(chr(ord(ch) + _HIRA_TO_KATA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def is_kana_only(s: str) -> bool:
    """カタカナと長音符のみで構成されているか.

    生成データのフィルタリングで使う (docs/training.md §3).
    """
    return all(KATAKANA_START <= ch <= KATAKANA_END or ch == PROLONGED for ch in s)


# --- 異表記の同一視 ---

_VOWEL_OF = {}
"""カタカナ 1 文字 -> その母音 (ア/イ/ウ/エ/オ)."""

for _row, _vowel in [
    ("アカサタナハマヤラワガザダバパャヮァ", "ア"),
    ("イキシチニヒミリヰギジヂビピィ", "イ"),
    ("ウクスツヌフムユルグズヅブプュゥヴッ", "ウ"),
    ("エケセテネヘメレヱゲゼデベペェ", "エ"),
    ("オコソトノホモヨロヲゴゾドボポョォ", "オ"),
]:
    for _ch in _row:
        _VOWEL_OF[_ch] = _vowel

_LONG_VOWEL_PARTNER = {"ア": "ア", "イ": "イ", "ウ": "ウ", "エ": "イ", "オ": "ウ"}
"""各母音の長音を表すのに使われる仮名.

オ段の長音は「ウ」(コウ), エ段の長音は「イ」(ヘイ) で書かれる.
"""


_YOTSUGANA = str.maketrans({"ヅ": "ズ", "ヂ": "ジ"})
"""四つ仮名. 現代標準語では ヅ = ズ, ヂ = ジ で音が同じ."""


def phonetic_key(reading: str) -> str:
    """長音の書き分けを吸収した比較用のキーを返す.

    UniDic の pron 形と kana 形は長音の書き方だけが違うことがある.

        亢 -> コウ (kana) / コー (pron)
        丙 -> ヘイ (kana) / ヘー (pron)

    これらは別の読みではない. 同一視できないと, 多音語の抽出結果が
    異表記ペアだらけになる (docs/pitfalls.md R-16).

    正規化の方針: 長音を表す仮名と長音符「ー」を, すべて母音そのものに寄せる.

        コウ -> コオ,  コー -> コオ
        ヘイ -> ヘエ,  ヘー -> ヘエ
        サイ -> サイ         (ア段 + イ は長音ではないのでそのまま)

    ★四つ仮名 (ヂ = ジ, ヅ = ズ) も吸収する.
    現代標準語では音が同じで, 表記の選択でしかない.

        縮む -> チヂム (現代仮名遣い) / チジム (表音式)
        鼓   -> ツヅミ            / ツズミ

    当初は audit だけに入れていたが (「dev で 0 件なので精度に効かない」),
    **benchmark 13,536 件で測ると 33 件あり, OpenJTalk との差の 5.1% を占めていた**
    (docs/evaluation.md §2). モデルの誤りではなく正規化の不一致である.

    ★normalize_kana とは目的が違う. こちらは比較専用であり,
    出力する読みとして使ってはいけない.
    """
    out: list[str] = []
    prev_vowel: str | None = None

    for ch in reading:
        if ch == PROLONGED:
            out.append(prev_vowel if prev_vowel else ch)
            continue

        vowel = _VOWEL_OF.get(ch)
        if prev_vowel is not None and vowel is not None and ch == _LONG_VOWEL_PARTNER[prev_vowel]:
            # 直前の母音の長音を表す仮名 -> 母音そのものに寄せる
            out.append(prev_vowel)
            prev_vowel = vowel
            continue

        out.append(ch)
        prev_vowel = vowel

    return "".join(out).translate(_YOTSUGANA)
