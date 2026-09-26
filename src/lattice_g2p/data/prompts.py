"""生成プロンプト.

★プロンプトを変更したら PROMPT_VERSION を上げること.
生成データにバージョンを記録し, 再現性の手掛かりにする (docs/training.md §10).
"""

PROMPT_VERSION = "v1"

KANA_RULES = """\
読みの表記規則:
- すべて全角カタカナで書く
- 助詞の「は」は ワ、「へ」は エ と書く（発音どおり）
- 助詞の「を」は ヲ と書く
- 長音は表記どおりに書く（「東京」は トウキョウ。トーキョー とは書かない）
- 外来語の長音符はそのまま使う（「コーヒー」は コーヒー、「ラーメン」は ラーメン）
- 促音「ッ」、拗音「ャュョ」、撥音「ン」は省略しない
- 空白・句読点は入れない
"""
"""★この規則を書かないと助詞の「は」が「ハ」で返ってくる.

規則自体は docs/evaluation.md §2 で実データから確定したもの.
"""

READING_GROUP_TOOL = {
    "name": "report_reading_groups",
    "description": "単語の読み候補を検証し、意味ごとにグループ化して報告する",
    "input_schema": {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "description": "意味ごとにグループ化した読み",
                "items": {
                    "type": "object",
                    "properties": {
                        "reading": {"type": "string", "description": "全角カタカナの読み"},
                        "valid": {
                            "type": "boolean",
                            "description": "現代日本語でこの読みが実在するか",
                        },
                        "sense": {"type": "string", "description": "この読みのときの意味"},
                        "pos": {"type": "string", "description": "品詞"},
                        "examples": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "この読みになる語例",
                        },
                        "reason": {
                            "type": "string",
                            "description": "valid が false の場合の理由",
                        },
                    },
                    "required": ["reading", "valid", "sense", "pos"],
                },
            }
        },
        "required": ["groups"],
    },
}

SENTENCE_TOOL = {
    "name": "report_sentences",
    "description": "指定された語を指定された読みで使った例文と、その全文読みを報告する",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "生成した日本語の文"},
                        "reading": {
                            "type": "string",
                            "description": "文全体の読み（全角カタカナ）",
                        },
                        "target_start": {
                            "type": "integer",
                            "description": "対象語が text 中で始まる文字位置（0 始まり）",
                        },
                    },
                    "required": ["text", "reading", "target_start"],
                },
            }
        },
        "required": ["sentences"],
    },
}


def build_reading_group_prompt(surface: str, readings: list[str], pos_list: list[str]) -> str:
    return f"""\
日本語の単語「{surface}」について、辞書に登録されている読み候補を検証してください。

読み候補: {", ".join(readings)}
品詞: {", ".join(pos_list)}

次の作業をしてください。

1. 各読みが現代日本語で実在するか判定する
   - 古語や、単独では使われない語幹（例:「生」の「ハ」）は valid=false にする
2. 実在する読みを意味ごとにグループ化する
3. 各グループに意味の説明を付ける
4. 各グループでその読みになる語例を挙げる

{KANA_RULES}
report_reading_groups ツールで報告してください。
"""


def build_sentence_prompt(surface: str, reading: str, sense: str, pos: str, n: int) -> str:
    return f"""\
日本語の単語「{surface}」を「{reading}」と読む文を {n} 文作ってください。

対象語: {surface}
読み: {reading}
意味: {sense}
品詞: {pos}

要件:
- 「{surface}」が必ず「{reading}」と読まれる自然な現代日本語の文にすること
- 文の長さは 10〜60 文字程度
- 文体と話題を多様にすること（すべて「〜です。」で終わらせない）
- 対象語が文中に現れる位置も多様にすること（文頭・文中・文末）
- 数字（1、一、２など）を含めない
- アルファベットを含めない
- 固有名詞（人名・地名・企業名・製品名）を含めない
- 記号は句読点のみ

各文について、文全体の読みと、対象語「{surface}」が始まる文字位置（0 始まり）を
報告してください。

{KANA_RULES}
report_sentences ツールで報告してください。
"""
