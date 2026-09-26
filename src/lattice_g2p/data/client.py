"""Claude API クライアント.

★API キーをログに出さない. vast.ai インスタンスに送らない
(データ生成はローカルで行う).
"""

import os
import random
import time
from typing import Protocol

import anthropic
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5"
"""大量生成のコスト効率を優先. 読みの検証だけ別モデルにしてもよい."""


class ToolClient(Protocol):
    """構造化出力を返すクライアント. テストでは差し替える."""

    model: str

    def call_tool(self, prompt: str, tool: dict) -> dict: ...


class ClaudeClient:
    """ツール呼び出しを強制して構造化出力を得るクライアント."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_retries: int = 5,
        max_tokens: int = 8192,
    ) -> None:
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY が設定されていません。"
                ".env に追加するか環境変数として渡してください。"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def call_tool(self, prompt: str, tool: dict) -> dict:
        """ツール呼び出しを強制して構造化出力を得る.

        Raises:
            RuntimeError: max_retries 回失敗した場合
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[{"role": "user", "content": prompt}],
                )
                for block in response.content:
                    if block.type == "tool_use":
                        return dict(block.input)
                raise RuntimeError("ツール呼び出しが返りませんでした")
            except (anthropic.RateLimitError, anthropic.APIStatusError, RuntimeError) as e:
                last_error = e
                time.sleep(min(2**attempt + random.random(), 60))
        raise RuntimeError(f"{self.max_retries} 回失敗しました: {last_error}")
