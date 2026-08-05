"""Provider-agnostic ChessAI wrapper."""

from chess_fight.common.common_types import ChatMessage
from chess_fight.models.chess_ai import ChessAI
from chess_fight.providers.registry import get_provider


class ProviderChessAI(ChessAI):
    """ChessAI implementation using the provider abstraction layer."""

    def __init__(self, provider_name: str, model_id: str, api_key: str, **params):
        super().__init__()
        self.provider_name = provider_name
        self.model_id = model_id
        self.api_key = api_key
        self.params = params  # temperature, max_tokens, etc.

        provider = get_provider(provider_name)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")
        self.provider = provider

        self.name = f"{provider_name}:{model_id}"

    async def _get_move_from_model(self, fen: str) -> str:
        prompt = self._create_prompt(fen)

        result = await self.provider.complete(
            self.api_key,
            self.model_id,
            [ChatMessage(role="user", content=prompt)],
            **self.params
        )

        self.last_completion_result = result
        return self._extract_move(result.text)

    def _extract_move(self, text: str) -> str:
        """Extract UCI move from LLM output."""
        import re

        # Strip thinking blocks
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

        # Find UCI pattern: [a-h][1-8][a-h][1-8][qrbn]?
        uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
        matches = re.findall(uci_pattern, text.lower())

        if matches:
            return str(matches[0])

        # Fallback: look for "I will play X" or similar patterns
        fallback_patterns = [
            r'(?:play|move|choose)\s+([a-h][1-8][a-h][1-8][qrbn]?)',
            r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b',
        ]
        for pattern in fallback_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return str(match.group(1))

        return ""
