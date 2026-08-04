"""Google (Gemini) provider implementation using google-genai."""

from google import genai
from google.genai import types

from .base import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from .registry import register_provider


@register_provider
class GoogleProvider(ModelProvider):
    name = "google"
    requires_api_key = True

    def validate_key(self, api_key: str) -> bool:
        return len(api_key) > 20

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = genai.Client(api_key=api_key)
        models_response = await client.models.list()

        chat_models = []
        for model in models_response:
            if "generateContent" in model.supported_actions:
                if "gemini" in model.name.lower():
                    model_id = model.name.replace("models/", "")
                    chat_models.append(ModelInfo(
                        id=model_id,
                        name=model_id,
                        provider="google",
                        context_window=1000000 if "1.5" in model_id else 32768,
                    ))
        return chat_models

    async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult:
        client = genai.Client(api_key=api_key)

        # Convert messages to Gemini format
        system_prompt = ""
        user_content = ""
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            elif m.role == "user":
                user_content = m.content
            elif m.role == "assistant":
                # In a real implementation, we'd build a proper conversation history
                pass

        full_prompt = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content

        temperature = params.get("temperature", 0.1)
        max_tokens = params.get("max_tokens", 100)

        import time
        start = time.time()

        response = await client.models.generate_content_async(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )

        latency_ms = int((time.time() - start) * 1000)

        # Extract token usage
        tokens_in = None
        tokens_out = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_in = response.usage_metadata.prompt_token_count
            tokens_out = response.usage_metadata.candidates_token_count

        return CompletionResult(
            text=response.text if response.text else "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
