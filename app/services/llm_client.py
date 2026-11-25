"""LLM client for OpenAI, Anthropic, and OpenRouter APIs."""
import logging
from typing import List, Dict, Any
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting OpenAI, Anthropic, and OpenRouter."""

    def __init__(self):
        self.settings = get_settings()
        self._providers = {}

    def _get_provider(self, provider: str = "openai"):
        """Get LLM provider instance."""
        if provider not in self._providers:
            if provider == "openai":
                from openai import AsyncOpenAI
                self._providers[provider] = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
            elif provider == "anthropic":
                from anthropic import AsyncAnthropic
                self._providers[provider] = AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
            elif provider == "openrouter":
                from openai import AsyncOpenAI
                self._providers[provider] = AsyncOpenAI(
                    api_key=self.settings.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1"
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        return self._providers[provider]

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        use_cache: bool = True,
    ) -> str:
        """Generate a response from the LLM."""
        model = model or self.settings.OPENAI_MODEL

        # OpenRouter implementation (uses OpenAI-compatible API)
        if "openrouter" in model.lower() or self.settings.OPENROUTER_API_KEY:
            client = self._get_provider("openrouter")
            response = await client.chat.completions.create(
                model=self.settings.OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        # OpenAI implementation
        elif model.startswith("gpt-") or "openai" in model.lower():
            client = self._get_provider("openai")
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        # Anthropic implementation
        elif model.startswith("claude-"):
            client = self._get_provider("anthropic")
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unknown model: {model}")

    async def generate_with_history(
        self,
        conversation_history: List[Dict[str, str]],
        new_message: str,
        model: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Generate response with conversation history."""
        model = model or self.settings.OPENAI_MODEL

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })
        messages.append({"role": "user", "content": new_message})

        if model.startswith("gpt-") or "openai" in model.lower():
            client = self._get_provider("openai")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        return ""


# Global instance
_llm_client = LLMClient()


async def get_llm_client() -> LLMClient:
    return _llm_client
