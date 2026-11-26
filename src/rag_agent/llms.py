"""ChatGroq factories.

Two tiers:
- ``heavy_model`` — strong reasoning model for planning + answer synthesis.
- ``fast_model``  — small model for routine steps (routing, judging, rewriting).

Both retry with exponential backoff on Groq 429 (rate limit) / 5xx errors via
``max_retries``. Groq's free tier is tight on tokens-per-minute, so callers
should keep prompts compact.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from rag_agent.config import settings


class MissingAPIKeyError(RuntimeError):
    """Raised when a model is requested but GROQ_API_KEY is not configured."""


def _require_key() -> str:
    if not settings.groq_api_key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Create a free key at https://console.groq.com/keys "
            "and add it to your .env file."
        )
    return settings.groq_api_key


def _supports_reasoning_effort(model: str) -> bool:
    # gpt-oss and qwen3 models accept reasoning_effort; Llama models reject it.
    return model.startswith(("openai/gpt-oss", "qwen/"))


def build_chat_model(model: str, *, streaming: bool) -> ChatGroq:
    kwargs: dict = {
        "model": model,
        "api_key": _require_key(),
        "temperature": settings.llm_temperature,
        "max_retries": settings.llm_max_retries,
        "timeout": settings.llm_timeout_s,
        "streaming": streaming,
    }
    if _supports_reasoning_effort(model):
        kwargs["reasoning_effort"] = settings.reasoning_effort
    return ChatGroq(**kwargs)


@lru_cache
def heavy_model(streaming: bool = True) -> ChatGroq:
    """Capable reasoning model for planning + answer synthesis."""
    return build_chat_model(settings.model_heavy, streaming=streaming)


@lru_cache
def fast_model(streaming: bool = False) -> ChatGroq:
    """Small, cheap model for routine steps."""
    return build_chat_model(settings.model_fast, streaming=streaming)


# --- per-agent model routing ---------------------------------------------------
#
# Each role maps to the heavy or fast tier (configurable in settings). Spreading
# roles across both tiers matters on Groq's free plan: each model has its own
# tokens-per-minute bucket, so one question doesn't exhaust a single budget.

Role = Literal["planner", "worker", "writer", "verifier", "summarizer"]

_overrides: dict[str, BaseChatModel] = {}


def get_model(role: Role) -> BaseChatModel:
    """The chat model for an agent role (heavy/fast tier chosen in settings)."""
    if role in _overrides:
        return _overrides[role]
    tier = getattr(settings, f"{role}_tier", "fast")
    streaming = role == "writer"  # only the final answer is streamed to the user
    model = settings.model_heavy if tier == "heavy" else settings.model_fast
    return _cached(model, streaming)


@lru_cache
def _cached(model: str, streaming: bool) -> ChatGroq:
    return build_chat_model(model, streaming=streaming)


@contextmanager
def override_models(models: dict[str, BaseChatModel]) -> Iterator[None]:
    """Swap in other models (e.g. scripted fakes in tests) for some roles."""
    previous = dict(_overrides)
    _overrides.update(models)
    try:
        yield
    finally:
        _overrides.clear()
        _overrides.update(previous)


def get_fallback_model(role: Role) -> BaseChatModel | None:
    """The model to try if ``role``'s primary fails (after its own retries).

    The fallback is the *other* tier. On Groq's free plan each model has its own
    rate-limit bucket, so when the heavy model is throttled the fast one usually
    still has budget (and vice versa). Returns ``None`` when there is no
    distinct fallback — or when the role was overridden (tests), unless a
    ``"<role>_fallback"`` override is given explicitly.
    """
    if f"{role}_fallback" in _overrides:
        return _overrides[f"{role}_fallback"]
    if role in _overrides or not settings.model_fallback:
        return None
    tier = getattr(settings, f"{role}_tier", "fast")
    primary = settings.model_heavy if tier == "heavy" else settings.model_fast
    other = settings.model_fast if tier == "heavy" else settings.model_heavy
    if other == primary:
        return None
    return _cached(other, role == "writer")


def with_fallback(role: Role, runnable_for):
    """Build ``runnable_for(model)`` for the primary model, falling back to the
    same runnable on the fallback model if the primary raises.

    ``runnable_for`` lets callers wrap the model first (e.g. with
    ``with_structured_output``) so the fallback has the identical shape.
    """
    primary = runnable_for(get_model(role))
    fallback = get_fallback_model(role)
    return primary.with_fallbacks([runnable_for(fallback)]) if fallback else primary
