"""Output guardrails: what may leave the system in an answer.

- **Sensitive data is redacted**: secrets, card numbers, government IDs —
  even if they appear in an uploaded document, an answer shouldn't echo them.
  Contact info (emails/phones) is allowed by default because the user's own
  documents legitimately contain it (configurable).
- **System-prompt leakage is blocked**: if an answer reproduces our internal
  prompts (a successful jailbreak), it's replaced with a refusal. Detected by
  overlapping 8-word sequences with the prompts — cheap and exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from rag_agent.agents import prompts
from rag_agent.config import settings
from rag_agent.guardrails.pii import mask_pii

_SHINGLE = 8
LEAK_REFUSAL = (
    "I can't share details of my internal instructions. "
    "Ask me about your documents or the web instead."
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


@lru_cache
def _prompt_shingles() -> set[tuple[str, ...]]:
    texts = [v for k, v in vars(prompts).items() if k.isupper() and isinstance(v, str)]
    return set().union(*(_shingles(_words(t)) for t in texts))


def leaks_system_prompt(answer: str, min_overlap: int = 3) -> bool:
    return len(_shingles(_words(answer)) & _prompt_shingles()) >= min_overlap


@dataclass
class OutputCheck:
    text: str
    redacted: dict[str, int]
    leaked_prompt: bool

    @property
    def changed(self) -> bool:
        return bool(self.redacted) or self.leaked_prompt


def check_output(answer: str) -> OutputCheck:
    if leaks_system_prompt(answer):
        return OutputCheck(LEAK_REFUSAL, {}, True)
    result = mask_pii(answer, only_sensitive=not settings.redact_contact_info, style="redact")
    return OutputCheck(result.text, dict(result.found), False)
