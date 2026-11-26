"""Neutralize indirect prompt injection in evidence (documents, web pages).

Every passage a tool hands to an agent passes through ``sanitize_evidence``:

1. **Invisible text is removed**: zero-width characters, bidi overrides and HTML
   comments. These are the classic ways to hide instructions from a human
   reader while a model still "sees" them.
2. **Instruction-like lines are replaced** with a visible marker, so the agent
   knows something was there but can't be steered by it.

Prompts already tell every agent that tool output is data, not instructions;
this is the second, mechanical layer of that same defence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_agent.guardrails.injection import heuristic_scan

_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
MARKER = "[removed by guardrail: instruction-like text]"

# Lines need a confident signal; a passage that merely discusses "system
# prompts" as a topic shouldn't be censored.
_LINE_THRESHOLD = 0.7


@dataclass
class SanitizeResult:
    text: str
    removed: list[str] = field(default_factory=list)  # reasons, one per removed line
    hidden_chars: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.removed or self.hidden_chars)


def sanitize_evidence(text: str) -> SanitizeResult:
    hidden = len(_INVISIBLE.findall(text)) + len(_HTML_COMMENT.findall(text))
    text = _HTML_COMMENT.sub(" ", _INVISIBLE.sub("", text))

    removed: list[str] = []
    out_lines = []
    for line in text.split("\n"):
        score, reasons = heuristic_scan(line)
        if score >= _LINE_THRESHOLD:
            removed.extend(reasons[:1])
            out_lines.append(MARKER)
        else:
            out_lines.append(line)
    return SanitizeResult("\n".join(out_lines), removed, hidden)
