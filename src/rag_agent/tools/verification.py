"""Fact Verifier tools — check a drafted answer before the user relies on it.

| Tool               | What it does                                               |
|--------------------|------------------------------------------------------------|
| verify_citations   | Deterministic: every [n] exists, and claims are cited      |
| check_groundedness | LLM judge: is each claim supported by the cited evidence?  |

Two complementary checks: the first is free and exact but only sees the
*form* of citations; the second understands *meaning* but costs a model call.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag_agent.llms import with_fallback

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Sentences worth checking: long enough to be a factual claim, not a heading.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    out = []
    for s in _SENTENCE_SPLIT.split(text):
        s = s.strip().lstrip("-*• ").strip()
        if len(s) >= 25 and not s.startswith("#"):
            out.append(s)
    return out


@tool
def verify_citations(answer: str, available: list[int]) -> dict:
    """Check citation markers: invented numbers, and factual sentences with no citation.

    ``available`` is the list of citation numbers that really exist this turn.
    """
    cited: set[int] = set()
    for group in _CITE.findall(answer):
        cited.update(int(n) for n in re.split(r"\s*,\s*", group))
    invalid = sorted(cited - set(available))
    uncited = [s for s in _sentences(answer) if not _CITE.search(s)]
    return {
        "cited": sorted(cited),
        "invalid": invalid,
        "uncited_sentences": uncited[:5],
        "has_citations": bool(cited),
        "ok": not invalid,
    }


class GroundednessVerdict(BaseModel):
    """Structured output of the groundedness judge."""

    grounded: bool = Field(description="True if every factual claim is supported by the evidence")
    score: float = Field(ge=0, le=1, description="Share of claims supported, 0 to 1")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims NOT supported by the evidence, quoted briefly (max 5)",
    )
    wrong_citations: list[str] = Field(
        default_factory=list,
        description="Claims whose [n] points to evidence that doesn't support them (max 5)",
    )
    coverage: Literal["full", "partial", "none"] = Field(
        default="full",
        description="How much of the QUESTION the answer actually answers with evidence: "
        "full, partial, or none (e.g. it only says the information wasn't found).",
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Parts of the question left unanswered (max 3, short phrases)",
    )


_JUDGE_PROMPT = """You are a strict fact-checker. Compare the ANSWER with the EVIDENCE.

For each factual claim in the answer, decide whether the evidence supports it.
- A claim is supported only if the evidence states it or it follows directly.
- General knowledge that is NOT in the evidence counts as unsupported.
- Statements that information is missing ("the documents don't say...") are fine.
- If a claim cites [n], check that passage [n] is the one that supports it.
- Coverage: does the answer actually answer EVERY part of the QUESTION from the
  evidence? "full" only if all parts are answered. "partial" if any part is
  unanswered — including when the answer itself says something "isn't in the
  documents"; list those parts in missing_aspects. "none" if it only says the
  information isn't available, or answers a different question.

Be concise: quote at most 5 unsupported claims and 5 wrong citations."""


@tool
def check_groundedness(
    answer: str, evidence: str, config: RunnableConfig, question: str = ""
) -> dict:
    """LLM judge: are the answer's claims supported by the numbered evidence,
    and how much of the question does it answer?"""
    judge = with_fallback("verifier", lambda m: m.with_structured_output(GroundednessVerdict))
    try:
        verdict = judge.invoke(
            [
                SystemMessage(_JUDGE_PROMPT),
                HumanMessage(
                    f"QUESTION:\n{question or '(not given)'}\n\n"
                    f"EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}"
                ),
            ],
            config,
        )
    except Exception as exc:  # noqa: BLE001 — a failed judge must not sink the answer
        return {"error": f"{type(exc).__name__}: {exc}", "grounded": None}
    if isinstance(verdict, dict):  # some providers return dicts
        verdict = GroundednessVerdict.model_validate(verdict)
    data = verdict.model_dump()
    data["unsupported_claims"] = data["unsupported_claims"][:5]
    data["wrong_citations"] = data["wrong_citations"][:5]
    data["missing_aspects"] = data["missing_aspects"][:3]
    return data


VERIFICATION_TOOLS = [verify_citations, check_groundedness]
