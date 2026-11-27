"""Intent router — decides, before any LLM call, whether a message needs research.

A chat assistant must answer "hey wassup" like a person, not run a web search
for it. Leaving that call to the planner LLM failed in practice: a small model
turned "thank you" into a search for "how to respond to thank you". So routing
has a deterministic front door, in two layers:

1. **Lexicon** — the whole message is made of pleasantries ("thanks a lot!",
   "ok cool", "hey wassup"). Exact, instant, explainable.
2. **Semantic router** — the message is embedded with the same local model the
   retriever uses (bge-small, ~5 ms, no API call) and compared with labelled
   example utterances. It catches what a word list can't: "how's your day
   going", "you're a lifesaver", "make that shorter", typos and slang.

Three intents come out:

- ``chat``    small talk or a question about the assistant → reply directly.
- ``rework``  reshape the previous answer (shorter, simpler, translate, as a
              table) → rewrite it from the SAME evidence, no new search.
- ``task``    anything else → the Query Planner decides how to research it.

The router only short-circuits when it is confident; anything ambiguous falls
through to the planner LLM. Message length caps keep a long question that
merely starts with "hey" from being mistaken for small talk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

Intent = Literal["chat", "rework", "task"]

# --- layer 1: lexicon ------------------------------------------------------------------

_PLEASANTRY = re.compile(
    r"\b(?:"
    # greetings & slang
    r"hi+|hello+|hey+|heya|hiya|howdy|yo+|sup|wass+up|wazzup|whatsup|what'?s up|hola|namaste|"
    r"good (?:morning|afternoon|evening|night|day)|gm|gn|"
    # how are you
    r"how (?:are|r) (?:you|u|ya)(?: doing)?|how'?s it going|how you doing|"
    r"hope you'?re (?:well|good)|"
    r"nice to meet you|"
    # thanks & praise
    r"thank you|thank u|thanks|thanx|thnx|thx|ty|tysm|cheers|much appreciated|appreciate it|"
    r"great job|good job|well done|nice one|you'?re (?:awesome|the best|great|amazing)|"
    r"awesome|amazing|perfect|brilliant|great|nice|cool|sweet|wow|"
    # farewells & acks
    r"bye+|goodbye|see you|see ya|later|take care|ok+|okay|okie|kk|alright|got it|"
    r"makes sense|understood|noted|no worries|np|"
    # acknowledging an answer ("ohhh i see", "ah got it", "interesting, makes sense")
    r"o+h+|a+h+|i see|i get it|i understand|gotcha|interesting|true|right|fair enough|"
    r"yes+|yeah+|yep|yup|sure|exactly|nice|"
    # reactions
    r"lol+|lmao|haha+|hehe+|hmm+|"
    # fillers & addressees
    r"so much|very much|a lot|again|for (?:the|your) help|for that|that'?s all|"
    r"bro|bruh|dude|man|mate|buddy|there|everyone|all|guys|"
    r"and|to|you|too|as well|doing|today"
    r")\b"
)


def is_small_talk(message: str) -> bool:
    """True only when the WHOLE message is pleasantries — nothing left to research.

    "thanks, what does AUR-2005 mean?" starts with thanks but leaves a real
    question behind, so it does not match.
    """
    text = message.lower().replace("’", "'").strip()
    if not text or len(text) > 80:
        return False
    words = re.sub(r"[^a-z' ]+", " ", text)
    if not words.split():
        return False
    leftover = _PLEASANTRY.sub(" ", words).replace("'", " ").split()
    return not leftover


# --- layer 2: semantic router -------------------------------------------------------------

EXAMPLES: dict[Intent, list[str]] = {
    "chat": [
        "hey wassup",
        "hello there",
        "hi, how are you?",
        "how's your day going",
        "what's new with you",
        "good morning!",
        "yo what's good",
        "thanks, that was really helpful",
        "you're a lifesaver",
        "that's awesome, appreciate it",
        "lol nice",
        "ok bye, see you later",
        "who are you?",
        "what can you do?",
        "what are you able to help me with",
        "who built you",
        "are you an AI or a human?",
        "what's your name",
        "how do you work?",
        "tell me about yourself",
        "I'm bored",
        "can we just chat for a bit",
        "oh I see, that makes sense",
        "ahh okay, got it now",
        "interesting, I didn't know that",
        "that's really cool",
        "I'm good, how about you?",
        "not much, what about you",
        "you rock",
        "are you ChatGPT?",
    ],
    "rework": [
        "make it shorter",
        "can you summarize that in two lines",
        "explain that more simply",
        "explain it like I'm five",
        "put that in bullet points",
        "turn your answer into a table",
        "translate that to Hindi",
        "say it in Spanish",
        "rewrite that more formally",
        "make it sound more casual",
        "write that as an email to my manager",
        "give me a TL;DR of your answer",
        "can you rephrase that",
        "expand on your last answer",
    ],
    "task": [
        "what is mutual TLS",
        "how do I fix error AUR-2005",
        "summarize the onboarding document",
        "what does the policy say about refunds",
        "latest news about the stock market",
        "who won the cricket world cup",
        "compare PostgreSQL and MySQL",
        "explain how transformers work",
        "what's the weather in Mumbai today",
        "hey, what is retrieval augmented generation?",
        "hi, can you tell me the deadline in the contract",
        "what are the key points in my documents",
        "how much does the premium plan cost",
        "why is the sky blue",
        "what's up with the new iPhone release",
        "give me the steps to rotate a certificate",
        "find the budget numbers for 2024",
        "who is the CEO of OpenAI",
    ],
}

# Confidence rules, calibrated on held-out phrasings (tests/test_intent.py). Real
# small talk scored 0.75-0.92 against these examples; the closest real question
# ("who are you voting for") scored 0.72. Below the bar, the planner decides.
MIN_SIM = 0.75  # best match must be this close to a labelled example
MARGIN = 0.06  # ...and beat the closest "task" example by this much
MAX_WORDS = {"chat": 10, "rework": 14}  # longer messages always go to the planner


@dataclass(frozen=True)
class IntentDecision:
    intent: Intent
    score: float  # similarity to the closest example of that intent (1.0 for lexicon)
    method: Literal["lexicon", "semantic", "default"]
    nearest: str = ""  # the example it matched, for the trace / debugging


class SemanticRouter:
    """Nearest-example classifier over sentence embeddings."""

    def __init__(self, embed) -> None:
        self._embed = embed
        self._labels: list[Intent] = []
        texts: list[str] = []
        for intent, examples in EXAMPLES.items():
            self._labels += [intent] * len(examples)
            texts += examples
        self._texts = texts
        self._matrix = self._normalise(np.asarray(embed(texts), dtype=np.float32))

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)

    def scores(self, message: str) -> dict[Intent, tuple[float, str]]:
        """Best cosine similarity per intent, with the example that produced it."""
        vec = self._normalise(np.asarray(self._embed([message])[0], dtype=np.float32))
        sims = self._matrix @ vec
        best: dict[Intent, tuple[float, str]] = {}
        for label, text, sim in zip(self._labels, self._texts, sims.tolist(), strict=True):
            if label not in best or sim > best[label][0]:
                best[label] = (sim, text)
        return best


@lru_cache
def get_router() -> SemanticRouter | None:
    """The shared router, or None if the embedding model can't load (lexicon only)."""
    try:
        from rag_agent.embeddings import get_embeddings

        return SemanticRouter(get_embeddings().embed_documents)
    except Exception:  # noqa: BLE001 — routing must never break a turn
        log.exception("semantic router unavailable; using the lexicon only")
        return None


def classify(message: str, has_previous_answer: bool = False) -> IntentDecision:
    """Route a user message: chat, rework (needs a previous answer), or task."""
    text = message.strip()
    if is_small_talk(text):
        return IntentDecision("chat", 1.0, "lexicon")

    words = len(text.split())
    router = get_router()
    if router is None or not text or words > max(MAX_WORDS.values()):
        return IntentDecision("task", 0.0, "default")

    best = router.scores(text)
    task_sim = best["task"][0]
    for intent in ("chat", "rework"):
        sim, nearest = best[intent]
        if intent == "rework" and not has_previous_answer:
            continue
        if words <= MAX_WORDS[intent] and sim >= MIN_SIM and sim - task_sim >= MARGIN:
            return IntentDecision(intent, round(sim, 3), "semantic", nearest)
    return IntentDecision("task", round(task_sim, 3), "default", best["task"][1])
