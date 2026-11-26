"""Keep internal names out of public web searches.

When a question involves the user's private documents, the Web Researcher's
job is to look up *general* knowledge (what is mTLS?). Small models tend to
append the document's product name anyway ("Aurora mTLS"), which:

1. returns results about an unrelated public product with the same name
   (Amazon Aurora), polluting the evidence, and
2. leaks internal project names to a third-party search engine.

So, while documents are in play, names that are specific to the library are
removed from web queries before they leave the server.

Which names are "internal": distinctive capitalised words that appear in the
titles / file names of at least two documents (e.g. "Aurora" across all the
sample docs), minus generic document words — plus anything listed explicitly
in ``PRIVATE_TERMS``. Trade-off: if a library is *about* a public technology
(say, a set of Kubernetes guides), its name would also be held back while those
documents are in play; set ``WEB_QUERY_PRIVACY=false`` to opt out.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from rag_agent import vectorstore
from rag_agent.config import settings

CONFIG_KEY = "private_names"

# Words that show up in document titles without naming anything private.
_GENERIC = frozenset(
    """
    about access account analysis analytics annual api app application architecture
    asked audit backend best book brief budget business case changelog chapter
    checklist client code company configuration contract customer data design
    developer development doc docs document documentation draft engineering error
    errors faq faqs final financial frequently guide guidelines handbook help how
    incident internal introduction issue issues knowledge manual meeting memo minutes
    note notes notice onboarding operations overview page paper plan platform policy
    policies postmortem presentation procedure process product project proposal
    quarterly questions readme reference release report requirements research
    review roadmap rules runbook sales security service services spec
    specification standard statement strategy summary support system team technical
    template terms test the training update user users version whitepaper
    """.split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


def _doc_tokens(source: str, title: str) -> set[str]:
    stem = re.sub(r"\.[a-z0-9]+$", "", source, flags=re.I)
    words = _WORD.findall(stem.replace("_", " ")) + [
        w for w in _WORD.findall(title) if w[0].isupper()
    ]
    return {w.lower() for w in words if w.lower() not in _GENERIC}


@lru_cache(maxsize=4)
def _names_for(generation: int) -> tuple[str, ...]:
    stats = vectorstore.source_stats()
    counts = Counter()
    for d in stats:
        counts.update(_doc_tokens(d["source"], d.get("title") or ""))
    auto = {w for w, n in counts.items() if n >= 2}
    return tuple(sorted(auto))


def internal_names() -> list[str]:
    """Names specific to the current library (cached until the library changes)."""
    if not settings.web_query_privacy:
        return []
    explicit = {t.strip().lower() for t in settings.private_terms.split(",") if t.strip()}
    try:
        auto = set(_names_for(vectorstore.generation()))
    except Exception:  # noqa: BLE001 — privacy scrubbing must never break search
        auto = set()
    return sorted(auto | explicit)


# Sentence-initial Title-case words that are never part of a product name.
_NOT_A_PREFIX = (
    "What|How|Why|When|Where|Which|Who|Does|Do|Is|Are|Can|Explain|Describe|Define|"
    "Compare|Show|List|Tell|Find|Search|Summarize|Use|Using|In|On|For|With|About|The|An|A"
)


def scrub_query(query: str, names: list[str]) -> tuple[str, list[str]]:
    """Remove ``names`` (whole words, any case, incl. possessives) from a query.

    A Title-case word right before the name goes too: models often "complete" an
    internal name into a public homonym ("Amazon Aurora"), and dropping only the
    name would leave a misleading "Amazon ..." query. The prefix check is
    case-sensitive (only the name itself matches in any case), so acronyms
    (TLS) and ordinary words ("does", "Explain") are kept.
    """
    removed: list[str] = []
    for name in names:
        pattern = re.compile(
            rf"(?:\b(?!(?:{_NOT_A_PREFIX})\b)[A-Z][a-z]+\s+)?"
            rf"\b(?i:{re.escape(name)}(?:'s|’s)?)\b"
        )
        if pattern.search(query):
            removed.append(name)
            query = pattern.sub(" ", query)
    query = re.sub(r"\s+([?.!,])", r"\1", re.sub(r"\s{2,}", " ", query)).strip(" -,:;")
    return query, removed
