# Document Intelligence Platform — Study Guide

This guide explains the whole system end to end: what happens to a question, what
every module does, **why** it was built that way, and what to say about it in an
interview. Read it with the code open; file references are clickable on GitHub.

**Contents**

1. [The big picture](#1-the-big-picture)
2. [A question's journey, step by step](#2-a-questions-journey-step-by-step)
3. [Models and configuration](#3-models-and-configuration)
4. [Ingestion and storage](#4-ingestion-and-storage)
5. [Hybrid retrieval](#5-hybrid-retrieval)
6. [The 12 tools](#6-the-12-tools)
7. [The four agents and the graph](#7-the-four-agents-and-the-graph)
8. [Failure handling](#8-failure-handling)
9. [Guardrails](#9-guardrails)
10. [Streaming to the UI](#10-streaming-to-the-ui)
11. [Evaluation](#11-evaluation)
12. [Testing strategy](#12-testing-strategy)
13. [Running, configuration, Docker](#13-running-configuration-docker)
14. [Design decisions and trade-offs](#14-design-decisions-and-trade-offs)
15. [Interview questions](#15-interview-questions)
16. [Known limitations and next steps](#16-known-limitations-and-next-steps)

---

## 1. The big picture

A user uploads documents and asks questions. The system must:

1. find the right evidence (in private documents and/or on the web),
2. write an answer that cites that evidence,
3. check the answer before showing it,
4. say "I couldn't find this" honestly when the evidence isn't there,
5. stay safe: no data leaks, no prompt injection, no harmful output.

| Layer | Technology | Why this choice |
|---|---|---|
| Agent orchestration | **LangGraph** `StateGraph` | explicit graph: parallel branches, loops, conditional routing, checkpointed memory |
| Agents | LangChain `create_agent` (ReAct) + middleware | tool-calling loops with built-in error handling, call limits, model fallback |
| LLMs | **Groq**: `openai/gpt-oss-120b` (heavy), `gpt-oss-20b` (fast) | free tier, very fast inference, strong tool calling |
| Embeddings | **fastembed** `BAAI/bge-small-en-v1.5` (local, ONNX) | Groq has no embedding API; local = free, private, no PyTorch |
| Vector store | **Chroma** (persistent, cosine) | zero-setup, embedded |
| Keyword search | **rank-bm25** | exact identifiers that embeddings miss |
| Re-ranker | fastembed cross-encoder `ms-marco-MiniLM-L-6-v2` | precise relevance on a short candidate list |
| Safety classifier | **Llama Prompt Guard 2 (86M)** on Groq | fast dedicated injection/jailbreak detector |
| API | **FastAPI** + Server-Sent Events | token streaming + live agent events |
| UI | React 19 + Vite + Tailwind | chat, research trace, evidence cards, library |
| Evaluation | **RAGAS**, custom benchmarks, LangSmith (optional) | measurable quality, regression baseline |
| Memory | LangGraph `AsyncSqliteSaver` | per-thread conversation history |

## 2. A question's journey, step by step

Take *"What is mTLS, and how does Aurora use it?"*:

1. **UI** — `useAgentStream.send()` POSTs to `/api/chat/stream` and reads the SSE
   stream ([frontend/src/hooks/useAgentStream.ts](frontend/src/hooks/useAgentStream.ts)).
2. **API** — rate-limit check, then `agent_event_stream()` starts the graph with a
   fresh config: thread id (memory), run id (LangSmith), and a new
   **`CitationRegistry`** ([src/rag_agent/api.py](src/rag_agent/api.py), [agent.py](src/rag_agent/agent.py)).
3. **guard_input** — masks PII in the message (the masked text *replaces* it in
   history), runs injection heuristics + Prompt Guard. Blocked → skip to step 9.
4. **Query Planner** — calls `list_documents` to see what's indexed, then one
   structured LLM call returns a `Plan`: standalone question (resolves "it" from
   earlier turns), route = `both`, doc queries, web queries.
5. **Document Analyst ‖ Web Researcher** — run **in parallel** as ReAct agents.
   The analyst calls `search_documents("mTLS Aurora")` (hybrid search); the
   researcher calls `wikipedia_lookup("mutual TLS")`. Every result passes the
   retrieval guardrail and is registered with a turn-wide citation number.
6. **assess** — the join point: was any evidence found? None → **no_answer**.
7. **Fact Verifier · compose** — the heavy model writes the answer from the
   numbered evidence (this is the only text streamed token-by-token).
8. **Fact Verifier · verify** — `verify_citations` (exact) + `check_groundedness`
   (LLM judge) + a deterministic coverage backstop. Problems → back to compose
   with concrete feedback (max 1 revision). Partial coverage and the web not yet
   searched → send the Web Researcher after the missing part. Doesn't answer at
   all → no_answer.
9. **guard_output** — redacts secrets/IDs, blocks system-prompt leakage, then
   appends the final answer to the conversation history.
10. **UI** — tokens, agent steps, sources and the verification badge arrive as SSE
    events and render live.

## 3. Models and configuration

**[config.py](src/rag_agent/config.py)** — every setting lives in one
`pydantic-settings` class, overridable from `.env`. Two details worth knowing:
placeholder values copied from `.env.example` (`tvly-...`) are treated as unset,
and LangSmith tracing is switched off when there is no real key.

**[llms.py](src/rag_agent/llms.py)** — model routing by **role**:

| Role | Tier | Used for |
|---|---|---|
| planner | fast | routing (one small structured call) |
| worker | fast | research agents' tool loops |
| writer | heavy | the answer the user reads |
| verifier | fast | groundedness judge |
| summarizer | fast | `summarize_document` |

*Why tiers?* On Groq's free plan **each model has its own tokens-per-minute
bucket** (~8K). Spreading roles across two models doubles the usable budget, and
`with_fallback()` retries a failed call on the *other* tier — whose bucket is
usually not exhausted. `override_models()` swaps in fake models for tests.

## 4. Ingestion and storage

**[ingest.py](src/rag_agent/ingest.py)** — load → split → embed → store.

- Loaders: `pypdf` for PDFs (per-page metadata), robust text decoding for
  TXT/Markdown. Bad files (binary, encrypted or scanned PDFs, wrong type) return a
  clear per-file error instead of crashing the upload.
- Splitting: `RecursiveCharacterTextSplitter`, 700 chars, 100 overlap.
- **Idempotent re-upload**: each chunk gets a deterministic ID (hash of source +
  index + content) and a file's old chunks are deleted before the new ones are
  added — uploading twice never duplicates, uploading an edit replaces.
- Each chunk keeps `source`, `page`, `chunk_index` (used for citations and
  `get_adjacent_chunks`).
- Uploads are scanned for injection text and flagged (Section 9).

**[api.py](src/rag_agent/api.py)** sanitizes file names (path traversal),
enforces size/count limits, and runs ingestion in a thread so live chats keep
streaming. **[vectorstore.py](src/rag_agent/vectorstore.py)** wraps Chroma
(cosine distance) and keeps a `generation` counter bumped on every write — the
BM25 index uses it to know when to rebuild.

## 5. Hybrid retrieval

**[retrieval/](src/rag_agent/retrieval/)** — the core of RAG quality.

```
query ─┬─> dense: Chroma + BGE (top 20) ─┐
       │                                 ├─> RRF ─> cross-encoder (top 10) ─> gate ─> top 4
       └─> BM25 over all chunks (top 20)─┘
```

- **Why hybrid?** Embeddings understand paraphrase ("how long is data kept?") but
  blur exact identifiers (`AUR-2005`, `query.timeout_ms`). BM25 is the opposite.
  The benchmark shows it: dense finds 75% of exact-ID queries, BM25 100%.
- **BM25 tokenizer** ([bm25.py](src/rag_agent/retrieval/bm25.py)) keeps compound
  identifiers whole *and* split: `aur-2001` → `aur-2001`, `aur`, `2001`.
- **Reciprocal Rank Fusion** ([fusion.py](src/rag_agent/retrieval/fusion.py)):
  `score(d) = Σ 1 / (k + rank_r(d))`, k = 60. It fuses **ranks, not scores** —
  BM25 scores are unbounded and cosine is 0–1, so adding them would be
  meaningless. A document ranked moderately by both retrievers beats one ranked
  first by only one.
- **Cross-encoder re-ranking** ([rerank.py](src/rag_agent/retrieval/rerank.py)):
  a bi-encoder embeds query and passage separately (fast, approximate); a
  cross-encoder reads them together (slow, accurate). So: retrieve wide, re-rank
  narrow. L-6 was chosen over L-12 after measuring: ~4× faster for a small MRR loss.
- **Relevance gate** ([hybrid.py](src/rag_agent/retrieval/hybrid.py)): a chunk
  is kept if *any* signal vouches for it (re-rank ≥ 0.5, cosine ≥ 0.55, or a top-2
  keyword hit). Measured finding: small cross-encoders score some correct
  paraphrases near 0, so using re-rank alone as the gate threw good evidence away.
  The gate is a cheap recall-oriented pre-filter; the Fact Verifier makes the
  final "does this answer the question?" call.

## 6. The 12 tools

**[tools/](src/rag_agent/tools/)** — every tool registers evidence in the turn's
`CitationRegistry` and streams it to the Sources panel.

| Tool | Notes worth explaining |
|---|---|
| `list_documents` | inventory for routing decisions |
| `search_documents` | hybrid search; returns passages labelled `[n]` |
| `search_within_document` | lenient name matching: "the error codes doc" resolves to the file |
| `get_adjacent_chunks` | uses `chunk_index` to read text around a cut-off passage |
| `summarize_document` | LLM summary, registered as citable evidence |
| `compare_documents` | evidence from two documents on one aspect |
| `calculator` | **never `eval` LLM text** — walks the Python AST, allows only arithmetic |
| `web_search` | Tavily if keyed, else DuckDuckGo; failures return a status, not an exception |
| `fetch_webpage` | **SSRF guard**: blocks localhost/private/metadata IPs and non-http, re-checks redirects |
| `wikipedia_lookup` | definitions; no key needed |
| `verify_citations` | deterministic: invented numbers, uncited claim sentences |
| `check_groundedness` | structured LLM verdict: grounded, score, unsupported claims, **coverage**, missing parts |

**The citation registry** ([citations.py](src/rag_agent/citations.py)) solves a
real bug from the original code: each tool call numbered its results from [1], so
two calls produced two different "[1]"s. Now one registry per turn assigns stable
numbers, the same chunk keeps its number, and it's thread-safe for parallel
agents. It also logs every search attempt, which the no-answer response uses.

## 7. The four agents and the graph

**[agents/graph.py](src/rag_agent/agents/graph.py)** wires the nodes; state is in
[state.py](src/rag_agent/agents/state.py) (`messages` is checkpointed memory;
`findings` uses a custom reducer so parallel agents can append safely and the
planner can reset it each turn).

### Query Planner — [planner.py](src/rag_agent/agents/planner.py)
A **structured-output** step, not a ReAct loop: one call returns a validated
`Plan` (route ∈ documents / web / both / direct / decline). Cheap, predictable,
loggable. Post-checks fix bad output (e.g. no documents indexed → route to web).
If the model fails, a fallback plan searches everything.

A deterministic backstop was added after the behaviour evaluation caught the
small planner routing *"Which pizza toppings does the on-call team prefer?"* to
`direct` — answering from memory without searching. Now `direct` is only allowed
for messages that are actually small talk or about the assistant
(`is_conversational()`); any other question is researched.

### Document Analyst & Web Researcher — [researchers.py](src/rag_agent/agents/researchers.py)
**Autonomous ReAct agents** (`create_agent`) running as sub-graphs. They choose
their own tool calls. Middleware:
- `ToolErrorMiddleware` — a crashing tool becomes an error message the model can
  react to (internal details never leak).
- `ToolCallLimitMiddleware` — bounded tool calls; graceful stop.
- `ModelFallbackMiddleware` — retry on the other model tier.

If the agent still fails (provider down), the node runs the planned queries
directly with no LLM, so evidence is still collected. Their output is evidence
in the registry plus short notes — they never write the answer.

### Fact Verifier — [verifier.py](src/rag_agent/agents/verifier.py)
The **reflection pattern**: `compose` → `verify` → revise.
- `compose` writes with the heavy model; output is normalized (gpt-oss writes
  citations as `【3】`; non-breaking hyphens break copy-paste of IDs).
- `verify` combines a deterministic check, an LLM judge, and a deterministic
  **coverage backstop** (if the answer's first sentence admits the info is
  missing, it can't be "fully answered" — the small judge sometimes said it was).
- Outcomes: finalize with a confidence level · revise once with specific
  feedback · **gap-fill** (docs answered part, so fetch the rest from the web) ·
  hand over to the no-answer response.

**Why some agents autonomous and others fixed?** Research benefits from freedom
(which tool, how many searches). Routing and verification need reliability and
low cost — a fixed, structured step is cheaper, testable and auditable.

## 8. Failure handling

| Situation | Behaviour | Where |
|---|---|---|
| Nothing found anywhere | structured no-answer: what was searched (with queries), why, closest passages, next steps — **built without an LLM** | [no_answer.py](src/rag_agent/agents/no_answer.py) |
| Evidence found but doesn't answer | verifier hands over to the same no-answer response | verifier.py |
| Partial answer | answer + "Partially answered — not found: X"; web gap-filling first | verifier.py |
| Unsupported claims remain | answer + caveat listing them, confidence "low" | verifier.py, UI |
| Model rate-limited | retries with backoff, then the other tier | llms.py |
| All writer models down | extractive answer: best-matching quotes with citations — only from evidence the re-ranker confirmed | verifier.py |
| Research model down | planned searches run directly (no LLM) | researchers.py |
| Tool raises | error returned to the agent, which adapts | researchers.py |
| Web search down | reported as "unavailable" (not "no results") so the user knows retrying may help | tools/web.py, no_answer.py |
| Turn too slow | stopped at 180 s; retryable error | streaming.py |
| Any fatal error | friendly message + `retryable` flag → Retry button only when it can help | streaming.py |

Policy decision: with no evidence, the default is **not** to answer from the
model's memory (`ANSWER_WITHOUT_EVIDENCE=false`). Turning it on answers with a
"general knowledge, unverified" disclaimer.

## 9. Guardrails

**[guardrails/](src/rag_agent/guardrails/)** + **[agents/guard.py](src/rag_agent/agents/guard.py)** — defence in depth:

| Layer | Threat | Defence |
|---|---|---|
| Input | personal data sent to the LLM / web / history | regex + checksums (Luhn for cards, Verhoeff for Aadhaar) → `[EMAIL_1]` placeholders |
| Input | direct prompt injection, jailbreaks | heuristics **and** Llama Prompt Guard 2 (score ≥ 0.9 blocks); fails open to heuristics if the classifier is down |
| Input | harmful / out-of-scope requests | Planner's `decline` route → fixed refusal (no LLM improvisation) |
| API | abuse / quota exhaustion | per-client sliding-window rate limit (429 + Retry-After) |
| Retrieval | **indirect** injection in documents or web pages | strip invisible text (zero-width chars, HTML comments), neutralize instruction-like lines before any agent reads them; flag poisoned uploads |
| Output | leaking secrets / IDs | redaction of keys, cards, Aadhaar, PAN, SSN (contact info allowed by default) |
| Output | system-prompt leakage | 8-word shingle overlap with internal prompts → refusal |
| Web | internal names leaking to search engines (and polluting results — "Aurora" became "Amazon Aurora") | names specific to the library are detected from document titles/file names and stripped from web queries while documents are in play, including a public-homonym prefix; the UI shows the query actually sent ([privacy.py](src/rag_agent/privacy.py)) |

Two design points: checksums keep false positives low (byte counts like
`52428800` aren't flagged as cards), and `guard_output` is the **single exit** —
only checked answers are ever written to memory.

## 10. Streaming to the UI

[streaming.py](src/rag_agent/streaming.py) turns LangGraph's
`astream(stream_mode=["messages","updates","custom"], subgraphs=True)` into SSE:

| SSE event | Source |
|---|---|
| `token` | LLM tokens from the `compose` node only |
| `agent_start` / `agent_end` | custom events emitted by each agent |
| `tool_start` / `tool_end` | sub-agent updates (namespace tells which agent) or direct tool calls |
| `sources` | tools, with citation numbers and retrieval signals |
| `plan`, `escalation`, `revision`, `verification` | planner / router / verifier |
| `no_answer`, `blocked`, `guardrail`, `replace_answer`, `degraded` | failure handling and guardrails |
| `error` (`retryable`) / `end` | stream lifecycle |

`subgraphs=True` matters: without it, tool calls inside the research sub-agents
would be invisible.

**How the UI uses them** ([frontend/src/hooks/useChat.ts](frontend/src/hooks/useChat.ts)):
each event updates the answer it belongs to, so every message carries its own
trace, sources and verification, and past chats are saved in the browser.
- The **research trace** turns agent/tool events into plain language ("Searched
  for “AUR-2005”") — live while the agents work, then folded into a one-line
  summary. Agent names stay visible as small tags.
- **Citations are renumbered** 1…n in reading order (the backend numbers
  evidence turn-wide), and only cited sources are shown; the rest sit behind
  "also searched".
- **Evidence cards**: hovering a citation shows the passage window that best
  matches the question, with the question's terms highlighted.
- The upload dialog renders through a **portal** into `<body>`: an ancestor
  with `backdrop-filter` creates a containing block for `position: fixed`
  elements, which is what clipped the original dialog into the header.

## 11. Evaluation

Three complementary layers:

1. **Retrieval benchmark** ([evals/retrieval_benchmark.py](evals/retrieval_benchmark.py)) —
   no LLM, deterministic, seconds to run. Hit@1, Hit@k, MRR and rejection rate
   for dense / BM25 / hybrid / hybrid+rerank on labelled queries.
2. **RAGAS** ([evals/ragas_eval.py](evals/ragas_eval.py)) on answerable questions
   from the [golden set](evals/golden_set.py):
   - *faithfulness* — share of answer claims supported by the retrieved contexts
     (hallucination detector);
   - *answer relevancy* — the judge regenerates questions from the answer and
     compares them to the real question by embedding similarity;
   - *context precision* — are relevant contexts ranked above irrelevant ones;
   - *context recall* — does the context contain what the reference answer needs.
   The judge is **Qwen** (a different model family from the gpt-oss system under
   test) to avoid self-preference bias, on its own rate-limit bucket.
3. **Behaviour checks** — answer/citation rates, verifier confidence, revision
   rate, latency, **abstention rate** on unanswerable questions and **block
   rate** on adversarial ones. RAGAS alone would never notice a system that
   confidently answers questions it should refuse.

Results are written to `evals/results/RAGAS_REPORT.md`. The runner caches agent
runs and RAGAS scores (keyed by a hash of each sample), so a run interrupted by
Groq's daily cap resumes where it stopped, and `--rerun <id>` re-runs one question.

The behaviour checks earned their place on the first real run: RAGAS scored the
answerable questions well, but the behaviour table showed two unanswerable
questions handled through the wrong route (one answered from memory without
searching, one wrongly refused as "confidential"). Both led to planner fixes
and regression tests.
`evals/run_evals.py` / `run_pairwise.py` are optional LangSmith experiments
(LLM-as-judge correctness, heavy-vs-fast writer comparison).

## 12. Testing strategy

135 tests run **offline** in ~40 s ([tests/](tests/)).

- **Scripted fake models** ([tests/fakes.py](tests/fakes.py)): replies are
  scripted *per agent* (matched by system prompt) because the Document Analyst
  and Web Researcher run in parallel and share a model — one global queue would
  race. Fakes support tool calls, structured output and streaming.
- Agent tests drive the **real graph** through the **real SSE layer**, asserting
  on the events the UI would receive: parallel research, revision loop,
  escalation, gap-filling, every failure path, guardrail blocking, memory.
- Unit tests cover RRF maths, tokenizer, SSRF cases, calculator escapes, PII
  checksums (and their false positives), sanitizer, rate limiter.
- Regression tests pin down bugs found in live Groq runs (citation format,
  coverage backstop, placeholder keys).

## 13. Running, configuration, Docker

```bash
cp .env.example .env            # set GROQ_API_KEY
uv sync && uv run uvicorn rag_agent.api:app --reload --port 8000
cd frontend && npm install && npm run dev
```

Key settings (all in `.env.example`, documented in `config.py`):

| Setting | Default | Effect |
|---|---|---|
| `MODEL_HEAVY` / `MODEL_FAST` | gpt-oss-120b / 20b | model tiers |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `dense` / `bm25` / `hybrid` / `hybrid_rerank` |
| `RERANKER_MODEL` | MiniLM-L-6 | L-12 for accuracy over speed |
| `WORKER_MAX_STEPS` | 3 | tool calls per research agent |
| `MAX_REVISIONS` | 1 | verifier rewrites |
| `WEB_FALLBACK` | true | docs-only plans may escalate to the web |
| `ANSWER_WITHOUT_EVIDENCE` | false | strict grounding policy |
| `GUARDRAILS_ENABLED`, `PII_MASKING`, `PROMPT_GUARD_ENABLED` | true | guardrail layers |
| `RATE_LIMIT_PER_MINUTE` | 20 | per-client chat requests |

**Docker**: `docker compose up --build`. The backend image installs from
`uv.lock` (reproducible); the Chroma index, chat memory and model cache live in
a named volume; nginx serves the UI and proxies `/api` with buffering off so SSE
streams.

**Troubleshooting**: "rate-limited" → wait a minute (free tier, ~8K tokens/min
per model); first start is slow → models downloading; old index after changing
the embedding model → the collection name includes the model family, or delete
`chroma_db/`.

## 14. Design decisions and trade-offs

- **Structured routing vs a free ReAct supervisor** — cheaper, deterministic,
  testable; loses some flexibility for unusual questions.
- **Recall-oriented retrieval gate + LLM verification** — measured that no single
  retrieval score separates hard answerable questions from off-topic ones; the
  final call is made with the answer in hand.
- **Deterministic no-answer** — the most failure-prone moment (no evidence) uses
  no LLM at all, so it can't hallucinate.
- **Deterministic backstops over the LLM judge** — citation checks and the
  coverage backstop catch what the small judge misses, for free.
- **Two model tiers + fallback** — doubles free-tier throughput and gives
  resilience; the cost is managing per-role prompts and budgets.
- **Fail-open Prompt Guard** — availability over strictness when the classifier
  is down; heuristics still apply. A bank would choose fail-closed.
- **Local embeddings and re-ranker** — free and private, at the cost of CPU
  latency (~0.8 s re-ranking on a laptop).
- **In-memory BM25 and rate limiter** — simple and correct for one instance;
  multiple replicas would need OpenSearch/Redis.

## 15. Interview questions

**Q: Why hybrid search instead of just embeddings?**
Embeddings capture meaning but blur exact tokens; BM25 matches exact tokens but
not paraphrases. Fusing them with RRF gave MRR 0.797 vs 0.552 on our benchmark,
and exact-ID recall went from 75% to 100%.

**Q: Why RRF rather than a weighted sum of scores?**
Scores from BM25 and cosine similarity live on different, query-dependent scales;
ranks are comparable. RRF needs no tuning beyond k (60, from the original paper).

**Q: What does the cross-encoder add?**
It reads query and passage together, so it judges relevance far better than
vector similarity — but it's too slow for the whole corpus, so it only re-ranks
the top 10 fused candidates.

**Q: How do you prevent hallucinations?**
Layers: answer only from numbered evidence; deterministic citation check; LLM
groundedness judge; deterministic coverage backstop; one revision with concrete
feedback; caveats for anything still unsupported; and a no-LLM response when there
is no evidence. Measured with RAGAS faithfulness.

**Q: What happens when the documents don't contain the answer?**
Escalate to the web if allowed; if nothing is found, return a structured response
listing what was searched, why nothing matched, the closest passages, and next
steps — never a guess.

**Q: How do the agents share information?**
LangGraph state for control data (plan, findings, answer) and a per-turn citation
registry for evidence, passed through the run config and lock-protected because
the research agents run in parallel threads.

**Q: Why are some agents ReAct and others not?**
Research benefits from autonomy; routing and verification benefit from
predictability and low cost. Autonomy where it adds value, structure elsewhere.

**Q: How do you handle prompt injection?**
Direct: heuristics + Llama Prompt Guard on input. Indirect (in documents/web
pages): sanitize evidence before any agent reads it, strip hidden text, flag
poisoned uploads. Plus prompts that treat tool output as data, and output checks
for prompt leakage.

**Q: How do you evaluate the system?**
Retrieval benchmark (no LLM), RAGAS for answer/context quality with a judge from
a different model family, and behaviour checks for abstention and blocking — the
cases RAGAS can't see. Latest run: faithfulness 0.952, answer relevancy 0.912,
context precision 0.894, context recall 0.958; 100% abstention on unanswerable
and 100% blocking on adversarial questions (after fixing two routing bugs that
the behaviour checks found).

**Q: How did you test LLM code deterministically?**
Scripted fake chat models keyed by agent, driving the real graph and the real SSE
layer; 135 tests run offline in under a minute.

**Q: What would you change for production?**
Managed vector DB with native hybrid search, Redis-backed rate limiting and
caching, authentication and per-user document isolation, fail-closed safety
classification, streaming-time redaction, async evaluation in CI.

## 16. Known limitations and next steps

- Answer tokens stream before the output guardrail runs; the UI replaces the text
  with the redacted version at the end (a stream-time redactor would close this).
- SSRF check and fetch resolve DNS separately (DNS-rebinding window).
- BM25 index and rate limiter are in-memory, per process.
- No authentication or per-user document isolation — single-tenant demo.
- The evaluation corpus is small and fictional; numbers are a regression
  baseline. A larger, real corpus would make the RAGAS scores more meaningful.
