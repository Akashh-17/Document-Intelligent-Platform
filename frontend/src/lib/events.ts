// SSE event contract — mirrors src/rag_agent/streaming.py

export type StepStatus = "running" | "done" | "error";

export type AgentId = "guardrails" | "planner" | "doc_analyst" | "web_researcher" | "verifier";

export interface Source {
  id: string;
  /** Turn-wide citation number — matches the [n] markers in the answer. */
  index: number;
  kind: "document" | "web";
  title: string;
  url?: string | null;
  snippet: string;
  score?: number | null;
  /** Set when the retrieval guardrail removed instruction-like text. */
  guardrail?: string;
  /** Hybrid-retrieval signals (document sources only). */
  retrieval?: {
    dense_rank?: number | null;
    bm25_rank?: number | null;
    rerank_score?: number | null;
    dense_score?: number | null;
    rrf_score?: number | null;
  };
}

export type Route = "documents" | "web" | "both" | "direct" | "decline" | "blocked";

export interface PlanInfo {
  route: Route;
  question: string;
  doc_queries: string[];
  web_queries: string[];
  rationale?: string;
}

export type Confidence =
  | "high"
  | "medium"
  | "low"
  | "none"
  | "unverified"
  | "extractive"
  | "n/a";

export interface Verification {
  checked: boolean;
  confidence: Confidence;
  grounded?: boolean | null;
  score?: number | null;
  unsupported_claims?: string[];
  invalid_citations?: number[];
  revisions?: number;
  evidence_count?: number;
  coverage?: "full" | "partial" | "none";
  missing_aspects?: string[];
  model_knowledge?: boolean;
}

/** Structured "I couldn't find this" response (see agents/no_answer.py). */
export interface NoAnswer {
  reason: "not_found" | "web_unavailable" | "no_documents" | "service_unavailable";
  headline: string;
  explanation: string;
  searched: {
    where: string;
    status: "empty" | "not_searched" | "no_results" | "irrelevant" | "unavailable";
    summary: string;
    queries: string[];
  }[];
  near_misses: { title: string; snippet: string; score?: number; url?: string | null }[];
  suggestions: string[];
  markdown: string;
}

type T = { thread_id: string };

export type AgentEvent =
  | ({ type: "start"; run_id: string } & T)
  | ({ type: "token"; delta: string } & T)
  | ({ type: "agent_start"; agent: AgentId; label: string; detail?: string } & T)
  | ({ type: "agent_end"; agent: AgentId; summary?: string; ok?: boolean } & T)
  | ({ type: "tool_start"; agent?: AgentId; id: string; tool: string; args?: unknown } & T)
  | ({
      type: "tool_end";
      agent?: AgentId;
      id: string;
      tool: string;
      ok: boolean;
      result_preview?: string;
    } & T)
  | ({ type: "sources"; tool: string; sources: Source[] } & T)
  | ({ type: "plan" } & PlanInfo & T)
  | ({ type: "escalation"; from: AgentId; to: AgentId; reason: string } & T)
  | ({ type: "revision"; reason: string } & T)
  | ({ type: "verification" } & Verification & T)
  | ({ type: "no_answer" } & NoAnswer & T)
  | ({ type: "degraded"; reason: string } & T)
  | ({ type: "replace_answer"; text: string } & T)
  | ({
      type: "guardrail";
      stage: "input" | "retrieval" | "output";
      action: "masked" | "blocked" | "declined" | "neutralized" | "redacted";
      detail: string;
    } & T)
  | ({ type: "blocked"; category: "prompt_injection" | "policy"; message: string; reasons: string[] } & T)
  | ({ type: "error"; message: string; fatal?: boolean; retryable?: boolean } & T)
  | ({ type: "end"; run_id?: string } & T);
