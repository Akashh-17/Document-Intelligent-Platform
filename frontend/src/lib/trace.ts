/**
 * Plain-language labels for the research trace.
 *
 * The backend reports agent and tool activity in system terms
 * (search_documents, check_groundedness, rerank scores). The UI describes what
 * happened from the reader's side — "Searched your documents for “AUR-2005”" —
 * and keeps the agent names as small tags for anyone curious about the system.
 */
import type { AgentId, PlanInfo, StepStatus } from "./events";
import { hostOf } from "./citations";

export type NoteKind =
  | "escalation"
  | "revision"
  | "degraded"
  | "masked"
  | "neutralized"
  | "redacted"
  | "blocked"
  | "declined"
  | "info";

export interface ToolStep {
  id: string;
  tool: string; // tool name, or "note"
  status: StepStatus;
  args?: Record<string, unknown>;
  note?: { kind: NoteKind; text: string };
}

export interface AgentStep {
  agent: AgentId;
  status: StepStatus;
  tools: ToolStep[];
}

export const AGENT_NAMES: Record<AgentId, string> = {
  guardrails: "Guardrails",
  planner: "Query Planner",
  doc_analyst: "Document Analyst",
  web_researcher: "Web Researcher",
  verifier: "Fact Verifier",
};

function q(v: unknown): string {
  const s = String(v ?? "").trim();
  return s.length > 60 ? `“${s.slice(0, 57)}…”` : `“${s}”`;
}

export function toolLabel(step: ToolStep): string {
  const a = step.args ?? {};
  switch (step.tool) {
    case "list_documents":
      return "Reviewed your library";
    case "search_documents":
      return a.query ? `Searched for ${q(a.query)}` : "Searched your documents";
    case "search_within_document":
      return `Looked inside ${String(a.document ?? "a document")}`;
    case "get_adjacent_chunks":
      return "Read the surrounding passage";
    case "summarize_document":
      return `Summarized ${String(a.document ?? "a document")}`;
    case "compare_documents":
      return `Compared ${String(a.document_a ?? "")} with ${String(a.document_b ?? "")}`;
    case "calculator":
      return `Calculated ${String(a.expression ?? "")}`;
    case "web_search":
      return a.query ? `Searched the web for ${q(a.query)}` : "Searched the web";
    case "fetch_webpage":
      return `Read ${hostOf(String(a.url ?? "")) || "a web page"}`;
    case "wikipedia_lookup":
      return `Looked up ${q(a.topic)} on Wikipedia`;
    case "verify_citations":
      return "Checked every citation";
    case "check_groundedness":
      return "Checked each claim against the sources";
    default:
      return step.tool.replace(/_/g, " ");
  }
}

const ROUTE_PLAN: Record<string, string> = {
  documents: "Decided to search your documents",
  web: "Decided to search the web",
  both: "Decided to search your documents and the web",
  direct: "No research needed",
  decline: "Declined the request",
};

export function agentLabel(step: AgentStep, plan?: PlanInfo): string {
  const running = step.status === "running";
  switch (step.agent) {
    case "guardrails":
      return step.status === "error" ? "Stopped by safety checks" : "Checked the request";
    case "planner":
      return running ? "Understanding your question" : ROUTE_PLAN[plan?.route ?? ""] ?? "Planned the search";
    case "doc_analyst":
      return running ? "Searching your documents" : "Searched your documents";
    case "web_researcher":
      return running ? "Searching the web" : "Searched the web";
    case "verifier":
      return running ? "Writing and fact-checking" : "Wrote the answer and checked it";
  }
}

/** What the live status line says right now. */
export function liveLabel(agents: AgentStep[], plan?: PlanInfo): string {
  for (let i = agents.length - 1; i >= 0; i--) {
    const ag = agents[i];
    const tool = [...ag.tools].reverse().find((t) => t.status === "running" && t.tool !== "note");
    if (tool) return `${toolLabel(tool)}…`;
    if (ag.status === "running") return `${agentLabel(ag, plan)}…`;
  }
  return "Thinking…";
}

export function noteFromGuardrail(action: string, detail: string): { kind: NoteKind; text: string } {
  switch (action) {
    case "masked":
      return { kind: "masked", text: detail.replace(/ before sending anything.*$/, " before searching") };
    case "private":
      return { kind: "masked", text: detail };
    case "neutralized":
      return { kind: "neutralized", text: "Ignored instructions hidden inside a source" };
    case "redacted":
      return { kind: "redacted", text: "Removed sensitive data from the answer" };
    case "blocked":
      return { kind: "blocked", text: "Blocked: looks like an attempt to override instructions" };
    case "declined":
      return { kind: "declined", text: "Declined: outside what this assistant helps with" };
    default:
      return { kind: "info", text: detail };
  }
}
