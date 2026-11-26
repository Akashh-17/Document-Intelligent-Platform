import { useCallback, useReducer, useRef } from "react";
import { parseSSEStream } from "@/lib/sse";
import { errorDetail } from "@/lib/api";
import type {
  AgentEvent,
  AgentId,
  NoAnswer,
  PlanInfo,
  Source,
  StepStatus,
  Verification,
} from "@/lib/events";

export interface ToolStep {
  id: string;
  tool: string; // tool name, or "note" for agent annotations
  status: StepStatus;
  detail?: string;
}

export interface AgentStep {
  agent: AgentId;
  label: string;
  status: StepStatus;
  detail?: string;
  summary?: string;
  tools: ToolStep[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  /** LangSmith root run id — lets the user leave feedback on this answer. */
  runId?: string;
  verification?: Verification;
  noAnswer?: NoAnswer;
  blocked?: "prompt_injection" | "policy";
}

interface StreamState {
  status: "idle" | "streaming" | "done" | "error";
  messages: ChatMessage[];
  answer: string; // live, in-progress assistant text
  agents: AgentStep[];
  sources: Source[];
  plan?: PlanInfo;
  verification?: Verification;
  noAnswer?: NoAnswer;
  blocked?: "prompt_injection" | "policy";
  error?: string;
  retryable?: boolean;
  threadId?: string;
  runId?: string;
}

const initial: StreamState = {
  status: "idle",
  messages: [],
  answer: "",
  agents: [],
  sources: [],
};

const LABELS: Record<AgentId, string> = {
  guardrails: "Guardrails",
  planner: "Query Planner",
  doc_analyst: "Document Analyst",
  web_researcher: "Web Researcher",
  verifier: "Fact Verifier",
};

type Action =
  | { kind: "user"; text: string }
  | { kind: "event"; event: AgentEvent }
  | { kind: "abort" };

function commitAnswer(s: StreamState): ChatMessage[] {
  const content = s.noAnswer ? s.noAnswer.markdown : s.answer;
  return content
    ? [
        ...s.messages,
        {
          role: "assistant",
          content,
          runId: s.runId,
          verification: s.verification,
          noAnswer: s.noAnswer,
          blocked: s.blocked,
        },
      ]
    : s.messages;
}

/** Apply ``fn`` to one agent's step, creating the step if it doesn't exist yet. */
function withAgent(s: StreamState, agent: AgentId, fn: (a: AgentStep) => AgentStep): AgentStep[] {
  const i = s.agents.findIndex((a) => a.agent === agent);
  if (i === -1) {
    const fresh: AgentStep = { agent, label: LABELS[agent], status: "running", tools: [] };
    return [...s.agents, fn(fresh)];
  }
  return s.agents.map((a, j) => (j === i ? fn(a) : a));
}

function note(id: string, detail: string): ToolStep {
  return { id, tool: "note", status: "done", detail };
}

function reducer(s: StreamState, a: Action): StreamState {
  if (a.kind === "user") {
    return {
      ...s,
      status: "streaming",
      messages: [...s.messages, { role: "user", content: a.text }],
      answer: "",
      agents: [],
      sources: [],
      plan: undefined,
      verification: undefined,
      noAnswer: undefined,
      blocked: undefined,
      error: undefined,
      retryable: undefined,
      runId: undefined,
    };
  }

  if (a.kind === "abort") {
    return { ...s, status: "idle", messages: commitAnswer(s), answer: "" };
  }

  const e = a.event;
  switch (e.type) {
    case "start":
      return { ...s, threadId: e.thread_id, runId: e.run_id };
    case "token":
      return { ...s, status: "streaming", answer: s.answer + e.delta };
    case "agent_start":
      return {
        ...s,
        agents: withAgent(s, e.agent, (ag) => ({
          ...ag,
          label: e.label || ag.label,
          status: "running",
          detail: e.detail,
        })),
      };
    case "agent_end":
      return {
        ...s,
        agents: withAgent(s, e.agent, (ag) => ({
          ...ag,
          status: e.ok === false ? "error" : "done",
          summary: e.summary,
        })),
      };
    case "tool_start": {
      const agent = e.agent ?? s.agents.at(-1)?.agent ?? "planner";
      const step: ToolStep = { id: e.id, tool: e.tool, status: "running" };
      return {
        ...s,
        agents: withAgent(s, agent, (ag) => ({
          ...ag,
          tools: ag.tools.some((t) => t.id === e.id)
            ? ag.tools.map((t) => (t.id === e.id ? step : t))
            : [...ag.tools, step],
        })),
      };
    }
    case "tool_end":
      return {
        ...s,
        agents: s.agents.map((ag) => ({
          ...ag,
          tools: ag.tools.map((t) =>
            t.id === e.id ? { ...t, status: e.ok ? "done" : "error", detail: e.result_preview } : t,
          ),
        })),
      };
    case "plan": {
      const { type: _t, thread_id: _id, ...plan } = e;
      return { ...s, plan };
    }
    case "escalation":
      return {
        ...s,
        agents: withAgent(s, e.from, (ag) => ({
          ...ag,
          tools: [...ag.tools, note(`escalate-${ag.tools.length}`, e.reason)],
        })),
      };
    case "revision":
      // The verifier rejected the draft: clear it, the revised answer streams next.
      return {
        ...s,
        answer: "",
        agents: withAgent(s, "verifier", (ag) => ({
          ...ag,
          tools: [
            ...ag.tools,
            note(`revision-${ag.tools.length}`, `Revising the draft: ${e.reason}`),
          ],
        })),
      };
    case "verification": {
      const { type: _t, thread_id: _id, ...verification } = e;
      return { ...s, verification };
    }
    case "no_answer": {
      // Replaces any draft that streamed: nothing reliable was found.
      const { type: _t, thread_id: _id, ...noAnswer } = e;
      return { ...s, answer: "", noAnswer };
    }
    case "replace_answer":
      return { ...s, answer: e.text };
    case "guardrail":
      return {
        ...s,
        agents: withAgent(s, "guardrails", (ag) => ({
          ...ag,
          status: e.action === "blocked" || e.action === "declined" ? "error" : ag.status,
          tools: [...ag.tools, note(`guard-${ag.tools.length}`, e.detail)],
        })),
      };
    case "blocked":
      return { ...s, answer: e.message, blocked: e.category };
    case "degraded":
      return {
        ...s,
        agents: withAgent(s, "verifier", (ag) => ({
          ...ag,
          tools: [...ag.tools, note(`degraded-${ag.tools.length}`, e.reason)],
        })),
      };
    case "sources": {
      // de-dup by citation number (the same chunk can be retrieved twice)
      const seen = new Set(s.sources.map((x) => x.index));
      const fresh = e.sources.filter((x) => !seen.has(x.index));
      const sources = [...s.sources, ...fresh].sort((x, y) => x.index - y.index);
      return { ...s, sources };
    }
    case "error":
      return {
        ...s,
        status: e.fatal ? "error" : s.status,
        error: e.message,
        retryable: e.retryable ?? false,
      };
    case "end":
      return {
        ...s,
        status: s.status === "error" ? "error" : "done",
        messages: commitAnswer(s),
        answer: "",
        agents: s.agents.map((ag) =>
          ag.status === "running" ? { ...ag, status: s.status === "error" ? "error" : "done" } : ag,
        ),
      };
    default:
      return s;
  }
}

export function useAgentStream() {
  const [state, dispatch] = useReducer(reducer, initial);
  const abortRef = useRef<AbortController | null>(null);
  const threadRef = useRef<string | undefined>(undefined);

  const send = useCallback(async (message: string) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    dispatch({ kind: "user", text: message });

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ message, thread_id: threadRef.current }),
        signal: ac.signal,
      });
      if (!res.ok) throw new Error(await errorDetail(res, "Chat request"));

      for await (const ev of parseSSEStream(res, ac.signal)) {
        if (ev.type === "start") threadRef.current = ev.thread_id;
        dispatch({ kind: "event", event: ev });
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        dispatch({
          kind: "event",
          event: {
            type: "error",
            thread_id: "",
            message: err instanceof Error ? err.message : String(err),
            fatal: true,
          },
        });
        dispatch({ kind: "event", event: { type: "end", thread_id: "" } });
      }
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ kind: "abort" });
  }, []);

  /** Re-ask the last question after a retryable failure (rate limit, timeout). */
  const lastQuestion = [...state.messages].reverse().find((m) => m.role === "user")?.content;
  const retry = useCallback(() => {
    if (lastQuestion) send(lastQuestion);
  }, [lastQuestion, send]);

  return { ...state, send, stop, retry };
}
