import { useCallback, useEffect, useReducer, useRef } from "react";
import { parseSSEStream } from "@/lib/sse";
import { errorDetail } from "@/lib/api";
import * as store from "@/lib/chats";
import type { ChatMeta } from "@/lib/chats";
import type { AgentEvent, AgentId, NoAnswer, PlanInfo, Source, Verification } from "@/lib/events";
import { noteFromGuardrail, type AgentStep, type ToolStep } from "@/lib/trace";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  // assistant only ------------------------------------------------------------
  pending?: boolean;
  question?: string; // the user message this answers (for evidence highlighting)
  agents?: AgentStep[];
  sources?: Source[];
  plan?: PlanInfo;
  verification?: Verification;
  noAnswer?: NoAnswer;
  blocked?: "prompt_injection" | "policy";
  runId?: string;
  startedAt?: number;
  durationMs?: number;
  error?: string;
  retryable?: boolean;
}

interface State {
  chatId: string;
  messages: ChatMessage[];
  chats: ChatMeta[];
}

type Action =
  | { kind: "ask"; question: string }
  | { kind: "event"; event: AgentEvent }
  | { kind: "stopped" }
  | { kind: "open"; chatId: string; messages: ChatMessage[] }
  | { kind: "chats"; chats: ChatMeta[] };

// --- reducer helpers -------------------------------------------------------------------

function withAgent(agents: AgentStep[], agent: AgentId, fn: (a: AgentStep) => AgentStep) {
  const i = agents.findIndex((a) => a.agent === agent);
  if (i === -1) return [...agents, fn({ agent, status: "running", tools: [] })];
  return agents.map((a, j) => (j === i ? fn(a) : a));
}

function addNote(agents: AgentStep[], agent: AgentId, note: ToolStep["note"]): AgentStep[] {
  return withAgent(agents, agent, (ag) => ({
    ...ag,
    tools: [...ag.tools, { id: `note-${ag.tools.length}-${Date.now()}`, tool: "note", status: "done", note }],
  }));
}

function finish(m: ChatMessage, failed: boolean): ChatMessage {
  return {
    ...m,
    pending: false,
    durationMs: m.startedAt ? Date.now() - m.startedAt : undefined,
    agents: (m.agents ?? []).map((ag) => ({
      ...ag,
      status: ag.status === "running" ? (failed ? "error" : "done") : ag.status,
      tools: ag.tools.map((t) => (t.status === "running" ? { ...t, status: failed ? "error" : "done" } : t)),
    })),
  };
}

/** Apply one SSE event to the in-progress assistant message. */
function applyEvent(m: ChatMessage, e: AgentEvent): ChatMessage {
  const agents = m.agents ?? [];
  switch (e.type) {
    case "start":
      return { ...m, runId: e.run_id };
    case "token":
      return { ...m, content: m.content + e.delta };
    case "agent_start":
      return { ...m, agents: withAgent(agents, e.agent, (ag) => ({ ...ag, status: "running" })) };
    case "agent_end":
      return {
        ...m,
        agents: withAgent(agents, e.agent, (ag) => ({ ...ag, status: e.ok === false ? "error" : "done" })),
      };
    case "tool_start": {
      const agent = e.agent ?? agents.at(-1)?.agent ?? "planner";
      const step: ToolStep = {
        id: e.id,
        tool: e.tool,
        status: "running",
        args: (e.args as Record<string, unknown>) ?? undefined,
      };
      return {
        ...m,
        agents: withAgent(agents, agent, (ag) => ({
          ...ag,
          tools: ag.tools.some((t) => t.id === e.id)
            ? ag.tools.map((t) => (t.id === e.id ? { ...t, ...step } : t))
            : [...ag.tools, step],
        })),
      };
    }
    case "tool_end":
      return {
        ...m,
        agents: agents.map((ag) => ({
          ...ag,
          tools: ag.tools.map((t) => (t.id === e.id ? { ...t, status: e.ok ? "done" : "error" } : t)),
        })),
      };
    case "plan": {
      const { type: _t, thread_id: _id, ...plan } = e;
      return { ...m, plan };
    }
    case "escalation":
      return {
        ...m,
        agents: addNote(agents, e.from, {
          kind: "escalation",
          text:
            e.from === "verifier"
              ? "Your documents only answered part of it — checked the web for the rest"
              : "Nothing relevant in your documents — checked the web",
        }),
      };
    case "revision":
      return {
        ...m,
        content: "",
        agents: addNote(agents, "verifier", {
          kind: "revision",
          text: "Rewrote the draft so every claim matches a source",
        }),
      };
    case "degraded":
      return {
        ...m,
        agents: addNote(agents, "verifier", {
          kind: "degraded",
          text: "Language model unavailable — quoted the sources directly",
        }),
      };
    case "guardrail": {
      // Internal name removed from a web query: show the query that was really sent.
      const scrubbed =
        e.action === "private" && e.original !== undefined
          ? agents.map((ag) => ({
              ...ag,
              tools: ag.tools.map((t) =>
                t.args?.query === e.original
                  ? { ...t, args: { ...t.args, query: e.sent } }
                  : t.args?.topic === e.original
                    ? { ...t, args: { ...t.args, topic: e.sent } }
                    : t,
              ),
            }))
          : agents;
      const target: AgentId =
        e.stage === "input" && e.action === "declined"
          ? "planner"
          : e.action === "private"
            ? "web_researcher"
            : "guardrails";
      return { ...m, agents: addNote(scrubbed, target, noteFromGuardrail(e.action, e.detail)) };
    }
    case "web_search":
      // Attach what the search pipeline did to the matching tool call in the trace.
      return {
        ...m,
        agents: agents.map((ag) => {
          let i = ag.tools.length - 1;
          while (i >= 0 && !(ag.tools[i].tool === "web_search" && !ag.tools[i].web)) i--;
          if (i === -1) return ag;
          const web = { results: e.results, pagesRead: e.pages_read, fresh: e.fresh, cached: e.cached };
          return { ...ag, tools: ag.tools.map((t, j) => (j === i ? { ...t, web } : t)) };
        }),
      };
    case "verification": {
      const { type: _t, thread_id: _id, ...verification } = e;
      return { ...m, verification };
    }
    case "no_answer": {
      const { type: _t, thread_id: _id, ...noAnswer } = e;
      return { ...m, content: noAnswer.markdown, noAnswer };
    }
    case "replace_answer":
      return { ...m, content: e.text };
    case "blocked":
      return { ...m, content: e.message, blocked: e.category };
    case "sources": {
      const known = new Set((m.sources ?? []).map((s) => s.index));
      const fresh = e.sources.filter((s) => !known.has(s.index));
      return { ...m, sources: [...(m.sources ?? []), ...fresh].sort((a, b) => a.index - b.index) };
    }
    case "error":
      return { ...m, error: e.message, retryable: e.retryable ?? false };
    case "end":
      return finish(m, Boolean(m.error));
    default:
      return m;
  }
}

function updateLast(messages: ChatMessage[], fn: (m: ChatMessage) => ChatMessage): ChatMessage[] {
  const i = messages.length - 1;
  if (i < 0 || messages[i].role !== "assistant") return messages;
  return [...messages.slice(0, i), fn(messages[i])];
}

function reducer(s: State, a: Action): State {
  switch (a.kind) {
    case "ask":
      return {
        ...s,
        messages: [
          ...s.messages,
          { id: store.newId(), role: "user", content: a.question },
          {
            id: store.newId(),
            role: "assistant",
            content: "",
            pending: true,
            question: a.question,
            agents: [],
            sources: [],
            startedAt: Date.now(),
          },
        ],
      };
    case "event":
      return { ...s, messages: updateLast(s.messages, (m) => applyEvent(m, a.event)) };
    case "stopped":
      return {
        ...s,
        messages: updateLast(s.messages, (m) =>
          m.pending ? { ...finish(m, false), error: m.content ? undefined : "Stopped." } : m,
        ),
      };
    case "open":
      return { ...s, chatId: a.chatId, messages: a.messages };
    case "chats":
      return { ...s, chats: a.chats };
  }
}

// --- hook ------------------------------------------------------------------------------------

export function useChat() {
  const [state, dispatch] = useReducer(reducer, undefined, () => ({
    chatId: store.newId(),
    messages: [],
    chats: store.listChats(),
  }));
  const abortRef = useRef<AbortController | null>(null);
  const streaming = state.messages.at(-1)?.pending === true;

  // Save the conversation when an answer finishes (not when a chat is merely opened,
  // which would bump it to the top of the list).
  const wasStreaming = useRef(false);
  useEffect(() => {
    if (wasStreaming.current && !streaming && state.messages.length) {
      dispatch({ kind: "chats", chats: store.saveChat(state.chatId, state.messages) });
    }
    wasStreaming.current = streaming;
  }, [streaming, state.chatId, state.messages]);

  const send = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text) return;
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      dispatch({ kind: "ask", question: text });

      try {
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
          body: JSON.stringify({ message: text, thread_id: state.chatId }),
          signal: ac.signal,
        });
        if (!res.ok) {
          const retryable = res.status === 429 || res.status >= 500;
          throw Object.assign(new Error(await errorDetail(res, "The request")), { retryable });
        }
        for await (const ev of parseSSEStream(res, ac.signal)) dispatch({ kind: "event", event: ev });
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        const message =
          err instanceof TypeError
            ? "Couldn't reach the server. Check that the backend is running on port 8000."
            : err instanceof Error
              ? err.message
              : String(err);
        dispatch({
          kind: "event",
          event: {
            type: "error",
            thread_id: "",
            message,
            fatal: true,
            retryable: (err as { retryable?: boolean }).retryable ?? true,
          },
        });
        dispatch({ kind: "event", event: { type: "end", thread_id: "" } });
      }
    },
    [state.chatId],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ kind: "stopped" });
  }, []);

  const lastQuestion = [...state.messages].reverse().find((m) => m.role === "user")?.content;
  const retry = useCallback(() => {
    if (lastQuestion) send(lastQuestion);
  }, [lastQuestion, send]);

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ kind: "open", chatId: store.newId(), messages: [] });
  }, []);

  const openChat = useCallback((id: string) => {
    abortRef.current?.abort();
    dispatch({ kind: "open", chatId: id, messages: store.loadChat(id) });
  }, []);

  const removeChat = useCallback(
    (id: string) => {
      dispatch({ kind: "chats", chats: store.deleteChat(id) });
      if (id === state.chatId) dispatch({ kind: "open", chatId: store.newId(), messages: [] });
    },
    [state.chatId],
  );

  return { ...state, streaming, send, stop, retry, newChat, openChat, removeChat };
}

export type Chat = ReturnType<typeof useChat>;
