import { useState } from "react";
import { Bot, User, ThumbsUp, ThumbsDown, ShieldCheck, ShieldAlert, Info } from "lucide-react";
import type { NoAnswer, Verification } from "@/lib/events";
import { NoAnswerCard } from "./NoAnswerCard";
import { cn } from "@/lib/utils";
import { sendFeedback } from "@/lib/api";
import { Markdown } from "./Markdown";

function Feedback({ runId }: { runId: string }) {
  const [state, setState] = useState<"idle" | "up" | "down" | "failed">("idle");

  const rate = async (score: number) => {
    const ok = await sendFeedback(runId, score);
    setState(ok ? (score === 1 ? "up" : "down") : "failed");
  };

  if (state === "failed") {
    return <span className="text-[11px] text-muted">Feedback not recorded (LangSmith off).</span>;
  }
  return (
    <div className="flex items-center gap-1">
      <button
        onClick={() => rate(1)}
        disabled={state !== "idle"}
        title="Helpful"
        className={cn(
          "rounded p-1 text-muted hover:bg-surface-2 hover:text-fg disabled:hover:bg-transparent",
          state === "up" && "text-ok",
        )}
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => rate(0)}
        disabled={state !== "idle"}
        title="Not helpful"
        className={cn(
          "rounded p-1 text-muted hover:bg-surface-2 hover:text-fg disabled:hover:bg-transparent",
          state === "down" && "text-danger",
        )}
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

const CONFIDENCE: Record<string, { label: string; className: string; title: string }> = {
  high: {
    label: "Verified",
    className: "text-ok",
    title: "Every claim was checked against the cited sources",
  },
  medium: {
    label: "Partly verified",
    className: "text-warn",
    title: "Most claims are supported; some could not be confirmed",
  },
  low: {
    label: "Low confidence",
    className: "text-danger",
    title: "Several claims are not supported by the sources found",
  },
  none: {
    label: "No sources found",
    className: "text-muted",
    title: "Nothing relevant was found in your documents or on the web",
  },
  unverified: {
    label: "Not verified",
    className: "text-muted",
    title: "The fact-check step could not run",
  },
  extractive: {
    label: "Quoted from sources",
    className: "text-muted",
    title: "The language model was unavailable, so these are verbatim passages",
  },
  knowledge: {
    label: "General knowledge — unverified",
    className: "text-warn",
    title: "No sources were found; this comes from the model's general knowledge",
  },
};

function ConfidenceBadge({ v }: { v: Verification }) {
  const c = CONFIDENCE[v.model_knowledge ? "knowledge" : v.confidence];
  if (!c) return null;
  const Icon = v.confidence === "high" ? ShieldCheck : ShieldAlert;
  const extra = v.revisions ? ` · revised ${v.revisions}×` : "";
  return (
    <span title={c.title} className={cn("inline-flex items-center gap-1 text-[11px]", c.className)}>
      <Icon className="h-3.5 w-3.5" />
      {c.label}
      {extra}
    </span>
  );
}

/** Honest caveats under an answer: what's missing or couldn't be verified. */
function Caveats({ v }: { v: Verification }) {
  const notes: string[] = [];
  if (v.coverage === "partial" && v.missing_aspects?.length) {
    notes.push(`Partially answered — not found in the sources: ${v.missing_aspects.join("; ")}.`);
  }
  if (v.confidence === "low" && v.unsupported_claims?.length) {
    notes.push(`Couldn't verify against the sources: ${v.unsupported_claims.join("; ")}`);
  }
  if (!notes.length) return null;
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs">
      {notes.map((n) => (
        <p key={n} className="flex items-start gap-1.5">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" />
          <span>{n}</span>
        </p>
      ))}
    </div>
  );
}

export function MessageBubble({
  role,
  content,
  streaming = false,
  runId,
  verification,
  noAnswer,
}: {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  runId?: string;
  verification?: Verification;
  noAnswer?: NoAnswer;
}) {
  const isUser = role === "user";
  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <span
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-surface-2 text-muted" : "bg-accent text-accent-fg",
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </span>
      <div className="flex max-w-[78%] flex-col gap-1">
        <div
          className={cn(
            "rounded-xl px-4 py-2.5",
            isUser ? "bg-accent text-accent-fg" : "border border-border bg-surface",
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap text-sm">{content}</p>
          ) : noAnswer ? (
            <NoAnswerCard data={noAnswer} />
          ) : (
            <>
              <Markdown text={content} highlight={!streaming} />
              {streaming && (
                <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse rounded-sm bg-accent" />
              )}
            </>
          )}
        </div>
        {!isUser && !streaming && verification && !noAnswer && <Caveats v={verification} />}
        {!isUser && !streaming && (verification || runId) && (
          <div className="flex items-center gap-3">
            {verification && !noAnswer && <ConfidenceBadge v={verification} />}
            {runId && <Feedback runId={runId} />}
          </div>
        )}
      </div>
    </div>
  );
}
