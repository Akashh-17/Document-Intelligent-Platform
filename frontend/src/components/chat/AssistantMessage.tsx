import { useMemo, useState } from "react";
import { Check, Copy, Info, RotateCcw, ShieldCheck, ShieldX, ThumbsDown, ThumbsUp } from "lucide-react";
import type { ChatMessage } from "@/hooks/useChat";
import { sendFeedback } from "@/lib/api";
import { highlightTerms, renumberCitations } from "@/lib/citations";
import { cn } from "@/lib/utils";
import { EvidenceContext } from "./Evidence";
import { Markdown } from "./Markdown";
import { NoAnswerCard } from "./NoAnswerCard";
import { ResearchTrace } from "./ResearchTrace";
import { Sources } from "./Sources";

const VERIFIED: Record<string, { label: string; tone: string; title: string } | undefined> = {
  high: { label: "Verified", tone: "text-ok", title: "Every claim was checked against the cited sources" },
  medium: {
    label: "Mostly verified",
    tone: "text-warn",
    title: "Most claims are supported by the sources; check the notes below",
  },
  low: { label: "Partly verified", tone: "text-warn", title: "Some claims couldn't be confirmed by the sources" },
  extractive: { label: "Quoted from sources", tone: "text-muted", title: "Shown as direct quotes from the sources" },
};

function Caveats({ m }: { m: ChatMessage }) {
  const v = m.verification;
  if (!v) return null;
  const notes: string[] = [];
  if (v.coverage === "partial" && v.missing_aspects?.length)
    notes.push(`Not covered by the sources: ${v.missing_aspects.join("; ")}.`);
  if (v.confidence === "low" && v.unsupported_claims?.length)
    notes.push(`Couldn't confirm: ${v.unsupported_claims.join("; ")}`);
  if (v.model_knowledge) notes.push("No sources were found, so this comes from general knowledge and is unverified.");
  if (!notes.length) return null;
  return (
    <div className="mt-4 space-y-1 rounded-xl border border-warn/25 bg-warn/[0.07] px-3.5 py-2.5 text-[13px] leading-relaxed">
      {notes.map((n) => (
        <p key={n} className="flex items-start gap-2">
          <Info className="mt-[3px] h-3.5 w-3.5 shrink-0 text-warn" />
          <span>{n}</span>
        </p>
      ))}
    </div>
  );
}

function ActionBar({ m, text, feedback, sourceCount }: { m: ChatMessage; text: string; feedback: boolean; sourceCount: number }) {
  const [copied, setCopied] = useState(false);
  const [rated, setRated] = useState<"up" | "down" | null>(null);
  const v = m.verification ? VERIFIED[m.verification.confidence] : undefined;

  const rate = async (score: number) => {
    if (!m.runId || rated) return;
    if (await sendFeedback(m.runId, score)) setRated(score ? "up" : "down");
  };

  return (
    <div className="mt-4 flex items-center gap-1 text-muted">
      {v && (
        <span title={v.title} className={cn("mr-2 inline-flex items-center gap-1.5 text-xs font-medium", v.tone)}>
          <ShieldCheck className="h-3.5 w-3.5" />
          {v.label}
          {sourceCount > 0 && m.verification?.confidence !== "extractive" && (
            <span className="font-normal text-faint">
              · {sourceCount} source{sourceCount === 1 ? "" : "s"}
            </span>
          )}
        </span>
      )}
      <button
        onClick={() =>
          navigator.clipboard.writeText(text.replace(/\[(\d+)\]/g, "")).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          })
        }
        aria-label="Copy answer"
        title="Copy"
        className="rounded-lg p-1.5 transition-colors hover:bg-surface-2 hover:text-fg"
      >
        {copied ? <Check className="h-4 w-4 text-ok" /> : <Copy className="h-4 w-4" />}
      </button>
      {feedback && m.runId && (
        <>
          <button
            onClick={() => rate(1)}
            aria-label="Good answer"
            title="Good answer"
            className={cn("rounded-lg p-1.5 transition-colors hover:bg-surface-2 hover:text-fg", rated === "up" && "text-ok")}
          >
            <ThumbsUp className="h-4 w-4" />
          </button>
          <button
            onClick={() => rate(0)}
            aria-label="Bad answer"
            title="Bad answer"
            className={cn("rounded-lg p-1.5 transition-colors hover:bg-surface-2 hover:text-fg", rated === "down" && "text-danger")}
          >
            <ThumbsDown className="h-4 w-4" />
          </button>
        </>
      )}
    </div>
  );
}

export function AssistantMessage({
  m,
  feedback,
  onRetry,
  isLast,
}: {
  m: ChatMessage;
  feedback: boolean;
  onRetry: () => void;
  isLast: boolean;
}) {
  const { text, cited, others } = useMemo(
    () => renumberCitations(m.content, m.sources ?? []),
    [m.content, m.sources],
  );
  const terms = useMemo(() => highlightTerms(m.question ?? ""), [m.question]);
  const evidence = useMemo(() => ({ byN: new Map(cited.map((s) => [s.n, s])), terms }), [cited, terms]);
  const showAnswer = !m.noAnswer && !m.blocked && text.trim().length > 0;
  const done = !m.pending;

  return (
    <article className="rise" aria-busy={m.pending}>
      <ResearchTrace message={m} />

      <EvidenceContext.Provider value={evidence}>
        {m.blocked ? (
          <div className="flex items-start gap-3 rounded-xl border border-border bg-panel px-4 py-3.5">
            <ShieldX className="mt-0.5 h-5 w-5 shrink-0 text-danger" />
            <div>
              <p className="font-medium">
                {m.blocked === "prompt_injection" ? "Blocked by safety checks" : "I can't help with that"}
              </p>
              <p className="mt-1 text-[0.95rem] leading-relaxed text-muted">{m.content}</p>
            </div>
          </div>
        ) : m.noAnswer ? (
          <NoAnswerCard data={m.noAnswer} />
        ) : showAnswer ? (
          <Markdown text={text} highlight={done} streaming={m.pending} />
        ) : null}

        {done && showAnswer && <Caveats m={m} />}
        {done && !m.blocked && !m.noAnswer && <Sources cited={cited} others={others} terms={terms} />}
      </EvidenceContext.Provider>

      {m.error && (
        <div className="mt-3 flex flex-wrap items-center gap-3 rounded-xl border border-danger/25 bg-danger/[0.06] px-4 py-2.5 text-sm">
          <span className="min-w-0 flex-1 text-fg/90">{m.error}</span>
          {m.retryable && isLast && (
            <button
              onClick={onRetry}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-2.5 py-1 text-sm font-medium hover:border-faint"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Try again
            </button>
          )}
        </div>
      )}

      {done && showAnswer && <ActionBar m={m} text={text} feedback={feedback} sourceCount={cited.length} />}
    </article>
  );
}

export function UserMessage({ content }: { content: string }) {
  return (
    <div className="rise flex justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-3xl rounded-br-lg bg-surface-2 px-4 py-2.5 text-[0.975rem] leading-relaxed">
        {content}
      </div>
    </div>
  );
}
