import { useState } from "react";
import { ChevronRight, CircleAlert, CornerDownRight, ShieldCheck } from "lucide-react";
import type { ChatMessage } from "@/hooks/useChat";
import { AGENT_NAMES, agentLabel, liveLabel, toolLabel, type AgentStep } from "@/lib/trace";
import { cn } from "@/lib/utils";

function seconds(ms?: number) {
  if (!ms) return "";
  const s = ms / 1000;
  return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
}

function summary(m: ChatMessage): string {
  const n = m.sources?.length ?? 0;
  const t = seconds(m.durationMs);
  if (m.blocked === "prompt_injection") return "Stopped by safety checks";
  if (m.blocked === "policy") return "Declined";
  if (m.plan?.route === "direct") return t ? `Answered directly · ${t}` : "Answered directly";
  if (m.noAnswer) return `Searched · nothing reliable found${t ? ` · ${t}` : ""}`;
  return [`Researched ${n} source${n === 1 ? "" : "s"}`, t].filter(Boolean).join(" · ");
}

function StepRow({ step, plan, last }: { step: AgentStep; plan: ChatMessage["plan"]; last: boolean }) {
  const failed = step.status === "error";
  return (
    <li className="relative pb-3 pl-6 last:pb-0">
      {!last && <span className="absolute bottom-0 left-[5px] top-4 w-px bg-border" aria-hidden />}
      <span
        aria-hidden
        className={cn(
          "absolute left-0 top-[7px] h-[11px] w-[11px] rounded-full border-2",
          step.status === "running"
            ? "animate-pulse border-accent bg-accent-soft"
            : failed
              ? "border-danger bg-danger/20"
              : "border-faint bg-panel",
        )}
      />
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className={cn("text-sm", failed ? "text-danger" : "text-fg")}>{agentLabel(step, plan)}</span>
        <span className="font-mono text-[10.5px] uppercase tracking-wide text-faint">{AGENT_NAMES[step.agent]}</span>
      </div>
      {step.tools.length > 0 && (
        <ul className="mt-1 space-y-0.5">
          {step.tools.map((t) =>
            t.tool === "note" && t.note ? (
              <li key={t.id} className="flex items-start gap-1.5 text-[13px] text-warn">
                {t.note.kind === "masked" || t.note.kind === "neutralized" || t.note.kind === "redacted" ? (
                  <ShieldCheck className="mt-[3px] h-3.5 w-3.5 shrink-0" />
                ) : (
                  <CircleAlert className="mt-[3px] h-3.5 w-3.5 shrink-0" />
                )}
                <span>{t.note.text}</span>
              </li>
            ) : (
              <li key={t.id} className="flex items-start gap-1.5 text-[13px] text-muted">
                <CornerDownRight className="mt-[3px] h-3.5 w-3.5 shrink-0 text-faint" />
                <span className={cn("min-w-0", t.status === "error" && "text-danger")}>
                  {toolLabel(t)}
                  {t.status === "error" && " — failed, continued without it"}
                </span>
              </li>
            ),
          )}
        </ul>
      )}
    </li>
  );
}

/**
 * One line above each answer, like a "thinking" indicator: live while the
 * agents work, then a quiet summary that expands into the step-by-step trace.
 */
export function ResearchTrace({ message }: { message: ChatMessage }) {
  const [open, setOpen] = useState(false);
  const agents = message.agents ?? [];
  const live = message.pending;

  if (live) {
    return (
      <div className="mb-3 flex items-center gap-2.5" role="status" aria-live="polite">
        <span className="relative flex h-2.5 w-2.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent/60" />
          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-accent" />
        </span>
        <span className="shimmer-text text-sm">{liveLabel(agents, message.plan)}</span>
      </div>
    );
  }

  if (!agents.length) return null;

  return (
    <div className="mb-3">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="group -ml-1 inline-flex items-center gap-1 rounded-md px-1 py-0.5 text-sm text-muted transition-colors hover:text-fg"
      >
        <span>{summary(message)}</span>
        <ChevronRight className={cn("h-4 w-4 transition-transform", open && "rotate-90")} />
      </button>
      {open && (
        <ol className="fade mt-2.5 rounded-xl border border-border bg-panel px-4 py-3.5">
          {agents.map((step, i) => (
            <StepRow key={step.agent} step={step} plan={message.plan} last={i === agents.length - 1} />
          ))}
        </ol>
      )}
    </div>
  );
}
