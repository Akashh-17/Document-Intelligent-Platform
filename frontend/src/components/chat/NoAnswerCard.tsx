import { ArrowRight, FileText, Globe } from "lucide-react";
import type { NoAnswer } from "@/lib/events";
import { cn } from "@/lib/utils";

const STATUS_TONE: Record<NoAnswer["searched"][number]["status"], string> = {
  empty: "text-faint",
  not_searched: "text-faint",
  no_results: "text-muted",
  irrelevant: "text-muted",
  unavailable: "text-warn",
};

/**
 * Shown instead of an answer when no source supports one. Built by the backend
 * without an LLM, so it can't invent anything: it says what was searched, why
 * nothing fit, the closest passages, and what to try next.
 */
export function NoAnswerCard({ data }: { data: NoAnswer }) {
  return (
    <div className="space-y-5">
      <div>
        <p className="text-[0.975rem] font-medium leading-relaxed">{data.headline}</p>
        <p className="mt-1.5 text-[0.95rem] leading-relaxed text-muted">{data.explanation}</p>
      </div>

      <div className="rounded-xl border border-border bg-panel px-4 py-3">
        <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.08em] text-faint">What I searched</p>
        <ul className="space-y-2">
          {data.searched.map((s) => {
            const Icon = s.where === "The web" ? Globe : FileText;
            return (
              <li key={s.where} className="flex gap-2.5 text-sm">
                <Icon className="mt-0.5 h-4 w-4 shrink-0 text-faint" />
                <div className="min-w-0">
                  <span className="font-medium">{s.where}</span>
                  <span className={cn("ml-1.5", STATUS_TONE[s.status])}>{s.summary}</span>
                  {s.queries.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {s.queries.map((q) => (
                        <code key={q} className="rounded-md bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-muted">
                          {q}
                        </code>
                      ))}
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      {data.near_misses.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.08em] text-faint">
            Closest passages — not enough to answer from
          </p>
          <ul className="space-y-1.5">
            {data.near_misses.map((n, i) => (
              <li key={i} className="rounded-lg border border-dashed border-border px-3 py-2">
                <p className="font-mono text-[11px] text-faint">{n.title}</p>
                <p className="mt-0.5 line-clamp-2 text-[13px] text-muted">{n.snippet}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.08em] text-faint">What you can do</p>
        <ul className="space-y-1">
          {data.suggestions.map((s) => (
            <li key={s} className="flex items-start gap-2 text-sm text-fg/90">
              <ArrowRight className="mt-[3px] h-3.5 w-3.5 shrink-0 text-accent" />
              <span>{s}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
