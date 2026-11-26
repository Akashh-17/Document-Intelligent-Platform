import { SearchX, FileText, Globe, Lightbulb, CircleSlash, CircleAlert, CircleCheck } from "lucide-react";
import type { NoAnswer } from "@/lib/events";
import { cn } from "@/lib/utils";

const STATUS_STYLE: Record<NoAnswer["searched"][number]["status"], { icon: typeof CircleCheck; className: string }> = {
  empty: { icon: CircleSlash, className: "text-muted" },
  not_searched: { icon: CircleSlash, className: "text-muted" },
  no_results: { icon: CircleCheck, className: "text-muted" },
  irrelevant: { icon: CircleCheck, className: "text-muted" },
  unavailable: { icon: CircleAlert, className: "text-warn" },
};

/**
 * Shown instead of an answer when neither the documents nor the web had
 * reliable information. Honest, specific, and actionable — never a guess.
 */
export function NoAnswerCard({ data }: { data: NoAnswer }) {
  return (
    <div className="flex flex-col gap-3 text-sm">
      <div className="flex items-start gap-2">
        <SearchX className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
        <div>
          <p className="font-semibold">{data.headline}</p>
          <p className="mt-1 text-muted">{data.explanation}</p>
        </div>
      </div>

      <section>
        <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">What I searched</h4>
        <ul className="flex flex-col gap-1.5">
          {data.searched.map((s) => {
            const { icon: StatusIcon, className } = STATUS_STYLE[s.status];
            const WhereIcon = s.where === "The web" ? Globe : FileText;
            return (
              <li key={s.where} className="flex items-start gap-2">
                <WhereIcon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" />
                <div className="min-w-0">
                  <span className="font-medium">{s.where}: </span>
                  <span className={cn("inline-flex items-center gap-1", className)}>
                    <StatusIcon className="h-3 w-3" />
                    {s.summary}
                  </span>
                  {s.queries.length > 0 && (
                    <div className="mt-0.5 flex flex-wrap gap-1">
                      {s.queries.map((q) => (
                        <code key={q} className="rounded bg-surface-2 px-1.5 py-0.5 text-[11px] text-muted">
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
      </section>

      {data.near_misses.length > 0 && (
        <section>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">
            Closest passages — not relevant enough to answer from
          </h4>
          <ul className="flex flex-col gap-1.5">
            {data.near_misses.map((n, i) => (
              <li key={i} className="rounded-lg border border-border bg-surface-2 px-3 py-2">
                <p className="text-xs font-medium">{n.title}</p>
                <p className="mt-0.5 line-clamp-2 text-xs text-muted">{n.snippet}</p>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">What you can do</h4>
        <ul className="flex flex-col gap-1">
          {data.suggestions.map((s) => (
            <li key={s} className="flex items-start gap-2 text-xs">
              <Lightbulb className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" />
              <span>{s}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
