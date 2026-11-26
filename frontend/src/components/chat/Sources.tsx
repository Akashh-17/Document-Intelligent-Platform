import { useState } from "react";
import { ChevronDown, ExternalLink } from "lucide-react";
import type { Source } from "@/lib/events";
import { bestWindow, type ShownSource } from "@/lib/citations";
import { cn } from "@/lib/utils";
import { Marked, SourceIcon, sourceMeta } from "./Evidence";

/**
 * The sources an answer actually cites, numbered in reading order. Selecting a
 * card opens its passage underneath; sources that were consulted but not cited
 * are tucked behind "also searched".
 */
export function Sources({ cited, others, terms }: { cited: ShownSource[]; others: Source[]; terms: string[] }) {
  const [openN, setOpenN] = useState<number | null>(null);
  const [showOthers, setShowOthers] = useState(false);
  if (!cited.length && !others.length) return null;
  const opened = cited.find((s) => s.n === openN);

  return (
    <div className="mt-5">
      {cited.length > 0 && (
        <>
          <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.08em] text-faint">Sources</p>
          <div className="flex flex-wrap gap-2">
            {cited.map((s) => {
              const { name, detail } = sourceMeta(s);
              const active = s.n === openN;
              return (
                <button
                  key={s.index}
                  onClick={() => setOpenN(active ? null : s.n)}
                  aria-expanded={active}
                  className={cn(
                    "flex max-w-[15rem] items-center gap-2 rounded-xl border px-2.5 py-1.5 text-left transition-colors",
                    active ? "border-accent bg-accent-soft" : "border-border bg-surface hover:border-faint",
                  )}
                >
                  <span className="flex h-5 min-w-5 items-center justify-center rounded-md bg-accent-soft px-1 font-mono text-[11px] font-semibold text-accent">
                    {s.n}
                  </span>
                  <SourceIcon source={s} className="h-3.5 w-3.5 text-faint" />
                  <span className="min-w-0">
                    <span className="block truncate text-[12.5px] font-medium leading-tight">{name}</span>
                    <span className="block truncate font-mono text-[10.5px] leading-tight text-faint">{detail}</span>
                  </span>
                </button>
              );
            })}
          </div>
        </>
      )}

      {opened && (
        <div className="fade mt-2.5 rounded-xl border border-border bg-panel px-4 py-3">
          <p className="text-[13.5px] leading-relaxed text-muted">
            <Marked text={bestWindow(opened.snippet, terms, 520)} terms={terms} />
          </p>
          {opened.url && (
            <a
              href={opened.url}
              target="_blank"
              rel="noreferrer"
              className="mt-2 inline-flex items-center gap-1 font-mono text-[11px] text-accent hover:underline"
            >
              Open source <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
      )}

      {others.length > 0 && (
        <div className="mt-3">
          <button
            onClick={() => setShowOthers((v) => !v)}
            aria-expanded={showOthers}
            className="inline-flex items-center gap-1 text-xs text-faint transition-colors hover:text-muted"
          >
            {cited.length ? "Also searched" : "Searched"} {others.length} other source{others.length === 1 ? "" : "s"}
            <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", showOthers && "rotate-180")} />
          </button>
          {showOthers && (
            <ul className="fade mt-2 space-y-1">
              {others.map((s) => {
                const { name, detail } = sourceMeta(s);
                return (
                  <li key={s.index} className="flex items-center gap-2 text-xs text-muted">
                    <SourceIcon source={s} className="h-3.5 w-3.5 text-faint" />
                    {s.url ? (
                      <a href={s.url} target="_blank" rel="noreferrer" className="truncate hover:text-fg hover:underline">
                        {name}
                      </a>
                    ) : (
                      <span className="truncate">{name}</span>
                    )}
                    <span className="shrink-0 font-mono text-[10.5px] text-faint">{detail}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
