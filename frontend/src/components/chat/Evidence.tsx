import { createContext, useContext, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { ArrowUpRight, ExternalLink, FileText, Globe } from "lucide-react";
import { useViewer, type ViewerTarget } from "@/components/viewer/ViewerContext";
import { bestWindow, hostOf, markRuns, splitTitle, type ShownSource } from "@/lib/citations";
import { cn } from "@/lib/utils";

/** Sources of the answer being rendered, plus the words to highlight in them. */
export const EvidenceContext = createContext<{ byN: Map<number, ShownSource>; terms: string[] }>({
  byN: new Map(),
  terms: [],
});

export function Marked({ text, terms }: { text: string; terms: string[] }) {
  return (
    <>
      {markRuns(text, terms).map((r, i) =>
        r.hit ? (
          <mark key={i} className="marker">
            {r.text}
          </mark>
        ) : (
          <span key={i}>{r.text}</span>
        ),
      )}
    </>
  );
}

export function SourceIcon({ source, className }: { source: { kind: string }; className?: string }) {
  const Icon = source.kind === "web" ? Globe : FileText;
  return <Icon className={cn("shrink-0", className)} />;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? ""
    : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function sourceMeta(s: ShownSource | { kind: string; title: string; url?: string | null }) {
  const { name, detail } = splitTitle(s.title);
  if (s.kind === "web") {
    const date = "date" in s && s.date ? formatDate(s.date) : "";
    return { name, detail: [hostOf(s.url) || "web", date].filter(Boolean).join(" · ") };
  }
  return { name, detail: detail ? `Your document · ${detail}` : "Your document" };
}

/** Where a document citation opens in the viewer (null for web sources). */
export function viewerTarget(source: ShownSource): ViewerTarget | null {
  if (source.kind !== "document" || !source.document) return null;
  const summary = source.title.endsWith("(summary)"); // an LLM summary: nothing to highlight
  return {
    document: source.document,
    page: source.page,
    passage: summary ? undefined : source.snippet,
    n: source.n,
  };
}

/** The card that appears over a citation: which source, and the passage itself. */
function EvidenceCard({ source, terms }: { source: ShownSource; terms: string[] }) {
  const { name, detail } = sourceMeta(source);
  const passage = bestWindow(source.snippet, terms, 300);
  const { open } = useViewer();
  const target = viewerTarget(source);
  return (
    <div className="w-[min(360px,calc(100vw-24px))] rounded-xl border border-border bg-surface p-3.5 text-left shadow-xl shadow-black/15">
      <div className="mb-2 flex items-center gap-2">
        <span className="flex h-5 min-w-5 items-center justify-center rounded-md bg-accent-soft px-1 font-mono text-[11px] font-semibold text-accent">
          {source.n}
        </span>
        <SourceIcon source={source} className="h-3.5 w-3.5 text-faint" />
        <span className="min-w-0 truncate text-[13px] font-medium">{name}</span>
      </div>
      <p className="text-[13px] leading-relaxed text-muted">
        <Marked text={passage} terms={terms} />
      </p>
      <div className="mt-2.5 flex items-center justify-between gap-2 font-mono text-[11px] text-faint">
        <span className="truncate">{detail}</span>
        {source.url && (
          <a
            href={source.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex shrink-0 items-center gap-1 text-accent hover:underline"
          >
            Open <ExternalLink className="h-3 w-3" />
          </a>
        )}
        {target && (
          <button
            type="button"
            onClick={() => open(target)}
            className="inline-flex shrink-0 items-center gap-1 text-accent hover:underline"
          >
            View in document <ArrowUpRight className="h-3 w-3" />
          </button>
        )}
      </div>
    </div>
  );
}

/** Hover / focus / tap popover anchored to an inline element, rendered in a portal. */
export function Popover({ anchor, children }: { anchor: (props: AnchorProps) => ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const anchorRef = useRef<HTMLElement | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<number | undefined>(undefined);

  const show = () => {
    window.clearTimeout(closeTimer.current);
    setOpen(true);
  };
  const hide = () => {
    closeTimer.current = window.setTimeout(() => setOpen(false), 120);
  };

  useLayoutEffect(() => {
    if (!open || !anchorRef.current || !cardRef.current) return;
    const a = anchorRef.current.getBoundingClientRect();
    const c = cardRef.current.getBoundingClientRect();
    const below = a.bottom + 8 + c.height < window.innerHeight;
    const top = below ? a.bottom + 8 : Math.max(8, a.top - c.height - 8);
    const left = Math.min(Math.max(12, a.left + a.width / 2 - c.width / 2), window.innerWidth - c.width - 12);
    setPos({ top, left });
  }, [open]);

  return (
    <>
      {anchor({
        ref: (el) => {
          anchorRef.current = el;
        },
        onMouseEnter: show,
        onMouseLeave: hide,
        onFocus: show,
        onBlur: hide,
        onClick: () => (open ? setOpen(false) : show()),
        "aria-expanded": open,
      })}
      {open &&
        createPortal(
          <div
            ref={cardRef}
            onMouseEnter={show}
            onMouseLeave={hide}
            className="fade fixed z-[80]"
            style={{ top: pos?.top ?? -9999, left: pos?.left ?? -9999 }}
          >
            {children}
          </div>,
          document.body,
        )}
    </>
  );
}

interface AnchorProps {
  ref: (el: HTMLElement | null) => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  onFocus: () => void;
  onBlur: () => void;
  onClick: () => void;
  "aria-expanded": boolean;
}

/** Inline [n] citation: a small numbered tab that opens the evidence card. */
export function CitationChip({ n }: { n: number }) {
  const { byN, terms } = useContext(EvidenceContext);
  const source = byN.get(n);
  if (!source) return null;
  return (
    <Popover
      anchor={({ ref, ...handlers }) => (
        <button
          ref={ref}
          type="button"
          {...handlers}
          aria-label={`Source ${n}: ${sourceMeta(source).name}`}
          className="mx-[1px] inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-[1px] items-center justify-center rounded-[5px] bg-accent-soft px-1 align-middle font-mono text-[10.5px] font-semibold leading-none text-accent transition-colors hover:bg-accent hover:text-accent-fg aria-expanded:bg-accent aria-expanded:text-accent-fg"
        >
          {n}
        </button>
      )}
    >
      <EvidenceCard source={source} terms={terms} />
    </Popover>
  );
}
