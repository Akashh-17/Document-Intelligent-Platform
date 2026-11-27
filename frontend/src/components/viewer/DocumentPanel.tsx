import { lazy, Suspense, useEffect, useState } from "react";
import { ExternalLink, Loader2, Quote, Sparkles, X } from "lucide-react";
import { fileUrl, type DocumentInfo } from "@/lib/api";
import { cn } from "@/lib/utils";
import { TextView } from "./TextView";
import type { ViewerTarget } from "./ViewerContext";

// pdf.js is large: load it only when a PDF is opened.
const PdfView = lazy(() => import("./PdfView"));

const KIND_LABEL: Record<DocumentInfo["kind"], string> = { pdf: "PDF", markdown: "Markdown", text: "Text" };

function formatBytes(n?: number | null): string {
  if (!n) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

/** Facts about the file, in the order a reader cares about them. */
function Details({ doc }: { doc: DocumentInfo }) {
  const rows: [string, string][] = [
    ["Type", KIND_LABEL[doc.kind]],
    ["Pages", doc.pages ? String(doc.pages) : doc.kind === "pdf" ? "—" : "1"],
    ["Passages", `${doc.chunks} indexed`],
    ["Size", formatBytes(doc.size_bytes)],
    ["Added", formatDate(doc.added_at)],
  ];
  return (
    <dl className="grid grid-cols-3 gap-x-4 gap-y-3 sm:grid-cols-5">
      {rows.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className="text-[10.5px] font-medium uppercase tracking-[0.08em] text-faint">{k}</dt>
          <dd className="mt-0.5 truncate font-mono text-[12.5px] text-fg/90">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * The document beside the chat. Opened from a citation, it jumps to the cited
 * page with the passage highlighted; opened from the library, it shows the
 * document's details and a couple of things to ask about it.
 */
export function DocumentPanel({
  target,
  doc,
  onClose,
  onAsk,
}: {
  target: ViewerTarget;
  doc?: DocumentInfo;
  onClose: () => void;
  onAsk: (question: string) => void;
}) {
  const url = fileUrl(target.document);
  const kind = doc?.kind ?? (target.document.toLowerCase().endsWith(".pdf") ? "pdf" : "text");
  const cited = Boolean(target.passage);
  const [available, setAvailable] = useState<boolean | null>(doc ? doc.has_file : null);

  useEffect(() => setAvailable(doc ? doc.has_file : null), [doc, target.document]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <section aria-label={`Document: ${target.document}`} className="flex h-full min-w-0 flex-col bg-panel">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border px-4">
        <span className="rounded-[5px] border border-border bg-surface px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider text-muted">
          {kind === "pdf" ? "PDF" : kind === "markdown" ? "MD" : "TXT"}
        </span>
        <h2 className="min-w-0 flex-1 truncate text-sm font-medium" title={target.document}>
          {target.document}
        </h2>
        {available && (
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            aria-label="Open the file in a new tab"
            title="Open in a new tab"
            className="rounded-lg p-1.5 text-muted hover:bg-surface-2 hover:text-fg"
          >
            <ExternalLink className="h-4 w-4" />
          </a>
        )}
        <button
          onClick={onClose}
          aria-label="Close document"
          title="Close (Esc)"
          className="rounded-lg p-1.5 text-muted hover:bg-surface-2 hover:text-fg"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      {cited ? (
        <div className="shrink-0 border-b border-border px-4 py-3">
          <p className="mb-1.5 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.08em] text-faint">
            <Quote className="h-3 w-3" />
            Cited passage
            {target.n !== undefined && (
              <span className="rounded-[4px] bg-accent-soft px-1 font-mono text-[10.5px] normal-case tracking-normal text-accent">
                [{target.n}]
              </span>
            )}
            {target.page ? <span className="font-mono normal-case tracking-normal">· p. {target.page}</span> : null}
          </p>
          <p className="line-clamp-3 border-l-2 border-marker/70 pl-3 text-[13px] leading-relaxed text-muted">
            {target.passage}
          </p>
        </div>
      ) : (
        doc && (
          <div className="shrink-0 space-y-4 border-b border-border px-4 py-4">
            {doc.title && doc.title !== doc.source && (
              <p className="line-clamp-2 text-[15px] font-semibold leading-snug tracking-tight">{doc.title}</p>
            )}
            <Details doc={doc} />
            <div className="flex flex-wrap gap-2">
              {[`Summarize ${doc.source}`, `What are the key facts and numbers in ${doc.source}?`].map((q, i) => (
                <button
                  key={q}
                  onClick={() => onAsk(q)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1.5 text-[12.5px] text-muted transition-colors hover:border-accent/50 hover:text-fg"
                >
                  <Sparkles className="h-3.5 w-3.5 text-accent" />
                  {i === 0 ? "Summarize it" : "Key facts & numbers"}
                </button>
              ))}
            </div>
          </div>
        )
      )}

      {available === false ? (
        <p className="m-auto max-w-xs px-6 text-center text-sm leading-relaxed text-muted">
          No preview for this file: it was indexed before originals were kept. Upload it again to view it here.
        </p>
      ) : kind === "pdf" ? (
        <Suspense
          fallback={
            <p className="flex flex-1 items-center justify-center gap-2 text-sm text-muted">
              <Loader2 className="h-4 w-4 animate-spin" /> Opening document…
            </p>
          }
        >
          <PdfView url={url} page={target.page} passage={target.passage} />
        </Suspense>
      ) : (
        <TextView url={url} passage={target.passage} />
      )}
    </section>
  );
}

/**
 * Wide screens (≥1280px): a column beside the chat, so both stay readable.
 * Narrower: a sheet that slides over the chat from the right, with a backdrop.
 */
export function DocumentPanelFrame({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  if (!open) return null;
  return (
    <>
      <div onClick={onClose} className="fade fixed inset-0 z-40 bg-black/60 xl:hidden" aria-hidden />
      <div
        className={cn(
          "fixed inset-y-0 right-0 z-50 w-full border-l border-border shadow-2xl shadow-black sm:w-[min(680px,92vw)]",
          "xl:static xl:z-auto xl:w-[min(44vw,760px)] xl:shrink-0 xl:shadow-none",
        )}
      >
        <div className="rise h-full">{children}</div>
      </div>
    </>
  );
}
