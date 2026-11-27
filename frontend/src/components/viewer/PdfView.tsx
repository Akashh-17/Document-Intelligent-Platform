import { useCallback, useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import type { TextContent } from "pdfjs-dist/types/src/display/api";
import { ChevronLeft, ChevronRight, Loader2, Minus, Plus } from "lucide-react";
import "react-pdf/dist/Page/TextLayer.css";
import { locatePassage } from "@/lib/locate";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

const escapeHtml = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const ZOOMS = [0.75, 1, 1.25, 1.5];

/**
 * One PDF page at a time, fitted to the panel, on a dark desk. When a cited
 * passage is given, its words are wrapped in <mark> in pdf.js's text layer,
 * so the highlight sits exactly over the printed text, and the page scrolls to it.
 */
export default function PdfView({ url, page, passage }: { url: string; page?: number | null; passage?: string }) {
  const startPage = page && page > 0 ? page : 1;
  const [current, setCurrent] = useState(startPage);
  const [numPages, setNumPages] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [marked, setMarked] = useState<Set<number>>(new Set());
  const [width, setWidth] = useState(600);
  const deskRef = useRef<HTMLDivElement>(null);
  const scrolledFor = useRef("");

  useEffect(() => {
    setCurrent(startPage);
    setMarked(new Set());
    scrolledFor.current = "";
  }, [url, startPage, passage]);

  useEffect(() => {
    const el = deskRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width - 48)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const onText = useCallback(
    (content: TextContent) => {
      if (!passage || current !== startPage) return setMarked(new Set());
      const items = content.items.map((it) => ("str" in it ? it.str : ""));
      setMarked(locatePassage(items, passage));
    },
    [passage, current, startPage],
  );

  const renderText = useCallback(
    ({ str, itemIndex }: { str: string; itemIndex: number }) =>
      marked.has(itemIndex) ? `<mark>${escapeHtml(str)}</mark>` : escapeHtml(str),
    [marked],
  );

  const onTextLayer = useCallback(() => {
    const key = `${url}|${current}|${marked.size}`;
    if (!marked.size || scrolledFor.current === key) return;
    scrolledFor.current = key;
    deskRef.current?.querySelector(".textLayer mark")?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [url, current, marked]);

  const go = (p: number) => setCurrent(Math.min(Math.max(1, p), numPages || 1));

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={deskRef} className="pdf-desk min-h-0 flex-1 overflow-auto px-6 py-6">
        <Document
          file={url}
          onLoadSuccess={({ numPages: n }) => setNumPages(n)}
          loading={<DeskMessage spinning>Opening document…</DeskMessage>}
          error={<DeskMessage>Couldn't open this PDF.</DeskMessage>}
          className="flex justify-center"
        >
          <Page
            key={current}
            pageNumber={current}
            width={width * zoom}
            renderAnnotationLayer={false}
            onGetTextSuccess={onText}
            customTextRenderer={renderText}
            onRenderTextLayerSuccess={onTextLayer}
            loading={<div style={{ width: width * zoom, aspectRatio: "1 / 1.29" }} className="rounded-sm bg-white/[0.04]" />}
            className="pdf-page"
          />
        </Document>
      </div>

      <div className="flex h-11 shrink-0 items-center justify-between border-t border-border px-3 font-mono text-[11.5px] text-muted">
        <div className="flex items-center gap-1">
          <IconButton label="Previous page" disabled={current <= 1} onClick={() => go(current - 1)}>
            <ChevronLeft className="h-4 w-4" />
          </IconButton>
          <span className="min-w-[4.5rem] text-center tabular-nums">
            {current} <span className="text-faint">/ {numPages || "–"}</span>
          </span>
          <IconButton label="Next page" disabled={!numPages || current >= numPages} onClick={() => go(current + 1)}>
            <ChevronRight className="h-4 w-4" />
          </IconButton>
          {passage && current !== startPage && (
            <button onClick={() => go(startPage)} className="ml-2 rounded-md px-2 py-1 text-accent hover:bg-surface-2">
              Back to p. {startPage}
            </button>
          )}
        </div>
        <div className="flex items-center gap-1">
          <IconButton label="Zoom out" disabled={zoom === ZOOMS[0]} onClick={() => setZoom(ZOOMS[ZOOMS.indexOf(zoom) - 1])}>
            <Minus className="h-3.5 w-3.5" />
          </IconButton>
          <span className="w-10 text-center tabular-nums">{Math.round(zoom * 100)}%</span>
          <IconButton label="Zoom in" disabled={zoom === ZOOMS.at(-1)} onClick={() => setZoom(ZOOMS[ZOOMS.indexOf(zoom) + 1])}>
            <Plus className="h-3.5 w-3.5" />
          </IconButton>
        </div>
      </div>
    </div>
  );
}

function IconButton({
  label,
  disabled,
  onClick,
  children,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="flex h-7 w-7 items-center justify-center rounded-md text-muted hover:bg-surface-2 hover:text-fg disabled:pointer-events-none disabled:opacity-30"
    >
      {children}
    </button>
  );
}

function DeskMessage({ children, spinning }: { children: React.ReactNode; spinning?: boolean }) {
  return (
    <p className="flex items-center gap-2 py-24 text-sm text-muted">
      {spinning && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </p>
  );
}
