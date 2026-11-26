import { FileText, Globe, ExternalLink } from "lucide-react";
import type { Source } from "@/lib/events";

/** Small pills explaining *why* hybrid retrieval picked this chunk. */
function RetrievalSignals({ source }: { source: Source }) {
  const r = source.retrieval;
  if (!r) return null;
  const pills: { label: string; title: string }[] = [];
  if (r.dense_rank != null)
    pills.push({ label: `semantic #${r.dense_rank}`, title: "Rank in vector (embedding) search" });
  if (r.bm25_rank != null)
    pills.push({ label: `keyword #${r.bm25_rank}`, title: "Rank in BM25 keyword search" });
  if (r.rerank_score != null)
    pills.push({
      label: `rerank ${Math.round(r.rerank_score * 100)}%`,
      title: "Cross-encoder relevance after fusion",
    });
  if (!pills.length) return null;
  return (
    <div className="mt-1.5 flex flex-wrap gap-1">
      {pills.map((p) => (
        <span
          key={p.label}
          title={p.title}
          className="rounded bg-surface px-1.5 py-0.5 text-[10px] font-medium text-muted ring-1 ring-border"
        >
          {p.label}
        </span>
      ))}
    </div>
  );
}

export function SourceCard({ source, index }: { source: Source; index: number }) {
  const Icon = source.kind === "web" ? Globe : FileText;
  return (
    <article
      id={`source-idx-${index}`}
      className="scroll-mt-4 rounded-lg border border-border bg-surface-2 p-3 transition-shadow target:ring-2 target:ring-accent"
    >
      <div className="flex items-start gap-2">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent text-[11px] font-semibold text-accent-fg">
          {index}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <Icon className="h-3.5 w-3.5 shrink-0 text-muted" />
            {source.url ? (
              <a
                href={source.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 truncate text-sm font-medium text-accent hover:underline"
              >
                <span className="truncate">{source.title}</span>
                <ExternalLink className="h-3 w-3 shrink-0" />
              </a>
            ) : (
              <span className="truncate text-sm font-medium">{source.title}</span>
            )}
          </div>
          <p className="mt-1 line-clamp-3 text-xs text-muted">{source.snippet}</p>
          {source.guardrail && (
            <span
              title="The retrieval guardrail removed text that tried to instruct the AI"
              className="mt-1.5 inline-block rounded bg-warn/15 px-1.5 py-0.5 text-[10px] font-medium text-warn"
            >
              ⚠ {source.guardrail}
            </span>
          )}
          <RetrievalSignals source={source} />
        </div>
      </div>
    </article>
  );
}
