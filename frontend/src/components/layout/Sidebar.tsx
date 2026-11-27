import { useState } from "react";
import { Loader2, MessageSquare, Plus, Trash2, Upload, X } from "lucide-react";
import type { DocumentInfo } from "@/lib/api";
import type { ChatMeta } from "@/lib/chats";
import type { Library } from "@/hooks/useLibrary";
import { cn } from "@/lib/utils";
import { Wordmark } from "./Logo";

const REPO_URL = "https://github.com/Akashh-17/Document-Intelligent-Platform";

function SectionLabel({ children, action }: { children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-3 pb-1.5 pt-5">
      <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-faint">{children}</span>
      {action}
    </div>
  );
}

function GithubMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className} fill="currentColor">
      <path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.71 1.26 3.37.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.58.23 2.75.11 3.04.74.81 1.19 1.83 1.19 3.09 0 4.41-2.7 5.38-5.27 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z" />
    </svg>
  );
}

const KIND_TAG = { pdf: "PDF", markdown: "MD", text: "TXT" } as const;

function DocumentRow({
  doc,
  active,
  onOpen,
  onRemove,
}: {
  doc: DocumentInfo;
  active: boolean;
  onOpen: () => void;
  onRemove: () => Promise<void>;
}) {
  const [confirming, setConfirming] = useState(false);
  const [removing, setRemoving] = useState(false);
  return (
    <li className="group relative">
      <button
        onClick={onOpen}
        aria-current={active || undefined}
        title={doc.title && doc.title !== doc.source ? `${doc.source} — ${doc.title}` : doc.source}
        className={cn(
          "flex w-full items-center gap-2.5 rounded-lg px-3 py-1.5 pr-8 text-left text-sm transition-colors",
          active ? "bg-surface-2 text-fg" : "text-muted hover:bg-surface-2 hover:text-fg",
        )}
      >
        <span
          className={cn(
            "w-[30px] shrink-0 rounded-[4px] border py-px text-center font-mono text-[9px] font-medium tracking-wider",
            active ? "border-accent/40 text-accent" : "border-border text-faint",
          )}
        >
          {KIND_TAG[doc.kind] ?? "TXT"}
        </span>
        <span className="min-w-0 flex-1 truncate text-[13px]">{doc.source}</span>
      </button>
      {!confirming && (
        <button
          onClick={() => setConfirming(true)}
          aria-label={`Remove ${doc.source}`}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-faint opacity-0 hover:text-danger focus:opacity-100 group-hover:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      )}
      {confirming && (
        <div className="fade mx-2 mb-1 flex items-center justify-between gap-2 rounded-lg bg-surface-2 px-3 py-1.5 text-xs">
          <span className="text-muted">Remove from library?</span>
          <span className="flex gap-1">
            <button
              onClick={() => setConfirming(false)}
              className="rounded-md px-2 py-0.5 text-muted hover:text-fg"
            >
              Keep
            </button>
            <button
              disabled={removing}
              onClick={async () => {
                setRemoving(true);
                try {
                  await onRemove();
                } finally {
                  setRemoving(false);
                  setConfirming(false);
                }
              }}
              className="inline-flex items-center gap-1 rounded-md bg-danger/10 px-2 py-0.5 font-medium text-danger hover:bg-danger/20"
            >
              {removing && <Loader2 className="h-3 w-3 animate-spin" />}
              Remove
            </button>
          </span>
        </div>
      )}
    </li>
  );
}

export function Sidebar({
  chats,
  activeChatId,
  library,
  onNewChat,
  onOpenChat,
  onDeleteChat,
  onUpload,
  onClose,
  activeDocument,
  onOpenDocument,
}: {
  chats: ChatMeta[];
  activeChatId: string;
  library: Library;
  onNewChat: () => void;
  onOpenChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
  onUpload: () => void;
  onClose?: () => void;
  activeDocument?: string;
  onOpenDocument: (name: string) => void;
}) {
  const { docs, health, offline } = library;
  const modelReady = health?.llm_configured;

  return (
    <aside className="flex h-full w-[272px] flex-col border-r border-border bg-panel">
      <div className="flex items-center justify-between px-4 pb-3 pt-4">
        <Wordmark />
        {onClose && (
          <button onClick={onClose} aria-label="Close menu" className="rounded-lg p-1.5 text-muted hover:bg-surface-2 md:hidden">
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="px-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center gap-2 rounded-xl border border-border bg-surface px-3 py-2 text-sm font-medium shadow-sm transition-colors hover:border-faint"
        >
          <Plus className="h-4 w-4 text-accent" />
          New chat
        </button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        <SectionLabel>Chats</SectionLabel>
        {chats.length === 0 ? (
          <p className="px-3 py-1 text-xs text-faint">Your conversations will appear here.</p>
        ) : (
          <ul>
            {chats.map((c) => (
              <li key={c.id} className="group relative">
                <button
                  onClick={() => onOpenChat(c.id)}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-lg px-3 py-1.5 pr-8 text-left text-sm transition-colors",
                    c.id === activeChatId ? "bg-surface-2 text-fg" : "text-muted hover:bg-surface-2 hover:text-fg",
                  )}
                >
                  <MessageSquare className="h-4 w-4 shrink-0 text-faint" />
                  <span className="truncate">{c.title}</span>
                </button>
                <button
                  onClick={() => onDeleteChat(c.id)}
                  aria-label={`Delete chat: ${c.title}`}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-faint opacity-0 hover:text-danger focus:opacity-100 group-hover:opacity-100"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}

        <SectionLabel
          action={
            <button
              onClick={onUpload}
              aria-label="Add documents"
              title="Add documents"
              className="rounded-md p-1 text-faint hover:bg-surface-2 hover:text-accent"
            >
              <Upload className="h-3.5 w-3.5" />
            </button>
          }
        >
          Library{docs ? ` · ${docs.length}` : ""}
        </SectionLabel>
        {docs === null ? (
          <p className="px-3 py-1 text-xs text-faint">{offline ? "Waiting for the server…" : "Loading…"}</p>
        ) : docs.length === 0 ? (
          <button
            onClick={onUpload}
            className="mx-1 w-[calc(100%-0.5rem)] rounded-lg border border-dashed border-border px-3 py-3 text-left text-xs text-muted hover:border-faint"
          >
            No documents yet. <span className="text-accent">Add your first file</span>
          </button>
        ) : (
          <ul>
            {docs.map((d) => (
              <DocumentRow
                key={d.source}
                doc={d}
                active={d.source === activeDocument}
                onOpen={() => onOpenDocument(d.source)}
                onRemove={() => library.remove(d.source)}
              />
            ))}
          </ul>
        )}
      </nav>

      <div className="flex items-center gap-2 border-t border-border px-3 py-3">
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 text-xs">
            <span
              className={cn(
                "h-1.5 w-1.5 shrink-0 rounded-full",
                offline ? "bg-danger" : modelReady ? "bg-ok" : "bg-warn",
              )}
            />
            <span className="truncate text-muted">
              {offline
                ? "Server offline"
                : !health
                  ? "Connecting…"
                  : modelReady
                    ? `Groq · ${health.model_heavy.split("/").pop()}`
                    : "Add GROQ_API_KEY to .env"}
            </span>
          </p>
        </div>
        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          aria-label="Source code on GitHub"
          title="Source code on GitHub"
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted hover:bg-surface-2 hover:text-fg"
        >
          <GithubMark className="h-4 w-4" />
        </a>
      </div>
    </aside>
  );
}
