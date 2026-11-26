import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { FileUp, Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { isAccepted } from "./useUploader";

/**
 * Rendered into <body> through a portal so no ancestor (e.g. a header with
 * backdrop-filter, which creates a containing block for fixed elements) can
 * clip or offset it.
 */
export function UploadDialog({
  open,
  busy,
  onClose,
  onUpload,
}: {
  open: boolean;
  busy: boolean;
  onClose: () => void;
  onUpload: (files: File[]) => Promise<boolean>;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !busy && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, busy, onClose]);

  if (!open) return null;

  const handle = async (list: FileList | null) => {
    const files = Array.from(list ?? []);
    if (!files.length) return;
    if (await onUpload(files)) onClose(); // success: close; failures stay visible as toasts
  };

  return createPortal(
    <div
      className="fade fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
      onMouseDown={() => !busy && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-title"
        className="rise w-full max-w-lg rounded-2xl border border-border bg-surface p-6 shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="mb-1 flex items-start justify-between gap-4">
          <h2 id="upload-title" className="font-display text-lg font-semibold tracking-tight">
            Add documents
          </h2>
          <button
            onClick={onClose}
            disabled={busy}
            aria-label="Close"
            className="rounded-lg p-1 text-muted hover:bg-surface-2 hover:text-fg disabled:opacity-40"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="mb-5 text-sm text-muted">
          They become searchable right away. Uploading a file with the same name replaces it.
        </p>

        <button
          type="button"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation(); // don't let the window-level drop target upload it again
            setDragging(false);
            handle(e.dataTransfer.files);
          }}
          className={cn(
            "flex w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors",
            dragging ? "border-accent bg-accent-soft" : "border-border hover:border-faint hover:bg-surface-2",
          )}
        >
          {busy ? (
            <>
              <Loader2 className="h-7 w-7 animate-spin text-accent" />
              <span className="text-sm text-muted">Reading and indexing…</span>
            </>
          ) : (
            <>
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-accent-soft text-accent">
                <FileUp className="h-5 w-5" />
              </span>
              <span className="text-sm">
                <span className="font-medium text-fg">Choose files</span>
                <span className="text-muted"> or drag them here</span>
              </span>
              <span className="font-mono text-xs text-faint">PDF · TXT · Markdown — up to 20 MB each</span>
            </>
          )}
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md,.markdown"
          className="hidden"
          onChange={(e) => {
            handle(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
    </div>,
    document.body,
  );
}

/** Full-window drop target: drag files anywhere onto the app to add them. */
export function useWindowDrop(onFiles: (files: File[]) => void, enabled = true) {
  const [active, setActive] = useState(false);
  const depth = useRef(0);

  useEffect(() => {
    if (!enabled) {
      depth.current = 0;
      setActive(false);
      return;
    }
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current += 1;
      setActive(true);
    };
    const leave = () => {
      depth.current = Math.max(0, depth.current - 1);
      if (!depth.current) setActive(false);
    };
    const over = (e: DragEvent) => hasFiles(e) && e.preventDefault();
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current = 0;
      setActive(false);
      const files = Array.from(e.dataTransfer?.files ?? []).filter(isAccepted);
      if (files.length) onFiles(files);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, [onFiles, enabled]);

  return active;
}

export function DropOverlay({ active }: { active: boolean }) {
  if (!active) return null;
  return createPortal(
    <div className="fade pointer-events-none fixed inset-0 z-[65] flex items-center justify-center bg-bg/80 p-6 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-accent bg-surface px-12 py-10 text-center">
        <FileUp className="h-8 w-8 text-accent" />
        <p className="font-display text-lg font-semibold">Drop to add to your library</p>
        <p className="font-mono text-xs text-faint">PDF · TXT · Markdown</p>
      </div>
    </div>,
    document.body,
  );
}
