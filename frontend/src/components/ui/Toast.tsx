import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { CircleCheck, CircleAlert, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface Toast {
  id: number;
  tone: "ok" | "warn" | "error";
  title: string;
  detail?: string;
}

const ToastContext = createContext<(t: Omit<Toast, "id">) => void>(() => {});

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => setToasts((ts) => ts.filter((t) => t.id !== id)), []);

  const push = useCallback(
    (t: Omit<Toast, "id">) => {
      const id = Date.now() + Math.random();
      setToasts((ts) => [...ts.slice(-2), { ...t, id }]);
      setTimeout(() => dismiss(id), t.tone === "ok" ? 4000 : 7000);
    },
    [dismiss],
  );

  return (
    <ToastContext.Provider value={push}>
      {children}
      {createPortal(
        <div
          aria-live="polite"
          className="pointer-events-none fixed inset-x-0 bottom-6 z-[70] flex flex-col items-center gap-2 px-4"
        >
          {toasts.map((t) => (
            <div
              key={t.id}
              role="status"
              className="rise pointer-events-auto flex w-full max-w-md items-start gap-3 rounded-xl border border-border bg-surface px-4 py-3 shadow-lg shadow-black/10"
            >
              {t.tone === "ok" ? (
                <CircleCheck className="mt-0.5 h-4 w-4 shrink-0 text-ok" />
              ) : (
                <CircleAlert
                  className={cn("mt-0.5 h-4 w-4 shrink-0", t.tone === "warn" ? "text-warn" : "text-danger")}
                />
              )}
              <div className="min-w-0 flex-1 text-sm">
                <p className="font-medium">{t.title}</p>
                {t.detail && <p className="mt-0.5 text-muted">{t.detail}</p>}
              </div>
              <button
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss"
                className="rounded-md p-0.5 text-faint hover:text-fg"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>,
        document.body,
      )}
    </ToastContext.Provider>
  );
}
