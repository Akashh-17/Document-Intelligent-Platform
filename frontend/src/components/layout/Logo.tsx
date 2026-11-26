import { cn } from "@/lib/utils";

/** Brand mark: a page with a highlighter stroke across one line — evidence, marked. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={cn("h-6 w-6", className)}>
      <rect x="4" y="2.5" width="14" height="19" rx="3" className="fill-surface-2 stroke-border" strokeWidth="1.2" />
      <rect x="7" y="7" width="8" height="1.6" rx="0.8" className="fill-faint" />
      <rect x="6" y="10.6" width="13" height="3.4" rx="1.2" className="fill-marker" opacity="0.9" />
      <rect x="7" y="11.5" width="7" height="1.6" rx="0.8" className="fill-marker-fg" opacity="0.55" />
      <rect x="7" y="16" width="6" height="1.6" rx="0.8" className="fill-faint" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className="flex items-center gap-2">
      <LogoMark />
      <span className="font-display text-[1.05rem] font-semibold tracking-tight">Docintel</span>
    </span>
  );
}
