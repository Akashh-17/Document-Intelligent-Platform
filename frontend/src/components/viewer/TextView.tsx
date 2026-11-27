import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { locateInText } from "@/lib/locate";

/** A text or Markdown file, shown as written, with the cited passage marked. */
export function TextView({ url, passage }: { url: string; passage?: string }) {
  const [text, setText] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let alive = true;
    setText(null);
    setFailed(false);
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((t) => alive && setText(t))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [url]);

  const range = text && passage ? locateInText(text, passage) : null;

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [text, passage]);

  if (failed) return <p className="px-6 py-24 text-center text-sm text-muted">Couldn't load this file.</p>;
  if (text === null)
    return (
      <p className="flex items-center justify-center gap-2 py-24 text-sm text-muted">
        <Loader2 className="h-4 w-4 animate-spin" /> Opening document…
      </p>
    );

  return (
    <div className="min-h-0 flex-1 overflow-auto">
      <pre className="mx-auto max-w-[68ch] whitespace-pre-wrap break-words px-6 py-6 font-sans text-[13.5px] leading-[1.75] text-fg/85">
        {range ? (
          <>
            {text.slice(0, range[0])}
            <mark ref={markRef} className="doc-highlight">
              {text.slice(range[0], range[1])}
            </mark>
            {text.slice(range[1])}
          </>
        ) : (
          text
        )}
      </pre>
    </div>
  );
}
