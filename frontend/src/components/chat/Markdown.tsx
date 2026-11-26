import { Children, memo, useEffect, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy } from "lucide-react";
import { CitationChip } from "./Evidence";

/** Replace [1], [2] … inside text children with citation chips. */
function withCitations(children: ReactNode): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child !== "string") return child;
    const parts: ReactNode[] = [];
    const re = /\[(\d+)\]/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(child)) !== null) {
      if (m.index > last) parts.push(child.slice(last, m.index));
      parts.push(<CitationChip key={`${m.index}-${m[1]}`} n={Number(m[1])} />);
      last = m.index + m[0].length;
    }
    if (last < child.length) parts.push(child.slice(last));
    return parts.length ? parts : child;
  });
}

function CodeBlock({ code, lang, highlight }: { code: string; lang: string; highlight: boolean }) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let active = true;
    if (!highlight) {
      setHtml(null);
      return;
    }
    import("shiki")
      .then(({ codeToHtml }) =>
        codeToHtml(code, {
          lang: lang || "text",
          themes: { light: "github-light", dark: "github-dark" },
          defaultColor: false,
        }),
      )
      .then((out) => active && setHtml(out))
      .catch(() => active && setHtml(null));
    return () => {
      active = false;
    };
  }, [code, lang, highlight]);

  return (
    <div className="group relative">
      <button
        onClick={() =>
          navigator.clipboard.writeText(code).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
          })
        }
        aria-label="Copy code"
        className="absolute right-2 top-2 z-10 rounded-md border border-border bg-surface p-1 text-muted opacity-0 transition-opacity focus:opacity-100 group-hover:opacity-100"
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
      {html ? (
        <div dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        <pre>
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}

export const Markdown = memo(function Markdown({
  text,
  highlight = true,
  streaming = false,
}: {
  text: string;
  highlight?: boolean;
  streaming?: boolean;
}) {
  return (
    <div className={streaming ? "prose-answer caret-last" : "prose-answer"}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p>{withCitations(children)}</p>,
          li: ({ children }) => <li>{withCitations(children)}</li>,
          td: ({ children }) => <td>{withCitations(children)}</td>,
          strong: ({ children }) => <strong>{withCitations(children)}</strong>,
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
          code: ({ className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || "");
            const isBlock = "node" in props && match;
            if (!isBlock) return <code className={className}>{children}</code>;
            return (
              <CodeBlock code={String(children).replace(/\n$/, "")} lang={match![1]} highlight={highlight} />
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
