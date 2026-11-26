import { useEffect, useRef, useState } from "react";
import { ArrowDown } from "lucide-react";
import type { ChatMessage } from "@/hooks/useChat";
import { AssistantMessage, UserMessage } from "./AssistantMessage";

/** The conversation, centered, with stick-to-bottom scrolling while answers stream. */
export function Thread({
  messages,
  feedback,
  onRetry,
  footer,
}: {
  messages: ChatMessage[];
  feedback: boolean;
  onRetry: () => void;
  footer: React.ReactNode;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const lastLen = useRef(messages.length);

  // A new question always scrolls into view; streaming follows only if pinned.
  useEffect(() => {
    const asked = messages.length > lastLen.current;
    lastLen.current = messages.length;
    if (asked) pinned.current = true;
    if (pinned.current) endRef.current?.scrollIntoView({ block: "end", behavior: asked ? "smooth" : "auto" });
  }, [messages]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    pinned.current = atBottom;
    setShowJump(!atBottom);
  };

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div ref={scrollRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-[760px] flex-col gap-8 px-4 pb-8 pt-8 sm:px-6">
          {messages.map((m, i) =>
            m.role === "user" ? (
              <UserMessage key={m.id} content={m.content} />
            ) : (
              <AssistantMessage key={m.id} m={m} feedback={feedback} onRetry={onRetry} isLast={i === messages.length - 1} />
            ),
          )}
          <div ref={endRef} />
        </div>
      </div>

      {showJump && (
        <button
          onClick={() => {
            pinned.current = true;
            endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
          }}
          aria-label="Scroll to the latest message"
          className="fade absolute bottom-[8.5rem] left-1/2 z-10 flex h-9 w-9 -translate-x-1/2 items-center justify-center rounded-full border border-border bg-surface text-muted shadow-md hover:text-fg"
        >
          <ArrowDown className="h-4 w-4" />
        </button>
      )}

      <div className="bg-gradient-to-t from-bg via-bg to-transparent px-4 pb-4 pt-2 sm:px-6">
        <div className="mx-auto w-full max-w-[760px]">{footer}</div>
      </div>
    </div>
  );
}
