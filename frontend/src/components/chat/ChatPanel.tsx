import { AlertTriangle, RotateCcw } from "lucide-react";
import type { useAgentStream } from "@/hooks/useAgentStream";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";

type Stream = ReturnType<typeof useAgentStream>;

export function ChatPanel({ stream }: { stream: Stream }) {
  const streaming = stream.status === "streaming";
  return (
    <div className="flex h-full flex-col">
      <MessageList
        messages={stream.messages}
        liveAnswer={stream.answer}
        streaming={streaming}
      />
      {stream.error && (
        <div className="mx-auto flex w-full max-w-3xl items-center gap-2 px-4 pb-2 text-xs text-danger">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          <span className="min-w-0 flex-1">{stream.error}</span>
          {stream.retryable && !streaming && (
            <button
              onClick={stream.retry}
              className="inline-flex shrink-0 items-center gap-1 rounded-md border border-border bg-surface px-2 py-1 text-fg hover:bg-surface-2"
            >
              <RotateCcw className="h-3 w-3" />
              Retry
            </button>
          )}
        </div>
      )}
      <Composer onSend={stream.send} onStop={stream.stop} streaming={streaming} />
    </div>
  );
}
