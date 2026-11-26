import { useCallback, useEffect, useRef, useState } from "react";
import { Menu, SquarePen } from "lucide-react";
import { useChat } from "@/hooks/useChat";
import { useLibrary } from "@/hooks/useLibrary";
import { Composer, type ComposerHandle } from "@/components/chat/Composer";
import { EmptyState } from "@/components/chat/EmptyState";
import { Thread } from "@/components/chat/Thread";
import { DropOverlay, UploadDialog, useWindowDrop } from "@/components/upload/UploadDialog";
import { useUploader } from "@/components/upload/useUploader";
import { cn } from "@/lib/utils";
import { LogoMark } from "./Logo";
import { Sidebar } from "./Sidebar";

export function AppShell() {
  const chat = useChat();
  const library = useLibrary();
  const { busy, upload } = useUploader(library.refresh);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const composerRef = useRef<ComposerHandle>(null);

  const dropFiles = useCallback((files: File[]) => void upload(files), [upload]);
  const dragging = useWindowDrop(dropFiles, !uploadOpen);

  // Ctrl/Cmd+K: new chat.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        chat.newChat();
        composerRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chat]);

  const closeDrawer = () => setDrawer(false);
  const empty = chat.messages.length === 0;
  const title = chat.chats.find((c) => c.id === chat.chatId)?.title;

  const composer = (
    <Composer
      ref={composerRef}
      onSend={chat.send}
      onStop={chat.stop}
      onAttach={() => setUploadOpen(true)}
      streaming={chat.streaming}
      disabled={library.offline}
      autoFocus
    />
  );

  const sidebar = (
    <Sidebar
      chats={chat.chats}
      activeChatId={chat.chatId}
      library={library}
      onNewChat={() => {
        chat.newChat();
        closeDrawer();
      }}
      onOpenChat={(id) => {
        chat.openChat(id);
        closeDrawer();
      }}
      onDeleteChat={chat.removeChat}
      onUpload={() => {
        setUploadOpen(true);
        closeDrawer();
      }}
      onClose={drawer ? closeDrawer : undefined}
    />
  );

  return (
    <div className="flex h-full">
      {/* desktop sidebar */}
      <div className="hidden md:block">{sidebar}</div>

      {/* mobile drawer */}
      <div className={cn("fixed inset-0 z-40 md:hidden", !drawer && "pointer-events-none")} aria-hidden={!drawer}>
        <div
          onClick={closeDrawer}
          className={cn("absolute inset-0 bg-black/40 transition-opacity", drawer ? "opacity-100" : "opacity-0")}
        />
        <div
          className={cn(
            "absolute inset-y-0 left-0 transition-transform duration-300 ease-out",
            drawer ? "translate-x-0" : "-translate-x-full",
          )}
        >
          {sidebar}
        </div>
      </div>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-2 px-3 md:px-5">
          <button
            onClick={() => setDrawer(true)}
            aria-label="Open menu"
            className="rounded-lg p-2 text-muted hover:bg-surface-2 hover:text-fg md:hidden"
          >
            <Menu className="h-5 w-5" />
          </button>
          <LogoMark className="h-5 w-5 md:hidden" />
          <p className="min-w-0 flex-1 truncate text-sm text-muted">{!empty && title ? title : ""}</p>
          {!empty && (
            <button
              onClick={() => {
                chat.newChat();
                composerRef.current?.focus();
              }}
              aria-label="New chat"
              title="New chat (Ctrl+K)"
              className="rounded-lg p-2 text-muted hover:bg-surface-2 hover:text-fg"
            >
              <SquarePen className="h-[18px] w-[18px]" />
            </button>
          )}
        </header>

        {empty ? (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <EmptyState
              composer={composer}
              onPick={chat.send}
              onUpload={() => setUploadOpen(true)}
              docs={library.docs}
            />
          </div>
        ) : (
          <Thread
            messages={chat.messages}
            feedback={Boolean(library.health?.langsmith_tracing)}
            onRetry={chat.retry}
            footer={
              <>
                {composer}
                <p className="mt-2 text-center text-[11.5px] text-faint">
                  Answers cite your documents and the web. Check the sources for anything important.
                </p>
              </>
            }
          />
        )}
      </main>

      <UploadDialog open={uploadOpen} busy={busy} onClose={() => setUploadOpen(false)} onUpload={upload} />
      <DropOverlay active={dragging} />
    </div>
  );
}
