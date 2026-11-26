import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { ArrowUp, Paperclip, Square } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ComposerHandle {
  focus: () => void;
}

export const Composer = forwardRef<
  ComposerHandle,
  {
    onSend: (text: string) => void;
    onStop: () => void;
    onAttach: () => void;
    streaming: boolean;
    disabled?: boolean;
    autoFocus?: boolean;
  }
>(function Composer({ onSend, onStop, onAttach, streaming, disabled, autoFocus }, ref) {
  const [value, setValue] = useState("");
  const textRef = useRef<HTMLTextAreaElement>(null);

  useImperativeHandle(ref, () => ({ focus: () => textRef.current?.focus() }));

  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
    el.style.overflowY = el.scrollHeight > 220 ? "auto" : "hidden"; // no stray scrollbar
  }, [value]);

  useEffect(() => {
    if (autoFocus) textRef.current?.focus();
  }, [autoFocus]);

  const submit = () => {
    const text = value.trim();
    if (!text || streaming || disabled) return;
    onSend(text);
    setValue("");
  };

  return (
    <div
      className={cn(
        "rounded-[26px] border border-border bg-surface shadow-[0_1px_2px_rgba(0,0,0,0.04),0_8px_24px_-12px_rgba(0,0,0,0.18)] transition-shadow",
        "focus-within:border-faint focus-within:shadow-[0_1px_2px_rgba(0,0,0,0.04),0_12px_32px_-12px_rgba(0,0,0,0.28)]",
      )}
    >
      <label htmlFor="composer" className="sr-only">
        Ask a question
      </label>
      <textarea
        id="composer"
        ref={textRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
        rows={1}
        maxLength={4000}
        placeholder="Ask about your documents, or anything on the web"
        className="block max-h-[220px] w-full resize-none bg-transparent px-5 pb-1 pt-4 text-[0.975rem] leading-relaxed outline-none placeholder:text-faint focus-visible:outline-none"
      />
      <div className="flex items-center justify-between px-3 pb-3 pt-1">
        <button
          type="button"
          onClick={onAttach}
          className="flex h-9 items-center gap-1.5 rounded-full px-3 text-sm text-muted transition-colors hover:bg-surface-2 hover:text-fg"
        >
          <Paperclip className="h-4 w-4" />
          Add files
        </button>
        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop generating"
            title="Stop"
            className="flex h-9 w-9 items-center justify-center rounded-full bg-fg text-bg transition-transform hover:scale-105"
          >
            <Square className="h-3.5 w-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!value.trim() || disabled}
            aria-label="Send"
            title="Send (Enter)"
            className="flex h-9 w-9 items-center justify-center rounded-full bg-accent text-accent-fg transition-all hover:scale-105 disabled:scale-100 disabled:bg-surface-2 disabled:text-faint"
          >
            <ArrowUp className="h-4 w-4" strokeWidth={2.5} />
          </button>
        )}
      </div>
    </div>
  );
});
