/**
 * Conversation persistence in the browser (localStorage).
 *
 * The backend remembers each thread's messages for follow-up questions; this
 * keeps what the UI needs to redraw past chats (answers, sources, traces).
 * Storage can be unavailable (private mode, quota) — every call degrades to
 * "nothing saved" instead of throwing.
 */
import type { ChatMessage } from "@/hooks/useChat";

export interface ChatMeta {
  id: string;
  title: string;
  updatedAt: number;
}

const INDEX = "docintel.chats";
const KEY = (id: string) => `docintel.chat.${id}`;
const MAX_CHATS = 30;

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage full or unavailable: keep working without persistence */
  }
}

export function listChats(): ChatMeta[] {
  return read<ChatMeta[]>(INDEX, []).sort((a, b) => b.updatedAt - a.updatedAt);
}

export function loadChat(id: string): ChatMessage[] {
  return read<ChatMessage[]>(KEY(id), []);
}

export function saveChat(id: string, messages: ChatMessage[]): ChatMeta[] {
  const firstQuestion = messages.find((m) => m.role === "user")?.content ?? "New chat";
  const title = firstQuestion.length > 52 ? `${firstQuestion.slice(0, 50).trimEnd()}…` : firstQuestion;
  write(KEY(id), messages);
  const others = listChats().filter((c) => c.id !== id);
  const index = [{ id, title, updatedAt: Date.now() }, ...others];
  for (const dropped of index.slice(MAX_CHATS)) removeKey(KEY(dropped.id));
  write(INDEX, index.slice(0, MAX_CHATS));
  return index.slice(0, MAX_CHATS);
}

export function deleteChat(id: string): ChatMeta[] {
  removeKey(KEY(id));
  const index = listChats().filter((c) => c.id !== id);
  write(INDEX, index);
  return index;
}

function removeKey(key: string) {
  try {
    localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}

export function newId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  }
}
