import {
  BookOpen,
  Calculator,
  Columns2,
  Compass,
  FileSearch,
  FileText,
  Globe,
  Library,
  Link,
  ListChecks,
  ScanSearch,
  ShieldCheck,
  StickyNote,
  UnfoldVertical,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import type { AgentId } from "@/lib/events";

export const AGENT_ICONS: Record<AgentId, LucideIcon> = {
  planner: Compass,
  doc_analyst: FileSearch,
  web_researcher: Globe,
  verifier: ShieldCheck,
};

const TOOLS: Record<string, { icon: LucideIcon; label: string }> = {
  list_documents: { icon: Library, label: "Listing documents" },
  search_documents: { icon: FileSearch, label: "Searching documents" },
  search_within_document: { icon: ScanSearch, label: "Searching one document" },
  get_adjacent_chunks: { icon: UnfoldVertical, label: "Reading surrounding text" },
  summarize_document: { icon: FileText, label: "Summarizing a document" },
  compare_documents: { icon: Columns2, label: "Comparing documents" },
  calculator: { icon: Calculator, label: "Calculating" },
  web_search: { icon: Globe, label: "Searching the web" },
  fetch_webpage: { icon: Link, label: "Reading a web page" },
  wikipedia_lookup: { icon: BookOpen, label: "Looking up Wikipedia" },
  verify_citations: { icon: ListChecks, label: "Checking citations" },
  check_groundedness: { icon: ShieldCheck, label: "Fact-checking claims" },
  note: { icon: StickyNote, label: "" },
};

export function StepIcon({ tool, className }: { tool: string; className?: string }) {
  const Icon = TOOLS[tool]?.icon ?? Wrench;
  return <Icon className={className} />;
}

export function toolLabel(tool: string): string {
  return TOOLS[tool]?.label ?? tool;
}
