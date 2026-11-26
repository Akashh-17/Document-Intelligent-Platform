import type { Source } from "./events";

/** A source as shown under one answer: renumbered 1..n in order of first citation. */
export interface ShownSource extends Source {
  n: number;
}

export interface Renumbered {
  text: string;
  cited: ShownSource[];
  /** Sources the agents consulted but the answer didn't cite. */
  others: Source[];
}

const CITE = /\[(\d+)\]/g;

/**
 * The backend numbers evidence turn-wide (a source found third is [3] even if
 * the answer only cites it). Readers expect 1, 2, 3 in reading order, so we
 * renumber by first appearance and rewrite the markers to match.
 */
export function renumberCitations(text: string, sources: Source[]): Renumbered {
  const byIndex = new Map(sources.map((s) => [s.index, s]));
  const mapping = new Map<number, number>();
  for (const m of text.matchAll(CITE)) {
    const old = Number(m[1]);
    if (byIndex.has(old) && !mapping.has(old)) mapping.set(old, mapping.size + 1);
  }
  const rewritten = text.replace(CITE, (whole, num) => {
    const n = mapping.get(Number(num));
    return n ? `[${n}]` : byIndex.size ? "" : whole; // drop markers pointing nowhere
  });
  const cited = [...mapping.entries()]
    .sort((a, b) => a[1] - b[1])
    .map(([old, n]) => ({ ...byIndex.get(old)!, n }));
  const citedSet = new Set(mapping.keys());
  const others = sources.filter((s) => !citedSet.has(s.index));
  return { text: rewritten, cited, others };
}

const STOP = new Set(
  "a an and are as at be by can do does for from how i in is it its me my of on or our so that the their then there these this to was we what when where which who why will with you your about into than also".split(
    " ",
  ),
);

/** Words from the question worth highlighting in an evidence passage. */
export function highlightTerms(question: string): string[] {
  const words = question.toLowerCase().match(/[a-z0-9][a-z0-9._-]*[a-z0-9]|[a-z0-9]/g) ?? [];
  return [...new Set(words.filter((w) => w.length > 2 && !STOP.has(w)))];
}

/** Split text into plain / highlighted runs for the given terms. */
export function markRuns(text: string, terms: string[]): { text: string; hit: boolean }[] {
  if (!terms.length) return [{ text, hit: false }];
  const escaped = terms
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${escaped.join("|")})`, "gi");
  return text
    .split(re)
    .filter(Boolean)
    .map((part) => ({ text: part, hit: terms.includes(part.toLowerCase()) }));
}

export function hostOf(url?: string | null): string {
  if (!url) return "";
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

/** "aurora_error_codes.md (p. 3)" -> { name: "aurora_error_codes.md", page: "p. 3" } */
export function splitTitle(title: string): { name: string; detail: string } {
  const m = /^(.*?)\s*\((p\. \d+|summary)\)$/.exec(title);
  return m ? { name: m[1], detail: m[2] } : { name: title, detail: "" };
}

/** Readable plain text from a markdown chunk (headings, tables, code marks removed). */
export function plainPassage(text: string): string {
  return text
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^\s*\|?\s*-{3,}[-|\s:]*$/gm, " ")
    .replace(/\s*\|\s*/g, " · ")
    .replace(/[`*]{1,2}/g, "") // keep underscores: they are part of identifiers like CLIENT_CERT_MISSING
    .replace(/\s+/g, " ")
    .replace(/^(\s*·\s*)+|(\s*·\s*)+$/g, "")
    .trim();
}

/**
 * The ~width-char window of a passage that shares the most terms with the
 * question, so the evidence card shows the part that matters, not the
 * chunk's opening lines.
 */
export function bestWindow(text: string, terms: string[], width = 300): string {
  const flat = plainPassage(text);
  if (flat.length <= width || !terms.length) {
    return flat.length > width ? `${flat.slice(0, width).trimEnd()}…` : flat;
  }
  const lower = flat.toLowerCase();
  let best = 0;
  let bestScore = -1;
  for (let start = 0; start < flat.length - width / 2; start += 40) {
    const win = lower.slice(start, start + width);
    const score = terms.reduce((n, t) => n + (win.includes(t) ? (t.length > 4 ? 2 : 1) : 0), 0);
    if (score > bestScore) {
      best = start;
      bestScore = score;
    }
  }
  // snap to a word boundary
  if (best > 0) {
    const space = flat.indexOf(" ", best);
    best = space > 0 && space - best < 25 ? space + 1 : best;
  }
  const slice = flat.slice(best, best + width).trimEnd();
  return `${best > 0 ? "…" : ""}${slice}${best + width < flat.length ? "…" : ""}`;
}
