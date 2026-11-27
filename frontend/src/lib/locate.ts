/**
 * Find a cited passage inside a PDF page's text, to highlight it.
 *
 * The passage was extracted by pypdf on the server; the page text here comes
 * from pdf.js in the browser. The two split text differently (line breaks,
 * hyphenation, spacing), so an exact string search fails. Instead we compare
 * word sequences: find a 5-word run from the START of the passage on the page,
 * and a 5-word run from its END after it, trying later/earlier runs if the
 * first ones don't match. Everything between the two anchors is the passage.
 */

const SHINGLE = 5;

function words(text: string): string[] {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .split(" ")
    .filter(Boolean);
}

function findRun(hay: string[], run: string[], from = 0): number {
  outer: for (let i = from; i <= hay.length - run.length; i++) {
    for (let j = 0; j < run.length; j++) if (hay[i + j] !== run[j]) continue outer;
    return i;
  }
  return -1;
}

/**
 * Indices of the page's text items that belong to ``passage``.
 * ``items`` is the page's text content in order ("" for non-text entries).
 */
export function locatePassage(items: string[], passage: string): Set<number> {
  const target = words(passage);
  const page: string[] = [];
  const owner: number[] = []; // page word -> text item it came from
  items.forEach((str, i) => {
    for (const w of words(str)) {
      page.push(w);
      owner.push(i);
    }
  });
  const marked = new Set<number>();
  if (target.length < SHINGLE || page.length < SHINGLE) return marked;

  // start anchor: the first 5-word run of the passage that appears on the page
  let start = -1;
  for (let s = 0; s + SHINGLE <= Math.min(target.length, 60); s += 3) {
    const at = findRun(page, target.slice(s, s + SHINGLE));
    if (at !== -1) {
      start = Math.max(0, at - s);
      break;
    }
  }
  if (start === -1) return marked;

  // end anchor: the last 5-word run of the passage that appears after the start
  let end = Math.min(page.length - 1, start + target.length - 1);
  for (let e = 0; e + SHINGLE <= Math.min(target.length, 60); e += 3) {
    const run = target.slice(target.length - SHINGLE - e, target.length - e);
    const at = findRun(page, run, start);
    if (at !== -1) {
      end = Math.min(page.length - 1, at + SHINGLE - 1 + e);
      break;
    }
  }

  for (let i = start; i <= end; i++) marked.add(owner[i]);
  return marked;
}

/** Character range of ``passage`` in a plain-text document, or null. */
export function locateInText(text: string, passage: string): [number, number] | null {
  const norm = (s: string) => s.replace(/\s+/g, " ").trim();
  const probe = norm(passage).slice(0, 80);
  if (!probe) return null;
  // Map normalized offsets back to the original text.
  const map: number[] = [];
  let flat = "";
  let space = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (/\s/.test(ch)) {
      if (!space && flat.length) {
        flat += " ";
        map.push(i);
      }
      space = true;
    } else {
      flat += ch;
      map.push(i);
      space = false;
    }
  }
  const at = flat.indexOf(probe);
  if (at === -1) return null;
  const length = Math.min(norm(passage).length, flat.length - at);
  return [map[at], map[at + length - 1] + 1];
}
