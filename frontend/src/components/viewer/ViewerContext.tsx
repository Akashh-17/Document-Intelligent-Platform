import { createContext, useContext } from "react";

/** What the document panel should show. */
export interface ViewerTarget {
  document: string;
  /** PDF page to open (1-based). */
  page?: number | null;
  /** The cited passage: highlighted on the page and shown in the panel. */
  passage?: string;
  /** Citation number, when opened from an answer. */
  n?: number;
}

export const ViewerContext = createContext<{
  target: ViewerTarget | null;
  open: (target: ViewerTarget) => void;
}>({ target: null, open: () => {} });

export const useViewer = () => useContext(ViewerContext);
