import { useCallback, useEffect, useState } from "react";
import { deleteDocument, fetchHealth, listDocuments, type DocumentInfo, type Health } from "@/lib/api";

/** The knowledge base (uploaded documents) and backend status. */
export function useLibrary() {
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([fetchHealth(), listDocuments()]);
      setHealth(h);
      setDocs(d);
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  // Load once on start.
  useEffect(() => {
    refresh();
  }, [refresh]);

  // The backend may still be starting: retry every 4 s, but ONLY until it answers.
  // (Depending on `health` here used to re-run the effect after every refresh —
  // each refresh stores a new object — which hammered the API in a tight loop.)
  const connected = health !== null && !offline;
  useEffect(() => {
    if (connected) return;
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, [connected, refresh]);

  const remove = useCallback(
    async (name: string) => {
      await deleteDocument(name);
      await refresh();
    },
    [refresh],
  );

  return { docs, health, offline, refresh, remove };
}

export type Library = ReturnType<typeof useLibrary>;
