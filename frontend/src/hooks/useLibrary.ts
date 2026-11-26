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

  useEffect(() => {
    refresh();
    // The backend may still be starting (indexing sample docs): retry until it answers.
    const timer = setInterval(() => {
      if (offline || !health) refresh();
    }, 4000);
    return () => clearInterval(timer);
  }, [refresh, offline, health]);

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
