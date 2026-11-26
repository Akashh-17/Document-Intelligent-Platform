import { useCallback, useState } from "react";
import { uploadDocuments } from "@/lib/api";
import { useToast } from "@/components/ui/Toast";

const ACCEPTED = [".pdf", ".txt", ".md", ".markdown"];

export function isAccepted(file: File) {
  return ACCEPTED.some((ext) => file.name.toLowerCase().endsWith(ext));
}

/** Upload files, report the outcome as toasts, return true if anything was added. */
export function useUploader(onAdded: () => void) {
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const upload = useCallback(
    async (files: File[]): Promise<boolean> => {
      if (!files.length) return false;
      setBusy(true);
      try {
        const r = await uploadDocuments(files);
        const ok = r.files.filter((f) => !f.error);
        const failed = r.files.filter((f) => f.error);
        if (ok.length) {
          const names = ok.map((f) => f.name);
          toast({
            tone: "ok",
            title:
              ok.length === 1 ? `Added ${names[0]} to your library` : `Added ${ok.length} files to your library`,
            detail: `${r.chunks_added} passage${r.chunks_added === 1 ? " is" : "s are"} ready to search.`,
          });
          onAdded();
        }
        for (const f of failed) toast({ tone: "error", title: `Couldn't add ${f.name}`, detail: f.error ?? undefined });
        for (const f of ok)
          for (const w of f.warnings ?? [])
            toast({ tone: "warn", title: `${f.name} contains hidden instructions`, detail: w });
        return ok.length > 0;
      } catch (e) {
        toast({ tone: "error", title: "Upload failed", detail: e instanceof Error ? e.message : String(e) });
        return false;
      } finally {
        setBusy(false);
      }
    },
    [onAdded, toast],
  );

  return { busy, upload };
}
