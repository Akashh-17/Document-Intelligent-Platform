export interface Health {
  status: string;
  llm_provider: string;
  llm_configured: boolean;
  model_fast: string;
  model_heavy: string;
  embedding_model: string;
  web_backend: string;
  langsmith_tracing: boolean;
  documents_indexed: number;
}

async function errorDetail(res: Response, label: string): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // not JSON
  }
  return `${label} failed (HTTP ${res.status})`;
}

export async function fetchHealth(): Promise<Health> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(await errorDetail(res, "health"));
  return res.json();
}

export interface IngestFileResult {
  name: string;
  chunks: number;
  replaced: number;
  error?: string | null;
}

export interface IngestResult {
  chunks_added: number;
  files: IngestFileResult[];
}

export async function uploadDocuments(files: File[]): Promise<IngestResult> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch("/api/ingest", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res, "Upload"));
  return res.json();
}

export async function sendFeedback(runId: string, score: number): Promise<boolean> {
  try {
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, score }),
    });
    const body = await res.json();
    return Boolean(body?.ok);
  } catch {
    return false;
  }
}

export { errorDetail };
