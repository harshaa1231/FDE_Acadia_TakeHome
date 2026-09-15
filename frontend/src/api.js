const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handle(resp) {
  if (!resp.ok) {
    let body;
    try {
      body = await resp.json();
    } catch {
      body = null;
    }
    const message = body?.error?.message || `Request failed with status ${resp.status}`;
    const err = new Error(message);
    err.status = resp.status;
    err.code = body?.error?.code;
    throw err;
  }
  return resp.json();
}

export async function uploadDataset(file) {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${BASE_URL}/datasets`, { method: "POST", body: form });
  return handle(resp);
}

export async function getJob(jobId) {
  const resp = await fetch(`${BASE_URL}/jobs/${jobId}`);
  return handle(resp);
}

export async function getJobResult(jobId) {
  const resp = await fetch(`${BASE_URL}/jobs/${jobId}/result`);
  return handle(resp);
}

export async function getJobTrace(jobId) {
  const resp = await fetch(`${BASE_URL}/jobs/${jobId}/trace`);
  return handle(resp);
}

export async function getDataset(datasetId) {
  const resp = await fetch(`${BASE_URL}/datasets/${datasetId}`);
  return handle(resp);
}

export async function askQuestion(datasetId, question) {
  const resp = await fetch(`${BASE_URL}/datasets/${datasetId}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return handle(resp);
}

export async function pollJob(jobId, { intervalMs = 700, timeoutMs = 120000 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const status = await getJob(jobId);
    if (status.status === "succeeded" || status.status === "failed") {
      return status;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("Timed out waiting for job to finish");
}
