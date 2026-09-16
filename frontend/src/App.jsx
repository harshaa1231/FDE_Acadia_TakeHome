import { useState, useRef } from "react";
import { uploadDataset, getJobResult, getDataset, askQuestion, getJobTrace, pollJob } from "./api.js";

function Chip({ label, tone = "default" }) {
  return <span className={`chip chip-${tone}`}>{label}</span>;
}

function Spinner({ muted }) {
  return <span className={`spinner${muted ? " spinner-muted" : ""}`} aria-hidden="true" />;
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy(e) {
    e.preventDefault(); // don't toggle the parent <details>
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard API can be unavailable (insecure context, permissions);
      // fail silently rather than surface an error for a copy convenience.
      return;
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <button type="button" className="copy-button" onClick={handleCopy}>
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function formatCell(v, format) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (format === "currency") {
      return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    if (format === "integer") {
      return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
    return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return String(v);
}

function UploadPanel({ onDatasetReady }) {
  const [status, setStatus] = useState("idle"); // idle | ingesting | ready | error
  const [message, setMessage] = useState("");
  const [dataset, setDataset] = useState(null);
  const fileInput = useRef(null);

  async function handleUpload(e) {
    e.preventDefault();
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    setStatus("ingesting");
    setMessage(`Uploading ${file.name}…`);
    try {
      const { job_id } = await uploadDataset(file);
      setMessage("Inferring schema and resolving concepts…");
      const finalStatus = await pollJob(job_id, { timeoutMs: 180000 });
      if (finalStatus.status === "failed") {
        throw new Error(finalStatus.error?.message || "Ingestion failed");
      }
      const result = await getJobResult(job_id);
      const info = await getDataset(result.dataset_id);
      setDataset(info);
      setStatus("ready");
      setMessage("");
      onDatasetReady(info);
    } catch (err) {
      setStatus("error");
      setMessage(err.message);
    }
  }

  return (
    <section className="panel">
      <h2><span className="step-badge">1</span>Load a CSV</h2>
      <form onSubmit={handleUpload} className="row">
        <input type="file" accept=".csv" ref={fileInput} />
        <button type="submit" disabled={status === "ingesting"}>
          {status === "ingesting" && <Spinner />}
          {status === "ingesting" ? "Working…" : "Upload"}
        </button>
      </form>
      {message && (
        <div className="status-line">
          {status === "ingesting" && <Spinner muted />}
          <span className={status === "error" ? "error-text" : "muted"}>{message}</span>
        </div>
      )}

      {dataset && (
        <div className="dataset-summary">
          <p>
            <strong>{dataset.original_filename}</strong> &mdash; {dataset.row_count.toLocaleString()} rows,{" "}
            {dataset.column_count} columns
            {dataset.rows_skipped > 0 && ` (${dataset.rows_skipped} rows skipped as malformed)`}
          </p>

          <div className="summary-label">Concepts resolved from this file</div>
          <div className="chip-row">
            {Object.entries(dataset.roles).map(([role, info]) => (
              <Chip
                key={role}
                tone={info.available ? "ok" : "default"}
                label={info.available ? `${role} → ${info.expression}` : `${role}: not found`}
              />
            ))}
          </div>

          {dataset.dimensions.length > 0 && (
            <>
              <div className="summary-label">Dimensions available for breakdown</div>
              <div className="chip-row">
                {dataset.dimensions.map((d) => (
                  <Chip key={d.column} label={d.column} tone="dim" />
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}

function ResultTable({ answer }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {answer.columns.map((c) => (
              <th key={c}>
                {c}
                {answer.column_formats?.[c] === "currency" && (
                  <span className="col-format-hint" title="Monetary value (currency unspecified in the source file)"> ¤</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {answer.rows.slice(0, 20).map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => {
                const format = answer.column_formats?.[answer.columns[j]];
                return (
                  <td key={j} className={typeof v === "number" ? "cell-number" : undefined}>
                    {formatCell(v, format)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function UsageNote({ usage }) {
  if (!usage) return null;
  const cost =
    usage.estimated_cost_usd !== undefined
      ? ` · ~$${usage.estimated_cost_usd.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 })}`
      : "";
  return (
    <p className="usage-note">
      {usage.total_tokens.toLocaleString()} tokens · {usage.llm_calls} LLM call{usage.llm_calls === 1 ? "" : "s"}
      {cost}
    </p>
  );
}

function TracePanel({ jobId }) {
  const [open, setOpen] = useState(false);
  const [stages, setStages] = useState(null);
  const [loading, setLoading] = useState(false);

  async function toggle() {
    if (!open && stages === null) {
      setLoading(true);
      try {
        const trace = await getJobTrace(jobId);
        setStages(trace.stages);
      } finally {
        setLoading(false);
      }
    }
    setOpen(!open);
  }

  return (
    <div className="trace-panel">
      <button type="button" className="link-button" onClick={toggle}>
        {open ? "Hide" : "Show"} how this was answered
      </button>
      {open && (
        <div className="trace-body">
          {loading && <p className="muted small">Loading trace…</p>}
          {stages?.map((s, i) => (
            <details key={i} open={i === stages.length - 1}>
              <summary>{s.stage}</summary>
              <pre>{JSON.stringify(s, null, 2)}</pre>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}

function QuestionPanel({ dataset }) {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("idle"); // idle | asking | done | error
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState(null);
  const [jobId, setJobId] = useState(null);

  async function handleAsk(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setStatus("asking");
    setAnswer(null);
    setMessage("Thinking…");
    try {
      const { job_id } = await askQuestion(dataset.dataset_id, question);
      setJobId(job_id);
      const finalStatus = await pollJob(job_id, { timeoutMs: 60000 });
      if (finalStatus.status === "failed") {
        throw new Error(finalStatus.error?.message || "Something went wrong answering this question");
      }
      const result = await getJobResult(job_id);
      setAnswer(result);
      setStatus("done");
      setMessage("");
    } catch (err) {
      setStatus("error");
      setMessage(err.message);
    }
  }

  return (
    <section className="panel">
      <h2><span className="step-badge">2</span>Ask a question</h2>
      <form onSubmit={handleAsk} className="row">
        <input
          type="text"
          placeholder="e.g. What are the top 10 products by revenue?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="question-input"
        />
        <button type="submit" disabled={status === "asking"}>
          {status === "asking" && <Spinner />}
          {status === "asking" ? "Working…" : "Ask"}
        </button>
      </form>
      {message && (
        <div className="status-line">
          {status === "asking" && <Spinner muted />}
          <span className={status === "error" ? "error-text" : "muted"}>{message}</span>
        </div>
      )}

      {answer?.status === "refused" && (
        <div className="answer-box refused">
          <div className="answer-kicker">Refused</div>
          <p className="answer-text">{answer.reason}</p>
          <UsageNote usage={answer.usage} />
          {jobId && <TracePanel jobId={jobId} />}
        </div>
      )}

      {answer?.status === "answered" && (
        <div className="answer-box">
          <div className="answer-kicker">Answer</div>
          <p className="answer-text">{answer.answer}</p>

          {answer.rows.length > 1 ? (
            <>
              <div className="summary-label">Result rows ({answer.rows.length})</div>
              <ResultTable answer={answer} />
            </>
          ) : (
            answer.rows.length === 1 && (
              <details>
                <summary>Result row</summary>
                <ResultTable answer={answer} />
              </details>
            )
          )}

          <details>
            <summary>Generated SQL</summary>
            <div className="code-block">
              <pre>{answer.sql}</pre>
              <CopyButton text={answer.sql} />
            </div>
          </details>
          <UsageNote usage={answer.usage} />
          {jobId && <TracePanel jobId={jobId} />}
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [dataset, setDataset] = useState(null);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-mark">NL</div>
        <div>
          <h1>Natural Language Insights Engine</h1>
          <p>Load any transactional CSV and ask questions about it in plain English.</p>
        </div>
      </header>

      <UploadPanel onDatasetReady={setDataset} />

      {dataset ? (
        <QuestionPanel dataset={dataset} />
      ) : (
        <section className="panel panel-empty">
          <p>Upload a dataset above to start asking questions.</p>
        </section>
      )}
    </div>
  );
}
