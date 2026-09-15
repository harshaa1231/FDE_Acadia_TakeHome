# Natural Language Insights Engine

Load any transactional CSV and ask questions about it in plain English. Built for a
merchandising team that keeps re-asking analytics the same questions — this turns
"what sold, where, to whom, what changed" into a self-serve tool instead of a
Slack thread.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture, the biggest decisions,
and where the schema-inference mechanism breaks.

## Quickstart (one command)

```bash
cp backend/.env.example backend/.env
# edit backend/.env and set GROQ_API_KEY (free tier: https://console.groq.com/keys)

docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API + docs: http://localhost:8000/docs

That's it — upload a CSV in the UI and start asking questions. No database to install,
no separate services to start; DuckDB and SQLite are embedded, and both containers come
up from the one command.

### Running without Docker

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set GROQ_API_KEY
uvicorn app.main:app --reload

# frontend, in a second terminal
cd frontend
npm install
cp .env.example .env
npm run dev
```

## Configuration

All config is environment variables, set in `backend/.env` (copy from `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Free-tier Groq API key — powers schema inference and question answering |
| `SQL_MODEL` | `openai/gpt-oss-120b` | Model used for SQL generation (the harder reasoning step) |
| `FAST_MODEL` | `openai/gpt-oss-20b` | Model used for concept mapping and answer phrasing (cheaper/faster) |
| `DATA_DIR` | `./data_store` | Where ingested DuckDB files, job state, and traces live |
| `MAX_UPLOAD_MB` | `200` | Upload size cap |
| `MAX_RESULT_ROWS` | `500` | Row cap injected into every generated SQL query |
| `JOB_WORKER_CONCURRENCY` | `4` | Bounded async worker pool size |

The frontend reads `VITE_API_BASE_URL` (see `frontend/.env.example`), defaulting to
`http://localhost:8000`.

The LLM provider is swappable: `app/nlq/llm_client.py` is a thin interface
(`complete_json` / `complete_text`); Groq is the only implementation wired up, but
nothing above that interface depends on it.

## Asking a question

Through the UI: upload a CSV, wait for ingestion to finish (schema inference runs
automatically), then type a question and hit Ask.

Via the API directly:

```bash
# 1. Upload a CSV — returns a job id immediately, ingestion runs in the background
curl -X POST http://localhost:8000/datasets -F "file=@your_data.csv"
# => {"job_id": "...", "status": "queued"}

# 2. Poll until it succeeds
curl http://localhost:8000/jobs/<job_id>

# 3. Get the dataset id and inspect what was inferred
curl http://localhost:8000/jobs/<job_id>/result
curl http://localhost:8000/datasets/<dataset_id>

# 4. Ask a question — also returns a job id immediately
curl -X POST http://localhost:8000/datasets/<dataset_id>/questions \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 10 products by revenue?"}'

# 5. Poll, then fetch the answer
curl http://localhost:8000/jobs/<job_id>
curl http://localhost:8000/jobs/<job_id>/result

# Optional: see exactly how the answer was produced (prompt, guardrail
# checks, generated SQL, execution outcome, per-stage latency)
curl http://localhost:8000/jobs/<job_id>/trace
```

Full endpoint list and request/response schemas: http://localhost:8000/docs (FastAPI's
auto-generated OpenAPI UI).

## Running tests

```bash
cd backend
source .venv/bin/activate
pytest                 # 28 tests, deterministic, no network calls, ~0.7s
ruff check app tests   # lint
```

CI (`.github/workflows/ci.yml`) runs this same suite on every push, plus a frontend
build. No live LLM calls in CI — every test that touches the LLM boundary uses a
stubbed client, so CI stays fast and never depends on an API key or network access.

## Running the evaluation harness

Unlike the unit/integration tests, this drives the **real, running API** end to end —
the actual async job flow, guardrails, and refusal logic, exactly as a user would hit
them. Requires a running server with a real `GROQ_API_KEY` configured.

```bash
cd backend
source .venv/bin/activate
python -m app.eval.eval_runner --csv ../online_retail_clean.csv
# or, against an already-ingested dataset:
python -m app.eval.eval_runner --dataset-id <dataset_id>
```

It runs the questions in [`app/eval/golden_questions.yaml`](backend/app/eval/golden_questions.yaml)
— the five example question types from the brief, plus deliberately unanswerable
questions the dataset genuinely can't answer — and checks each one against a
ground-truth expected value (computed independently via direct SQL, not reused model
output) or an expected refusal. Prints a pass/fail summary and exits non-zero on any
failure.

## What I cut, given the scope of this exercise

- **Redis/Celery** for the job queue — an in-process asyncio worker pool instead.
  Zero extra infrastructure, trivially satisfies the one-command setup requirement.
  Costs: doesn't survive a process restart mid-job, doesn't scale past one process.
  Documented as the first thing to swap in `docs/DESIGN.md` if this needed to run at
  real scale.
- **Caching, follow-up questions, charts, cost tracking, LLM tracing beyond the
  built-in trace store** — all in the brief's Optional list, skipped per its own
  instruction that skipping costs nothing. Prioritized instead: making the required
  core (schema inference on a genuinely unfamiliar file, async jobs, guardrails,
  eval) solid enough to defend live over adding unlisted features.
- Two items from that Optional list — a semantic layer and LLM observability — were
  built anyway, because they directly reinforce required, heavily-weighted items
  (schema-context construction from an unfamiliar file, and being able to explain a
  live answer or refusal) rather than adding unrelated scope. That reasoning is in
  `docs/DESIGN.md` §4.
- **Multi-entity roles** (e.g. a dataset with both a rider and a driver) — the
  concept map has one `entity_id` slot; a second "who" concept currently falls back
  to a generic dimension rather than being fully modeled. Named as the top
  "what's next" item in `docs/DESIGN.md`.
