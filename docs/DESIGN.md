# System Design — Natural Language Insights Engine

![Architecture](architecture.svg)

## 1. Components and what each owns

| Component | Owns |
|---|---|
| **FastAPI service** (`app/main.py`, `app/api/*`) | HTTP surface: request validation, structured error envelope (`{"error": {"code","message"}}`), status codes. Every mutating endpoint returns `202 {job_id}` immediately. |
| **Job queue** (`app/jobs/queue.py`) | A bounded in-process asyncio worker pool. Owns concurrency limits, backpressure, and the `queued → running → succeeded/failed` state machine. |
| **Job / dataset / trace stores** (`app/jobs/store.py`, `app/ingestion/dataset_store.py`, `app/jobs/trace_store.py`) | SQLite-backed persistence for job status, per-dataset schema profile + concept map, and per-job pipeline traces. |
| **Ingestion** (`app/ingestion/csv_loader.py`, `schema_profiler.py`) | Loads an arbitrary CSV into a dedicated DuckDB file via `read_csv_auto`, then computes purely descriptive column statistics (type, nulls, cardinality, samples). No business meaning here. |
| **Semantic layer** (`app/semantic/concept_mapper.py`, `metric_registry.py`) | Turns the descriptive profile into business meaning: an LLM proposes structural roles (grouping id, entity id, item id, quantity, monetary amounts, event time) from the file's own column names/types/samples; code deterministically validates every proposal before trusting it. `metric_registry` defines named metrics purely in terms of resolved roles — the extensibility hook. |
| **NL→SQL pipeline** (`app/nlq/*`) | `prompt_builder` assembles the grounded context; the LLM generates SQL, a refusal, or an overview request; `sql_guardrails` validates the SQL is a single, safe SELECT against known tables/columns; `answer_synth` phrases the result, grounded only in what the query actually returned. |
| **React frontend** (`frontend/src`) | Upload, concept-map/schema preview, question box, job polling, answer + SQL + result table, and a trace panel showing how the answer was produced. |

## 2. The path a question takes, end to end

1. **Ingest** — `POST /datasets` saves the upload and returns `202 {job_id}` immediately. A worker runs `csv_loader` (DuckDB's own CSV sniffer handles delimiter/header/type detection), then `schema_profiler` computes objective stats per column. The **semantic layer** sends those stats — plus real sample values pulled from the file — to the LLM, which proposes a role for each of: `event_group_id`, `entity_id`, `item_id`, `item_label`, `quantity`, `unit_amount`, `line_amount`, `event_time`. Every proposal is checked against the real schema (column exists, not already claimed, structurally sane type) before being accepted; a rejected or absent proposal is `not_found`, never guessed. Any column not claimed by a role becomes a `dimension` (if it's a reasonable categorical breakdown) or is exposed raw as `unmapped` — nothing is dropped. The resolved profile + concept map + dimensions are persisted against the dataset id.
2. **Ask** — `POST /datasets/{id}/questions` returns `202 {job_id}`. A worker:
   - loads the persisted concept map and metric registry;
   - runs the **answerability gate**, a structural pre-check (does this dataset have any usable grain + measure at all?) that refuses instantly without an LLM round trip if not;
   - builds a prompt from the resolved roles, available named metrics, dimensions (with real sample values), a few sample rows, and the question — **never** a raw guessed column name;
   - the LLM responds with a SQL `SELECT`, `REFUSE: <reason>`, or `OVERVIEW` (a general "describe this dataset" request, answered from the same structure rather than one query);
   - `sql_guardrails` parses the SQL and rejects anything that isn't a single, safe SELECT against known tables/columns, then caps/injects a `LIMIT`;
   - the query executes against DuckDB with a timeout; on failure, the error is fed back to the LLM once for self-correction before refusing;
   - `answer_synth` phrases the result in natural language, constrained to only state numbers that are actually in the result rows;
   - every stage above is written to `trace_store` under the job id.
3. **Retrieve** — the client polls `GET /jobs/{id}` for status, then `GET /jobs/{id}/result` for the answer + generated SQL + result rows (so the answer is independently checkable), and can inspect `GET /jobs/{id}/trace` for exactly how it was produced.

## 3. How schema context is built from an unfamiliar file, and where it breaks

The mechanism is deliberately two-layered:

- **`schema_profiler` is pure statistics** — type, null fraction, distinct count, uniqueness ratio, min/max, samples. This never changes regardless of domain; it's the same code for a retail CSV or a hospital billing export.
- **`concept_mapper` is LLM-driven, not keyword-driven.** Earlier in development this was a hand-written synonym-list matcher (`"invoice"`, `"customer"`, `"product"`...). It broke on a synthetic rideshare CSV where the customer column was named `rider_handle` — not in any list I'd written — and got silently mis-assigned to the wrong role. The fix wasn't a bigger list (that's an unwinnable game against "column names you did not anticipate"); it was to stop guessing from a fixed vocabulary and instead give the LLM the file's own evidence — real column names, types, cardinality, and sample values — and let it propose roles using genuine language understanding. Every proposal is then deterministically validated: the column must exist, must not already be claimed, and must have a structurally sane type for that role. A proposal that fails validation is rejected outright, not silently patched.

This was stress-tested against real, previously-unseen datasets during development (not just the UCI dev set): a raw UCI Online Retail II export with zero pre-computed columns (revenue correctly **derived** as `quantity × price`), a rideshare CSV with unrelated vocabulary, a Superstore export with no product ID at all (`item_id` correctly stayed `not_found` rather than being forced), and a dataset with a column literally named "记录数" (Chinese for "record count") — normalized and correctly routed to `unmapped` rather than crashing anything.

**Where it still breaks, honestly:**
- **Business rules encoded in value conventions, not columns.** The raw UCI dataset marks cancelled orders by prefixing the invoice number with `C` — there's no `is_cancelled` column. The LLM's sample of that column (20 values) didn't happen to include one, so a question about cancellations was correctly refused rather than guessed — but a differently-sampled run might have caught it and answered inconsistently. Sample-based context can miss rare-but-important patterns.
- **Two legitimate entities, one `entity_id` slot.** A dataset with both a rider and a driver (two "who" concepts) can only have one mapped to `entity_id`; the other has to fall back to a generic `dimension`. The structural role catalog assumes one dominant entity grain.
- **Ambiguity between near-identical columns** (e.g. `unit_cost` vs `unit_price`, both numeric, both plausible `unit_amount` candidates) is resolved by the LLM's single choice with no second opinion recorded — a wrong pick isn't currently self-correcting until a specific question reveals it.
- **The LLM call itself can fail** (malformed JSON, rate limit, timeout). One retry is built in (this was found and fixed during testing — a real transient failure on an external dataset), but a second failure degrades to a conservative structural-only fallback that resolves almost nothing rather than risking a wrong guess. Safe, but a real capability cliff.

## 4. The biggest decisions, and what was rejected

**1. DuckDB (embedded) over Postgres.** DuckDB needs zero server setup, has excellent native CSV type-sniffing (`read_csv_auto`), and is fast enough on 500k+ rows without tuning. Postgres would look more "production," but loading an arbitrary CSV's inferred schema into a real RDBMS is meaningfully more code, and a server dependency works against the 15-minute clean-machine setup requirement. Rejected because the cost wasn't worth it for what this system needs to prove.

**2. LLM-driven semantic layer with deterministic validation, over a hand-written keyword matcher.** Covered in detail in §3 — this is the decision that actually failed once during development and was replaced for a concrete, demonstrated reason, not a hypothetical one. The rejected alternative (name-pattern heuristics) is fundamentally bounded by whatever vocabulary was anticipated in advance, which directly conflicts with "cope with column names you did not anticipate." The tradeoff accepted: every ingest now costs one LLM round trip and depends on the LLM's judgment, mitigated by validating every proposal against the real schema before trusting it, and falling back conservatively (not guessing) when the LLM is unavailable.

**3. In-process asyncio job queue over Redis/Celery.** Zero extra infrastructure, trivially satisfies one-command setup, and is easy to reason about and defend live. The real cost: it doesn't survive a process restart mid-job and doesn't scale past one process. Concurrency is bounded by a worker-pool semaphore and a bounded queue (backpressure returns `429` rather than growing unbounded); a job's failure is caught and recorded per-job, never crashing a worker or blocking the rest of the queue. Documented as the first thing to replace if this needed to run at real scale (§5).

**4. Groq (free-tier, open-weight models) over a paid frontier model, with a strict guardrail layer as compensation.** Open-weight models are less reliable at raw SQL generation and structured output than a frontier model — this was the direct cause of the malformed-JSON failure in §3. Rather than accept that risk unguarded, the system leans harder on deterministic validation everywhere the LLM's output touches something real: column existence and type checks on every role proposal, an AST-level SQL guardrail (single SELECT, known tables/columns only, no DDL/DML) before any generated SQL executes, and answer synthesis constrained to only repeat numbers present in the actual query result. The model can be swapped freely — `llm_client.py` is a thin provider-agnostic interface — without touching this validation layer.

## 5. What I'd build next, in order

1. **Harden the semantic layer's ambiguity handling** — when two columns are both plausible for the same role, surface both to the prompt as labeled candidates instead of the LLM silently picking one, so genuine ambiguity is visible rather than hidden in a single confident-looking proposal.
2. **Redis/Celery** for the job queue, once this needs to survive a restart or run across more than one process — the concrete next step flagged in decision 3.
3. **A small semantic cache** — keyed on (dataset schema hash, question), so re-asking or re-loading the same file skips re-profiling and re-answering. Free 10x on repeated demo/eval runs.
4. **Multi-entity role support** — let the concept map hold more than one `entity_id`-shaped role (e.g. buyer and seller) instead of forcing a single slot, closing the gap named in §3.
5. **Follow-up questions** — thread prior question/answer/SQL into the next prompt so "and by country?" works without restating context.
6. **Charts** — the result rows are already tabular; a minimal chart view is a frontend-only addition once the data contract is stable.

## What I cut, given the scope of this exercise

- **Redis/Celery, caching, follow-up questions, charts, cost tracking** — all in the Optional list, deliberately skipped per the brief ("skipping all of it costs you nothing"), documented above as what's next.
- **The two optional items I built anyway** (semantic layer as a first-class component, LLM observability via the trace store) were elevated specifically because they directly serve required, heavily-weighted requirements — genuine schema inference on an unfamiliar file, and being able to defend a live answer or refusal — not because more features are inherently better.
