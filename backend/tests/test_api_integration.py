"""End-to-end API tests: FastAPI TestClient driving the real async flow
(upload -> poll -> result, question -> poll -> result -> trace) against a
stubbed LLM client. No network calls, no API key needed, fully
deterministic - this is what runs in CI on every push.
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.jobs import handlers as handlers_module

TINY_CSV = (
    "order_ref,customer,product,qty,price,order_date\n"
    "O1,cust_a,Widget,2,9.99,2024-01-01\n"
    "O1,cust_a,Gadget,1,19.99,2024-01-01\n"
    "O2,cust_b,Widget,3,9.99,2024-01-02\n"
    "O3,cust_a,Gadget,1,19.99,2024-01-05\n"
)


class FakeLLMClient:
    """complete_json is used for concept mapping; complete_text is used
    both for SQL generation (sql_model) and answer synthesis (fast_model) -
    differentiate by which model was requested, exactly like the real
    handler does."""

    async def complete_json(self, system: str, user: str, model: str) -> dict:
        return {
            "event_group_id": {"column": "order_ref", "confidence": 0.9, "reasoning": "repeats"},
            "entity_id": {"column": "customer", "confidence": 0.9, "reasoning": "who bought"},
            "item_id": {"column": "product", "confidence": 0.9, "reasoning": "what was bought"},
            "quantity": {"column": "qty", "confidence": 0.9, "reasoning": "unit count"},
            "unit_amount": {"column": "price", "confidence": 0.9, "reasoning": "per-unit price"},
            "line_amount": {"column": None, "derive_from_quantity_and_unit_amount": True, "confidence": 0.8, "reasoning": "qty*price"},
            "event_time": {"column": "order_date", "confidence": 0.9, "reasoning": "transaction date"},
        }

    async def complete_text(self, system: str, user: str, model: str) -> str:
        if model == settings.sql_model:
            if "REFUSE_ME" in user:
                return "REFUSE: no such concept is available in this dataset"
            return 'SELECT "product", SUM("qty" * "price") AS revenue FROM transactions GROUP BY "product" ORDER BY revenue DESC'
        return "Here is a canned natural-language answer describing the result."


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Fresh data_dir per test, and fresh store/queue singletons to match -
    these are cached at module level, so just changing settings.data_dir
    would silently leave every test sharing the first test's instances."""
    import app.ingestion.dataset_store as dataset_store_module
    import app.jobs.queue as queue_module
    import app.jobs.store as job_store_module
    import app.jobs.trace_store as trace_store_module

    monkeypatch.setattr(handlers_module, "get_llm_client", lambda: FakeLLMClient())
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(job_store_module, "_store", None)
    monkeypatch.setattr(trace_store_module, "_store", None)
    monkeypatch.setattr(dataset_store_module, "_store", None)
    monkeypatch.setattr(queue_module, "_queue", None)
    yield


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _poll_job(client: TestClient, job_id: str, timeout_polls: int = 100) -> dict:
    import time

    for _ in range(timeout_polls):
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def test_full_ingest_and_question_flow(client):
    resp = client.post("/datasets", files={"file": ("tiny.csv", io.BytesIO(TINY_CSV.encode()), "text/csv")})
    assert resp.status_code == 202
    ingest_job_id = resp.json()["job_id"]

    status = _poll_job(client, ingest_job_id)
    assert status["status"] == "succeeded", status

    result = client.get(f"/jobs/{ingest_job_id}/result").json()
    dataset_id = result["dataset_id"]
    assert result["row_count"] == 4
    assert "event_group_id" in result["resolved_roles"]

    ds = client.get(f"/datasets/{dataset_id}")
    assert ds.status_code == 200
    ds_body = ds.json()
    assert ds_body["row_count"] == 4
    assert ds_body["roles"]["entity_id"]["available"] is True

    q_resp = client.post(f"/datasets/{dataset_id}/questions", json={"question": "Which product earned the most revenue?"})
    assert q_resp.status_code == 202
    q_job_id = q_resp.json()["job_id"]

    q_status = _poll_job(client, q_job_id)
    assert q_status["status"] == "succeeded", q_status

    q_result = client.get(f"/jobs/{q_job_id}/result").json()
    assert q_result["status"] == "answered"
    assert "revenue" in q_result["sql"].lower()
    assert q_result["answer"]

    trace = client.get(f"/jobs/{q_job_id}/trace").json()
    stage_names = [s["stage"] for s in trace["stages"]]
    assert "dataset_loaded" in stage_names
    assert "answerability_gate" in stage_names
    assert any(s.startswith("sql_attempt") for s in stage_names)
    assert "answer_synthesized" in stage_names


def test_refusal_flow_returns_succeeded_job_with_refused_status(client):
    resp = client.post("/datasets", files={"file": ("tiny.csv", io.BytesIO(TINY_CSV.encode()), "text/csv")})
    ingest_job_id = resp.json()["job_id"]
    result = client.get(f"/jobs/{_poll_job(client, ingest_job_id) and ingest_job_id}/result").json()
    dataset_id = result["dataset_id"]

    q_resp = client.post(f"/datasets/{dataset_id}/questions", json={"question": "REFUSE_ME please"})
    q_job_id = q_resp.json()["job_id"]
    q_status = _poll_job(client, q_job_id)

    assert q_status["status"] == "succeeded"  # a refusal is a valid successful outcome, not a job failure
    q_result = client.get(f"/jobs/{q_job_id}/result").json()
    assert q_result["status"] == "refused"
    assert q_result["reason"]


def test_rejects_non_csv_upload(client):
    resp = client.post("/datasets", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "validation_error"


def test_rejects_empty_csv_upload(client):
    resp = client.post("/datasets", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")})
    assert resp.status_code == 400


def test_question_against_unknown_dataset_returns_404(client):
    resp = client.post("/datasets/does-not-exist/questions", json={"question": "anything"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_empty_question_returns_structured_422(client):
    resp = client.post("/datasets", files={"file": ("tiny.csv", io.BytesIO(TINY_CSV.encode()), "text/csv")})
    ingest_job_id = resp.json()["job_id"]
    result = client.get(f"/jobs/{_poll_job(client, ingest_job_id) and ingest_job_id}/result").json()
    dataset_id = result["dataset_id"]

    resp = client.post(f"/datasets/{dataset_id}/questions", json={"question": ""})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert "detail" not in body  # never the raw FastAPI/Pydantic shape


def test_unknown_job_id_returns_404(client):
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
