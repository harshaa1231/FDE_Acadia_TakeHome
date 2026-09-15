"""Guardrails and evaluation harness.

Drives the real, running API - not a shortcut through internal functions -
so this exercises the actual async job flow, guardrails, and refusal logic
exactly as a user would hit them. Requires a running server (see README)
and a GROQ_API_KEY configured on it.

Usage:
    python -m app.eval.eval_runner --csv ../online_retail_clean.csv
    python -m app.eval.eval_runner --dataset-id <existing-dataset-id>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import yaml

_QUESTIONS_PATH = Path(__file__).parent / "golden_questions.yaml"


def _poll(client: httpx.Client, job_id: str, timeout_s: float = 120.0) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        r = client.get(f"/jobs/{job_id}")
        r.raise_for_status()
        status = r.json()["status"]
        if status in ("succeeded", "failed"):
            return r.json()
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s")


def ingest(client: httpx.Client, csv_path: str) -> str:
    with open(csv_path, "rb") as f:
        r = client.post("/datasets", files={"file": (Path(csv_path).name, f, "text/csv")})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    status = _poll(client, job_id, timeout_s=300.0)
    if status["status"] != "succeeded":
        raise RuntimeError(f"ingestion failed: {status.get('error')}")
    result = client.get(f"/jobs/{job_id}/result").json()
    print(f"Ingested dataset {result['dataset_id']}: {result['row_count']} rows, "
          f"roles resolved: {result['resolved_roles']}")
    return result["dataset_id"]


def ask(client: httpx.Client, dataset_id: str, question: str) -> dict:
    r = client.post(f"/datasets/{dataset_id}/questions", json={"question": question})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    status = _poll(client, job_id)
    if status["status"] != "succeeded":
        return {"status": "job_failed", "error": status.get("error")}
    return client.get(f"/jobs/{job_id}/result").json()


def check(case: dict, result: dict) -> tuple[bool, str]:
    qtype = case["type"]

    if qtype == "refusal":
        if result.get("status") == "refused":
            return True, f"refused: {result.get('reason')}"
        return False, f"expected refusal, got status={result.get('status')!r}"

    if qtype == "exists":
        if result.get("status") == "answered" and result.get("rows"):
            return True, f"answered with {len(result['rows'])} row(s)"
        return False, f"expected an answer with rows, got {result}"

    if qtype == "numeric_row":
        if result.get("status") != "answered":
            return False, f"expected an answer, got status={result.get('status')!r}"
        row_i = case["check"]["row"]
        try:
            row = result["rows"][row_i]
        except IndexError:
            return False, f"result has no row {row_i}: {result.get('rows')}"
        # Search the whole row rather than a fixed column index: the LLM
        # generates SQL fresh per question, so column order/count between
        # two equally-correct queries isn't guaranteed - pinning to a
        # position is fragile in a way that has nothing to do with
        # correctness. What matters is the expected value appears somewhere
        # in the row DuckDB actually returned.
        expected, tol = case["expected"], case.get("tolerance", 0)
        for cell in row:
            if isinstance(cell, (int, float)) and abs(cell - expected) <= tol:
                return True, f"{cell} (expected {expected} ± {tol}) in row {row}"
        return False, f"expected {expected} (±{tol}) somewhere in row, got {row}"

    return False, f"unknown check type {qtype!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--csv", help="CSV to ingest fresh before running the eval")
    parser.add_argument("--dataset-id", help="Use an already-ingested dataset instead of --csv")
    args = parser.parse_args()

    if not args.csv and not args.dataset_id:
        parser.error("one of --csv or --dataset-id is required")

    cases = yaml.safe_load(_QUESTIONS_PATH.read_text())

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        dataset_id = args.dataset_id or ingest(client, args.csv)

        passed, failed = 0, 0
        print(f"\nRunning {len(cases)} golden questions against dataset {dataset_id}\n")
        for i, case in enumerate(cases):
            if i > 0:
                time.sleep(2.0)  # be polite to free-tier LLM rate limits between questions
            question = case["question"]
            try:
                result = ask(client, dataset_id, question)
                ok, detail = check(case, result)
            except Exception as e:
                ok, detail = False, f"exception: {e}"

            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"[{status}] {question}\n       {detail}")

        print(f"\n{passed} passed, {failed} failed out of {len(cases)}")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
