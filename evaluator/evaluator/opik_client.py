"""Opik client wrapper for the evaluator.

Mirrors (a much smaller version of) circuit's Opik client. It gives the
evaluate job four things:
  - a dataset (the benchmark questions + ground truth, visible in the Opik UI),
  - an experiment (one benchmark run, grouping all the scored items),
  - trace fetch by id (to read back the agent's tool spans), and
  - feedback-score writes onto each trace.

Everything is optional: when ``OPIK_URL_OVERRIDE`` is unset the methods no-op so
the jobs still run (just without anything showing up in Opik).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from evaluator.config import settings
from evaluator.types import BenchmarkItem, Score


def _uuid7() -> str:
    from uuid6 import uuid7

    return str(uuid7())



class OpikClient:
    """Thin convenience wrapper around the high-level Opik SDK."""

    def __init__(self) -> None:
        self.project_name = settings.opik_project_name
        self._client: Any = None
        if settings.opik_url_override:
            try:
                import opik

                self._client = opik.Opik(project_name=self.project_name)
            except Exception:  # noqa: BLE001
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ---- datasets ---------------------------------------------------------
    def sync_dataset(self, name: str, items: list[BenchmarkItem]) -> None:
        """Create (or reuse) an Opik dataset holding the benchmark questions."""
        if self._client is None:
            return
        dataset = self._client.get_or_create_dataset(name)
        if dataset.get_items():
            return
        dataset.insert(
            [
                {
                    "input": it.question,
                    "expected_output": it.answer,
                    "golden_docs": it.golden_docs,
                    "query_id": it.query_id,
                }
                for it in items
            ]
        )

    # ---- experiments ------------------------------------------------------
    def new_experiment(self, dataset_name: str) -> str | None:
        """Create an experiment for one benchmark run; return its name."""
        if self._client is None:
            return None
        stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M")
        name = f"{dataset_name}-{stamp}"
        try:
            self._client.create_experiment(name=name, dataset_name=dataset_name)
        except Exception:  # noqa: BLE001
            return None
        return name

    # ---- traces -----------------------------------------------------------
    def get_trace_spans(self, trace_id: str) -> list[dict[str, Any]]:
        """Return the trace's spans as plain dicts (name/type/input/output/metadata)."""
        if self._client is None or not trace_id:
            return []
        try:
            rest = self._client.rest_client
            spans: list[dict[str, Any]] = []
            page = 1
            while True:
                resp = rest.spans.get_spans_by_project(
                    project_name=self.project_name,
                    trace_id=trace_id,
                    page=page,
                    size=100,
                )
                content = getattr(resp, "content", []) or []
                for s in content:
                    spans.append(s.model_dump() if hasattr(s, "model_dump") else dict(s))
                if len(content) < 100:
                    break
                page += 1
            return spans
        except Exception:  # noqa: BLE001
            return []

    # ---- feedback ---------------------------------------------------------
    def post_scores(self, trace_id: str, scores: list[Score]) -> None:
        if self._client is None or not trace_id or not scores:
            return
        try:
            self._client.log_traces_feedback_scores(
                [
                    {
                        "id": trace_id,
                        "name": s.name,
                        "value": s.value,
                        "reason": s.reason,
                    }
                    for s in scores
                ],
                project_name=self.project_name,
            )
        except Exception:  # noqa: BLE001
            pass

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                pass