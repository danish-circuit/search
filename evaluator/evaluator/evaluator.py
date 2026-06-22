"""Run the full metric set on one item's agent response."""

from __future__ import annotations

from evaluator import metrics
from evaluator.judges import Judge
from evaluator.types import ItemResult, Score

_JUDGED = [
    metrics.accuracy,
    metrics.answer_completeness,
    metrics.answer_correctness,
    metrics.faithfulness,
    metrics.context_precision,
    metrics.context_recall,
]

_DETERMINISTIC = [
    metrics.document_recall,
    metrics.gold_hit_at_5,
    metrics.gold_mrr,
]



class Evaluator:
    """Owns one shared Judge and scores items against it."""

    def __init__(self) -> None:
        self.judge = Judge()

    def score(self, res: ItemResult) -> list[Score]:
        out: list[Score] = []
        for fn in _JUDGED:
            try:
                score = fn(self.judge, res)
            except Exception as e:  # noqa: BLE001 -- one bad judge shouldn't sink the item
                score = Score(name=fn.__name__, value=0.0, reason=f"judge error: {e}")
            if score is not None:
                out.append(score)
        for det in _DETERMINISTIC:
            score = det(res)
            if score is not None:
                out.append(score)
        return out