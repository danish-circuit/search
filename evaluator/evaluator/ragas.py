"""RAGAS-style judge pipelines.

Each metric is broken into small, single-purpose LLM calls (extract claims,
then classify each claim) rather than asking the model for one holistic score.
That is both more reliable and easier to explain in a talk: you can see the
intermediate statements and verdicts.

These helpers are deliberately plain functions over a shared :class:`Judge`.
Prompts come from ``config/metrics.yaml``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from evaluator.judges import Judge
from evaluator.types import (
    AttributedStatementList,
    ChunkVerdict,
    StatementList,
    StatementVerdictList,
    TpFpFnClassification,
)

_METRICS_PATH = Path(__file__).resolve().parent.parent / "config" / "metrics.yaml"



@functools.lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    data = yaml.safe_load(_METRICS_PATH.read_text())
    return {k: v["template"] for k, v in data.items()}



def prompt(key: str) -> str:
    return _prompts()[key]



# --- statement extraction (shared by Completeness, Correctness, Faithfulness) -
def extract_statements(judge: Judge, *, question: str, answer: str) -> list[str]:
    if not answer.strip():
        return []
    result = judge.run(
        prompt("statement_extractor"),
        StatementList,
        question=question,
        answer=answer,
    )
    return [s for s in result.statements if s and s.strip()]



def classify_correctness(
    judge: Judge,
    *,
    question: str,
    answer_statements: list[str],
    ground_truth_statements: list[str],
) -> TpFpFnClassification:
    return judge.run(
        prompt("answer_correctness_classifier"),
        TpFpFnClassification,
        question=question,
        answer_statements=answer_statements,
        ground_truth_statements=ground_truth_statements,
    )



def f1_from_classification(c: TpFpFnClassification) -> float:
    """F1 = TP / (TP + 0.5 * (FP + FN))."""
    tp, fp, fn = len(c.TP), len(c.FP), len(c.FN)
    denom = tp + 0.5 * (fp + fn)
    return tp / denom if denom else 0.0



def recall_from_classification(c: TpFpFnClassification) -> float:
    """Completeness/recall = TP / (TP + FN)."""
    tp, fn = len(c.TP), len(c.FN)
    denom = tp + fn
    return tp / denom if denom else 0.0



def nli_verdicts(
    judge: Judge, *, context: str, statements: list[str]
) -> StatementVerdictList:
    return judge.run(
        prompt("faithfulness_nli"),
        StatementVerdictList,
        context=context,
        statements=statements,
    )



def chunk_verdict(
    judge: Judge, *, question: str, context: str, answer: str
) -> ChunkVerdict:
    return judge.run(
        prompt("context_precision_verdict"),
        ChunkVerdict,
        question=question,
        context=context,
        answer=answer,
    )



def average_precision_at_k(verdicts: list[int]) -> float:
    """RAGAS-style average precision over a ranked list of binary verdicts."""
    if not verdicts:
        return 0.0
    cumsum = 0
    numerator = 0.0
    for i, v in enumerate(verdicts):
        v = 1 if v else 0
        cumsum += v
        if v:
            numerator += cumsum / (i + 1)
    return numerator / cumsum if cumsum else 0.0



def context_recall_classify(
    judge: Judge, *, question: str, context: str, answer: str
) -> AttributedStatementList:
    return judge.run(
        prompt("context_recall_classification"),
        AttributedStatementList,
        question=question,
        context=context,
        answer=answer,
    )



def attribution_score(c: AttributedStatementList) -> float:
    if not c.classifications:
        return 0.0
    return sum(1 for s in c.classifications if s.attributed) / len(c.classifications)