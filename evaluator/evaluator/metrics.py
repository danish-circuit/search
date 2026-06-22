"""The metric set, computed per benchmark item.

Two families:
  - Judged (LLM) metrics built on the RAGAS pipelines:
      Accuracy, AnswerCompleteness, AnswerCorrectness, Faithfulness,
      ContextPrecision, ContextRecall.
  - Deterministic retrieval metrics over the golden documents:
      DocumentRecall, GoldHit@5, GoldMRR.

Each returns a :class:`Score` (name/value/reason) or ``None`` when it can't be
computed (e.g. no retrieved context). The evaluator collects the non-None ones.
"""

from __future__ import annotations

from evaluator import ragas
from evaluator.judges import Judge
from evaluator.types import IntScore, ItemResult, Score


# --------------------------------------------------------------------------- #
# Judged (LLM) metrics                                                          #
# --------------------------------------------------------------------------- #
def accuracy(judge: Judge, res: ItemResult) -> Score:
    """Single-shot judge: does the answer contain every ground-truth claim?"""
    result = judge.run(
        ragas.prompt("accuracy"),
        IntScore,
        question=res.item.question,
        expected_answer=res.item.answer,
        agent_answer=res.answer,
    )
    return Score(name="Accuracy", value=float(result.value), reason=result.reason)



def answer_completeness(judge: Judge, res: ItemResult) -> Score:
    """Recall over ground-truth statements: TP / (TP + FN)."""
    q = res.item.question
    ans_stmts = ragas.extract_statements(judge, question=q, answer=res.answer)
    gt_stmts = ragas.extract_statements(judge, question=q, answer=res.item.answer)
    if not gt_stmts:
        return Score(name="AnswerCompleteness", value=0.0, reason="no GT statements")
    if not ans_stmts:
        return Score(name="AnswerCompleteness", value=0.0, reason="no answer statements")
    c = ragas.classify_correctness(
        judge, question=q, answer_statements=ans_stmts, ground_truth_statements=gt_stmts
    )
    score = ragas.recall_from_classification(c)
    return Score(
        name="AnswerCompleteness",
        value=score,
        reason=f"TP={len(c.TP)} FN={len(c.FN)} -> recall={score:.3f}",
    )



def answer_correctness(judge: Judge, res: ItemResult) -> Score:
    """F1 over statements: penalises both missing and extra/unsupported facts."""
    q = res.item.question
    ans_stmts = ragas.extract_statements(judge, question=q, answer=res.answer)
    gt_stmts = ragas.extract_statements(judge, question=q, answer=res.item.answer)
    if not ans_stmts or not gt_stmts:
        return Score(name="AnswerCorrectness", value=0.0, reason="no statements on one side")
    c = ragas.classify_correctness(
        judge, question=q, answer_statements=ans_stmts, ground_truth_statements=gt_stmts
    )
    score = ragas.f1_from_classification(c)
    return Score(
        name="AnswerCorrectness",
        value=score,
        reason=f"TP={len(c.TP)} FP={len(c.FP)} FN={len(c.FN)} -> F1={score:.3f}",
    )



def faithfulness(judge: Judge, res: ItemResult) -> Score | None:
    """Fraction of answer claims supported by the retrieved context."""
    context = "\n\n---\n\n".join(res.contexts)
    if not context:
        return Score(name="Faithfulness", value=0.0, reason="no retrieved context")
    stmts = ragas.extract_statements(judge, question=res.item.question, answer=res.answer)
    if not stmts:
        return Score(name="Faithfulness", value=1.0, reason="answer has no factual claims")
    v = ragas.nli_verdicts(judge, context=context, statements=stmts)
    supported = sum(1 for s in v.statements if s.verdict)
    total = len(v.statements) or 1
    score = supported / total
    return Score(
        name="Faithfulness", value=score, reason=f"{supported}/{total} supported"
    )



def context_precision(judge: Judge, res: ItemResult) -> Score | None:
    """Rank-weighted precision: were the retrieved chunks useful for the answer?"""
    if not res.contexts:
        return Score(name="ContextPrecision", value=0.0, reason="no retrieved chunks")
    verdicts: list[int] = []
    for chunk in res.contexts:
        v = ragas.chunk_verdict(
            judge, question=res.item.question, context=chunk, answer=res.item.answer
        )
        verdicts.append(1 if v.verdict else 0)
    score = ragas.average_precision_at_k(verdicts)
    return Score(
        name="ContextPrecision",
        value=score,
        reason=f"{sum(verdicts)}/{len(verdicts)} useful (AP={score:.3f})",
    )



def context_recall(judge: Judge, res: ItemResult) -> Score | None:
    """Fraction of ground-truth sentences attributable to the retrieved context."""
    context = "\n\n---\n\n".join(res.contexts)
    if not context:
        return Score(name="ContextRecall", value=0.0, reason="no retrieved context")
    c = ragas.context_recall_classify(
        judge, question=res.item.question, context=context, answer=res.item.answer
    )
    score = ragas.attribution_score(c)
    attributed = sum(1 for s in c.classifications if s.attributed)
    return Score(
        name="ContextRecall",
        value=score,
        reason=f"{attributed}/{len(c.classifications)} GT sentences attributable",
    )



# --------------------------------------------------------------------------- #
# Deterministic retrieval metrics (golden documents)                            #
# --------------------------------------------------------------------------- #
def _retrieved_rank(res: ItemResult) -> int | None:
    """1-based rank of the first golden document among retrieved titles."""
    golden = {d.removesuffix(".pdf") for d in res.item.golden_docs}
    for rank, title in enumerate(res.retrieved_titles, start=1):
        if title.removesuffix(".pdf") in golden:
            return rank
    return None



def document_recall(res: ItemResult) -> Score:
    """Did the agent retrieve at least one golden document?"""
    rank = _retrieved_rank(res)
    hit = rank is not None
    return Score(
        name="DocumentRecall",
        value=1.0 if hit else 0.0,
        reason=f"golden doc at rank {rank}" if hit else "no golden doc retrieved",
    )



def gold_hit_at_5(res: ItemResult) -> Score:
    rank = _retrieved_rank(res)
    hit = rank is not None and rank <= 5
    return Score(
        name="GoldHit@5",
        value=1.0 if hit else 0.0,
        reason=f"rank {rank}" if rank else "not retrieved",
    )



def gold_mrr(res: ItemResult) -> Score:
    rank = _retrieved_rank(res)
    value = 1.0 / rank if rank else 0.0
    return Score(
        name="GoldMRR",
        value=value,
        reason=f"reciprocal rank of golden doc (rank {rank})" if rank else "not retrieved",
    )