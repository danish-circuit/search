"""Shared data types for the evaluator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenPage(BaseModel):
    """One ViDoRe corpus page selected as a "document" to index.

    Each golden page becomes a single-page PDF named ``{doc_id}_p{page}.pdf``;
    that filename is the unit retrieval metrics match against.
    """

    corpus_id: int
    doc_id: str
    page_number: int
    markdown: str

    @property
    def pdf_name(self) -> str:
        return f"{self.doc_id}_p{self.page_number}.pdf"


class BenchmarkItem(BaseModel):
    """One question + ground truth, with the golden documents that answer it."""

    query_id: int
    question: str
    answer: str
    # PDF filenames (``pdf_name`` above) that are the golden references.
    golden_docs: list[str] = Field(default_factory=list)


class Benchmark(BaseModel):
    """The materialised benchmark subset: documents to index + questions to ask."""

    name: str
    pages: list[GoldenPage]
    items: list[BenchmarkItem]


class Score(BaseModel):
    """A single metric result, mirroring an Opik feedback score."""

    name: str
    value: float
    reason: str = ""


class ItemResult(BaseModel):
    """Everything we computed for one benchmark item."""

    item: BenchmarkItem
    answer: str
    trace_id: str | None = None
    retrieved_titles: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    scores: list[Score] = Field(default_factory=list)
    error: str | None = None


# --- structured outputs for the RAGAS-style multi-step judges ---
class StatementList(BaseModel):
    statements: list[str]


class StatementWithReason(BaseModel):
    statement: str
    reason: str


class TpFpFnClassification(BaseModel):
    """Answer Correctness/Completeness: classify each statement as TP/FP/FN."""

    TP: list[StatementWithReason] = Field(default_factory=list)
    FP: list[StatementWithReason] = Field(default_factory=list)
    FN: list[StatementWithReason] = Field(default_factory=list)


class StatementVerdict(BaseModel):
    statement: str
    reason: str
    verdict: int  # 1 if supported by context, else 0


class StatementVerdictList(BaseModel):
    statements: list[StatementVerdict] = Field(default_factory=list)


class ChunkVerdict(BaseModel):
    reason: str
    verdict: int  # 1 if the chunk was useful, else 0


class AttributedStatement(BaseModel):
    statement: str
    reason: str
    attributed: int  # 1 if attributable to context, else 0


class AttributedStatementList(BaseModel):
    classifications: list[AttributedStatement] = Field(default_factory=list)


class IntScore(BaseModel):
    value: int
    reason: str = ""