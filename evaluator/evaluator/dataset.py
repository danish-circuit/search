"""Build a tiny benchmark subset from ViDoRe v3.

ViDoRe v3 ships four configs per domain:
  - ``corpus``: one row per document *page* (``corpus_id``, ``doc_id``,
    ``page_number_in_doc``, ``markdown`` text, page ``image``).
  - ``queries``: ``query_id``, ``query``, and a ground-truth ``answer``.
  - ``qrels``: which corpus pages (``corpus_id``) are relevant to each query.
  - ``documents_metadata``: per-document provenance.

Real ViDoRe documents are huge (industrial manuals, hundreds of pages), so for
a classroom demo we treat *each relevant page as its own one-page document*.
We:
  1. score each query by how many of its golden pages we can cover,
  2. greedily pick pages until we hit ``max_documents``,
  3. keep only the questions whose golden pages are ALL within that set (so the
     answer is genuinely retrievable from what we indexed),
  4. cap the questions at ``max_questions``.
"""

from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset  # pyright: ignore[reportMissingImports]

from evaluator.config import settings
from evaluator.types import Benchmark, BenchmarkItem, GoldenPage


def _load_split(config: str):
    return load_dataset(settings.vidore_dataset, config, split="test")



def build_benchmark(
    *,
    max_documents: int | None = None,
    max_questions: int | None = None,
) -> Benchmark:
    """Materialise the benchmark subset (documents to index + questions)."""
    max_documents = max_documents or settings.max_documents
    max_questions = max_questions or settings.max_questions

    corpus = _load_split("corpus")
    queries = _load_split("queries")
    qrels = _load_split("qrels")

    # corpus_id -> GoldenPage (built lazily; corpus is large).
    corpus_by_id: dict[int, dict] = {}
    for row in corpus:
        corpus_by_id[int(row["corpus_id"])] = row

    # query_id -> set of golden corpus_ids.
    golden_by_query: dict[int, set[int]] = defaultdict(set)
    for row in qrels:
        if int(row["score"]) > 0:
            golden_by_query[int(row["query_id"])].add(int(row["corpus_id"]))

    # query_id -> question/answer.
    query_rows = {int(q["query_id"]): q for q in queries}

    # Prefer questions with FEW golden pages -- they're cheaper to cover, so we
    # fit more complete questions inside the document budget.
    candidates = sorted(
        (
            qid
            for qid, golden in golden_by_query.items()
            if qid in query_rows and golden and golden.issubset(corpus_by_id.keys())
        ),
        key=lambda qid: len(golden_by_query[qid]),
    )

    selected_pages: dict[int, GoldenPage] = {}
    items: list[BenchmarkItem] = []
    for qid in candidates:
        if len(items) >= max_questions:
            break
        golden = golden_by_query[qid]
        # Would adding this question's golden pages exceed the doc budget?
        new_pages = golden - selected_pages.keys()
        if len(selected_pages) + len(new_pages) > max_documents:
            continue
        for cid in golden:
            if cid not in selected_pages:
                row = corpus_by_id[cid]
                selected_pages[cid] = GoldenPage(
                    corpus_id=cid,
                    doc_id=str(row["doc_id"]),
                    page_number=int(row["page_number_in_doc"]),
                    markdown=str(row["markdown"] or ""),
                )
        q = query_rows[qid]
        items.append(
            BenchmarkItem(
                query_id=qid,
                question=str(q["query"]),
                answer=str(q["answer"]),
                golden_docs=[selected_pages[cid].pdf_name for cid in golden],
            )
        )

    pages = list(selected_pages.values())
    return Benchmark(name=settings.vidore_dataset, pages=pages, items=items)