"""The index job: pull a ViDoRe v3 subset and ingest it into the search agent.

Two assets:
  - ``setup_index``: build the benchmark subset (golden pages -> documents,
    questions -> dataset) and register the dataset in Opik.
  - ``index_documents``: render each golden page to a PDF and POST it to the
    backend's /ingest endpoint, so the agent can retrieve it.
"""


from dagster import AssetExecutionContext, Config, asset

from evaluator import client, pdfs
from evaluator.dataset import build_benchmark
from evaluator.opik_client import OpikClient
from evaluator.types import Benchmark


class IndexConfig(Config):
    max_documents: int = 10
    max_questions: int = 15



@asset
def setup_index(context: AssetExecutionContext, config: IndexConfig) -> Benchmark:
    """Build the benchmark subset and sync the question dataset to Opik."""
    benchmark = build_benchmark(
        max_documents=config.max_documents,
        max_questions=config.max_questions,
    )
    context.log.info(
        "Built benchmark: %d documents, %d questions from %s",
        len(benchmark.pages),
        len(benchmark.items),
        benchmark.name,
    )
    opik = OpikClient()
    if opik.enabled:
        opik.sync_dataset("search-agent-bench", benchmark.items)
        opik.flush()
        context.log.info("Synced %d questions to Opik dataset", len(benchmark.items))
    return benchmark



@asset
def index_documents(
    context: AssetExecutionContext, setup_index: Benchmark
) -> Benchmark:
    """Render each golden page to a PDF and ingest it into the search agent."""
    if not client.healthz():
        raise RuntimeError("search-agent backend is not reachable; is the `api` service up?")
    indexed = 0
    for page in setup_index.pages:
        pdf_bytes = pdfs.render_pdf(page)
        result = client.ingest_pdf(page, pdf_bytes)
        indexed += 1
        context.log.info(
            "Ingested %s -> document_id=%s (%s chunks)",
            page.pdf_name,
            result.get("document_id"),
            result.get("n_chunks"),
        )
    context.log.info("Indexed %d documents", indexed)
    return setup_index