"""The evaluate job: ask each benchmark question and judge the agent's answer.

Two assets:
  - ``setup_evaluate``: rebuild the benchmark subset (same selection as the
    index job) and open an Opik experiment for this run.
  - ``run_evaluate``: for each question, call the agent (/ask), run the LLM
    judges + deterministic retrieval metrics, and post the scores to Opik.
"""


from statistics import mean

from dagster import AssetExecutionContext, Config, asset

from evaluator import client
from evaluator.dataset import build_benchmark
from evaluator.evaluator import Evaluator
from evaluator.opik_client import OpikClient
from evaluator.types import Benchmark, ItemResult


class EvaluateConfig(Config):
    max_documents: int = 10
    max_questions: int = 15



@asset
def setup_evaluate(context: AssetExecutionContext, config: EvaluateConfig) -> Benchmark:
    benchmark = build_benchmark(
        max_documents=config.max_documents,
        max_questions=config.max_questions,
    )
    context.log.info("Evaluating %d questions", len(benchmark.items))
    return benchmark



@asset
def run_evaluate(
    context: AssetExecutionContext, setup_evaluate: Benchmark
) -> dict:
    """Ask every question, judge the answers, and report aggregate metrics."""
    if not client.healthz():
        raise RuntimeError("search-agent backend is not reachable; is the `api` service up?")

    opik = OpikClient()
    if opik.enabled:
        opik.sync_dataset("search-agent-bench", setup_evaluate.items)
        opik.new_experiment("search-agent-bench")

    evaluator = Evaluator()
    results: list[ItemResult] = []
    for i, item in enumerate(setup_evaluate.items):
        context.log.info("[%d/%d] %s", i + 1, len(setup_evaluate.items), item.question[:80])
        try:
            resp = client.ask(item.question)
        except Exception as e:  # noqa: BLE001
            context.log.error("ask failed for q%d: %s", item.query_id, e)
            results.append(ItemResult(item=item, answer="", error=str(e)))
            continue

        res = ItemResult(
            item=item,
            answer=resp.get("answer", ""),
            trace_id=resp.get("trace_id"),
            retrieved_titles=resp.get("retrieved_titles", []),
            contexts=resp.get("contexts", []),
        )
        res.scores = evaluator.score(res)
        results.append(res)
        for s in res.scores:
            context.log.info("    %s = %.3f (%s)", s.name, s.value, s.reason)
        if opik.enabled and res.trace_id:
            opik.post_scores(res.trace_id, res.scores)

    if opik.enabled:
        opik.flush()

    summary = _aggregate(results)
    context.log.info("=== aggregate metrics ===")
    for name, value in summary.items():
        context.log.info("%s: %.3f", name, value)
    return {
        "n_items": len(results),
        "n_errors": sum(1 for r in results if r.error),
        "metrics": summary,
    }



def _aggregate(results: list[ItemResult]) -> dict[str, float]:
    """Mean of each metric across all scored items."""
    buckets: dict[str, list[float]] = {}
    for r in results:
        for s in r.scores:
            buckets.setdefault(s.name, []).append(s.value)
    return {name: mean(vals) for name, vals in buckets.items() if vals}