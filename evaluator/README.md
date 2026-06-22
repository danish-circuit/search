# Evaluator

A small **Dagster** benchmarking harness that scores the search agent with
**LLM-as-judge** metrics, and ships every run to **Opik** for inspection. It is
a deliberately lightweight, classroom-sized version of a production eval
pipeline.

## What it does

Two Dagster jobs:

1. **`index`** — pull a tiny subset of the
   [ViDoRe v3](https://huggingface.co/datasets/vidore) *industrial* benchmark,
   turn each relevant corpus page into a one-page PDF, and `POST` it to the
   search agent's `/ingest` endpoint. Each page becomes one "document" so the
   answers are genuinely retrievable from what we indexed (≤10 documents).
2. **`evaluate`** — ask each benchmark question through the agent (`/ask`),
   then run the metric set on the answer + the retrieved context, and post the
   scores back onto the agent's Opik trace.

## The metrics

RAGAS-style multi-step LLM judges (each judge is one Claude Sonnet call;
prompts in [`config/metrics.yaml`](config/metrics.yaml), ported from
[ragas](https://github.com/explodinggradients/ragas)):

| Metric | Measures |
|---|---|
| **Accuracy** | Single-shot: does the answer contain every ground-truth claim? |
| **AnswerCompleteness** | Recall of ground-truth facts: `TP / (TP + FN)` |
| **AnswerCorrectness** | F1 over facts: penalises missing *and* unsupported facts |
| **Faithfulness** | Fraction of answer claims supported by the retrieved context |
| **ContextPrecision** | Were the retrieved chunks useful? (rank-weighted) |
| **ContextRecall** | Fraction of ground-truth sentences attributable to context |

Plus deterministic retrieval metrics over the golden documents (no LLM):
**DocumentRecall**, **GoldHit@5**, **GoldMRR**.

## Architecture

```
  evaluator        (Dagster UI, http://localhost:3001)
  evaluator-code   (gRPC code location :4000 — serves the job graph)
  evaluator-daemon (run queue / schedules)
        │
        ├── HTTP ──▶ api  (POST /ingest, POST /ask)
        └── REST ──▶ Opik (datasets, experiments, traces, feedback scores)
```

All three Dagster services run from one image (three build targets), wired
into the repo's top-level `docker-compose.yml`. Run history lives in a
separate `dagster` database inside the same Postgres container as the app.

## Running it

From the repo root:

```bash
# 1. Start Opik (clones comet-ml/opik into ./opik on first run; UI at :5173)
make opik

# 2. Bring up the whole stack (app + evaluator)
docker compose up --build

# 3. Open the Dagster UI and launch the jobs (index first, then evaluate)
open http://localhost:3001
```

Or launch the jobs from the CLI:

```bash
docker compose exec evaluator-code dagster job execute -m evaluator.app -j index
docker compose exec evaluator-code dagster job execute -m evaluator.app -j evaluate
```

Aggregate metrics are printed at the end of the `evaluate` run; per-item scores
and the agent's tool spans are visible in Opik at http://localhost:5173.

## Configuration

Environment variables (see `evaluator/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `API_URL` | `http://api:8000` | The search-agent backend |
| `ANTHROPIC_API_KEY` | — | Key for the judge model |
| `JUDGE_MODEL` | `claude-sonnet-4-6` | LLM used by the judges |
| `VIDORE_DATASET` | `vidore/vidore_v3_industrial` | ViDoRe v3 domain |
| `MAX_DOCUMENTS` | `10` | Cap on indexed documents |
| `MAX_QUESTIONS` | `15` | Cap on evaluated questions |
| `OPIK_URL_OVERRIDE` | (set by compose) | Local Opik; unset to disable tracing |

You can override the document/question caps per run via Dagster run config:

```yaml
ops:
  setup_evaluate:
    config:
      max_questions: 5
```