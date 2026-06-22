"""Dagster definitions for the evaluator.

Two jobs:
  - ``index``: setup_index -> index_documents (pull ViDoRe v3 PDFs and ingest).
  - ``evaluate``: setup_evaluate -> run_evaluate (ask questions, judge answers).
"""

from dagster import AssetSelection, Definitions, define_asset_job

from evaluator.evaluate import run_evaluate, setup_evaluate
from evaluator.index import index_documents, setup_index

index_job = define_asset_job(
    name="index",
    selection=AssetSelection.assets(setup_index, index_documents),
)

evaluate_job = define_asset_job(
    name="evaluate",
    selection=AssetSelection.assets(setup_evaluate, run_evaluate),
)

defs = Definitions(
    assets=[setup_index, index_documents, setup_evaluate, run_evaluate],
    jobs=[index_job, evaluate_job],
)