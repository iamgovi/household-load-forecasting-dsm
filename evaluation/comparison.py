"""Utilities that enforce identical forecast populations across models."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from evaluation.contract import DEPLOYMENT_NON_RESEARCH, HORIZON

COMMON_KEY = ["city", "meter", "split", "forecast_origin"]


def common_origin_predictions(predictions: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = predictions[predictions.model.isin(models)].copy()
    completeness = subset.groupby(["model", *COMMON_KEY]).horizon.agg(
        rows="size", unique_horizons="nunique"
    ).reset_index()
    complete = completeness[(completeness.rows == HORIZON) & (completeness.unique_horizons == HORIZON)]
    origin_models = complete.groupby(COMMON_KEY).model.nunique()
    common_index = origin_models[origin_models == len(set(models))].index
    keys = pd.DataFrame(list(common_index), columns=COMMON_KEY)
    common = subset.merge(keys, on=COMMON_KEY, how="inner")
    report = completeness.assign(
        complete_forecast=(completeness.rows == HORIZON) & (completeness.unique_horizons == HORIZON)
    )
    return common, report


def discover_research_prediction_files(report_dir: Path) -> list[Path]:
    """Only manifests explicitly marked research-eligible are discoverable."""
    discovered = []
    for manifest_path in report_dir.glob("*_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("research_eligible") is True:
            prediction_path = Path(manifest["prediction_path"])
            if prediction_path.name not in DEPLOYMENT_NON_RESEARCH:
                discovered.append(prediction_path)
    return discovered
