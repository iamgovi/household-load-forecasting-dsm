"""Safe metadata adapter for existing Review-3 forecasting implementations.

This module describes legacy implementations without importing or executing
their training scripts. Training remains an explicit, separate operation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyImplementation:
    """Location and capabilities of a legacy implementation."""

    module: str
    entry_point: str
    writes_review3_outputs: bool = True
    training_required: bool = True
    safe_to_invoke_automatically: bool = False


LEGACY_IMPLEMENTATIONS = {
    "seasonal_naive": LegacyImplementation(
        "review3.individual_models", "run_seasonal_naive"
    ),
    "holt_winters": LegacyImplementation(
        "review3.individual_models", "run_statistical"
    ),
    "arima": LegacyImplementation(
        "review3.individual_models", "run_statistical"
    ),
    "sarima": LegacyImplementation(
        "review3.individual_models", "attempt_sarima"
    ),
    "linear_regression": LegacyImplementation(
        "review3.individual_models", "run_linear"
    ),
    "xgboost": LegacyImplementation(
        "review3.individual_models", "run_xgboost"
    ),
    "gru": LegacyImplementation(
        "review3.individual_models", "run_gru"
    ),
    "arima_gru": LegacyImplementation(
        "review3.arima_gru_hybrid", "main"
    ),
    "pso_arima_gru": LegacyImplementation(
        "review3.arima_gru_pso", "main"
    ),
    "arima_gru_attention_ablation": LegacyImplementation(
        "review3.arima_gru_attention", "main"
    ),
}


def get_legacy_implementation(model_name: str) -> LegacyImplementation:
    """Return legacy metadata without importing or running model code."""
    try:
        return LEGACY_IMPLEMENTATIONS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"No legacy implementation registered for {model_name!r}"
        ) from exc


def list_legacy_implementations() -> dict[str, LegacyImplementation]:
    """Return a copy of the legacy implementation mapping."""
    return dict(LEGACY_IMPLEMENTATIONS)
