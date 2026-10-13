import pytest

from forecasting.legacy import (
    LEGACY_IMPLEMENTATIONS,
    get_legacy_implementation,
    list_legacy_implementations,
)


def test_legacy_entries_match_existing_runner_locations():
    assert get_legacy_implementation("gru").module == (
        "review3.individual_models"
    )
    assert get_legacy_implementation("arima_gru").module == (
        "review3.arima_gru_hybrid"
    )
    assert get_legacy_implementation("pso_arima_gru").module == (
        "review3.arima_gru_pso"
    )


def test_legacy_implementations_are_not_safe_to_invoke_automatically():
    assert LEGACY_IMPLEMENTATIONS
    assert all(
        not spec.safe_to_invoke_automatically
        for spec in LEGACY_IMPLEMENTATIONS.values()
    )


def test_listing_returns_a_copy():
    listed = list_legacy_implementations()
    listed.clear()
    assert list_legacy_implementations()


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="No legacy implementation"):
        get_legacy_implementation("unknown_model")
