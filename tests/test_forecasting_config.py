
"""Tests for shared forecasting configuration."""

from evaluation import contract
from forecasting import config


def test_forecasting_config_matches_evaluation_contract():
    assert config.LOOKBACK == contract.LOOKBACK == 168
    assert config.HORIZON == contract.HORIZON == 24
    assert config.FREQUENCY == contract.FREQUENCY == "1h"
    assert config.ORIGIN_STRIDE_HOURS == contract.ORIGIN_STRIDE_HOURS == 24
    assert config.SEED == contract.SEED == 42
    assert config.PREDICTION_COLUMNS == contract.PREDICTION_COLUMNS


def test_forecasting_config_exposes_separate_data_splits():
    assert config.TRAIN_PATH == contract.TRAIN_PATH
    assert config.VALIDATION_PATH == contract.VALIDATION_PATH
    assert config.TEST_PATH == contract.TEST_PATH
