import json

import pytest

from forecasting.run_metadata import (
    create_run_directory,
    write_run_manifest,
)


def test_run_directory_is_isolated(tmp_path):
    path = create_run_directory(tmp_path, "gru", "test-run-001")
    assert path.name == "test-run-001"
    assert "final_forecasting_preparation" in str(path)
    assert path.is_dir()


def test_unsafe_run_id_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="safe filename"):
        create_run_directory(tmp_path, "gru", "../review3")


def test_existing_run_directory_is_not_overwritten(tmp_path):
    create_run_directory(tmp_path, "gru", "duplicate-run")
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "gru", "duplicate-run")


def test_manifest_records_non_execution(tmp_path):
    path = create_run_directory(tmp_path, "gru", "metadata-test")
    manifest_path = write_run_manifest(path, "gru", "metadata-test", {"lookback": 168})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["training_executed"] is False
    assert manifest["final_test_accessed"] is False
    assert manifest["status"] == "prepared_not_executed"


def test_manifest_does_not_overwrite(tmp_path):
    path = create_run_directory(tmp_path, "gru", "manifest-test")
    write_run_manifest(path, "gru", "manifest-test", {})
    with pytest.raises(FileExistsError):
        write_run_manifest(path, "gru", "manifest-test", {})
