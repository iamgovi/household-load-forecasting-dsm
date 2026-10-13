
"""Isolated run metadata for forecasting preparation and experiments."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path


SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def _validate_safe_name(value: str, label: str) -> str:
    """Validate a model name or run ID for safe directory use."""
    if not isinstance(value, str) or not SAFE_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must contain only safe filename characters"
        )
    if value in {".", ".."}:
        raise ValueError(f"{label} is not a valid name")
    return value


def create_run_directory(root, model_name, run_id):
    """Create an isolated model/run directory without overwriting existing runs."""
    model_name = _validate_safe_name(model_name, "model_name")
    run_id = _validate_safe_name(run_id, "run_id")

    base = (
        Path(root).resolve()
        / "reports"
        / "final_forecasting_preparation"
        / "runs"
    )
    destination = base / model_name / run_id

    if not destination.resolve().is_relative_to(base.resolve()):
        raise ValueError(
            "Run directory must remain inside the preparation runs directory"
        )

    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_run_manifest(
    run_directory,
    model_name,
    run_id,
    config,
    seed=42,
):
    """Write preparation metadata without training or evaluating a model."""
    model_name = _validate_safe_name(model_name, "model_name")
    run_id = _validate_safe_name(run_id, "run_id")

    directory = Path(run_directory).resolve()

    if not directory.is_dir():
        raise ValueError("Run directory must already exist")

    if directory.name != run_id or directory.parent.name != model_name:
        raise ValueError(
            "Run directory must match the supplied model_name and run_id"
        )

    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    manifest = {
        "model": model_name,
        "run_id": run_id,
        "status": "prepared_not_executed",
        "seed": seed,
        "config": config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_executed": False,
        "final_test_accessed": False,
    }

    target = directory / "manifest.json"

    if target.exists():
        raise FileExistsError(f"Manifest already exists: {target}")

    temporary = directory / "manifest.json.tmp"

    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()

    return target
