import json
from pathlib import Path

from design_models import JsonObject, RunRecord


def run_record_to_json(record: RunRecord) -> JsonObject:
    return {
        "run_id": record.run_id,
        "schema_version": record.schema_version,
        "input_snapshot": record.input_snapshot,
        "assumptions": list(record.assumptions),
        "warnings": list(record.warnings),
        "status": record.status.value,
        "converged": record.converged,
        "result": record.result,
        "error": record.error,
    }


def write_run_record(path: Path, record: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        run_record_to_json(record),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{serialized}\n", encoding="utf-8")
