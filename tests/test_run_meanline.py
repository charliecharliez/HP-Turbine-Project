import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from design_models import RunStatus
from run_meanline import load_meanline_inputs, run_meanline


class MeanlineRunTests(unittest.TestCase):
    def test_config_uses_explicit_unit_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "meanline.json"
            config_path.write_text(
                json.dumps({"shaft_speed_rpm": 12_500, "radius_2_m": 0.45}),
                encoding="utf-8",
            )

            inputs = load_meanline_inputs(config_path)

        self.assertEqual(inputs.N_shaft, 12_500)
        self.assertEqual(inputs.radius_2, 0.45)

    def test_config_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "meanline.json"
            config_path.write_text(json.dumps({"speed": 12_500}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown meanline configuration keys"):
                load_meanline_inputs(config_path)

    def test_run_writes_si_inputs_and_solver_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "run.json"
            record = run_meanline(config_path=None, output_path=output_path)
            saved_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(record.status, RunStatus.SUCCEEDED)
        self.assertEqual(saved_record["status"], RunStatus.SUCCEEDED.value)
        self.assertEqual(
            saved_record["input_snapshot"]["boundary_conditions"]["inlet_total_pressure_pa"],
            2_000_000,
        )
        self.assertIn("euler_specific_work_j_kg", saved_record["result"])
        self.assertTrue(saved_record["converged"])

    def test_failed_solver_writes_failure_record_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "failed-run.json"
            with patch("run_meanline.solve_meanline", side_effect=ValueError("infeasible case")):
                with self.assertRaisesRegex(ValueError, "infeasible case"):
                    run_meanline(config_path=None, output_path=output_path)
            saved_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_record["status"], RunStatus.FAILED.value)
        self.assertEqual(saved_record["error"], "infeasible case")
        self.assertIsNone(saved_record["result"])


if __name__ == "__main__":
    unittest.main()
