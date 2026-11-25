import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from boundary_conditions import boundary_conditions
from design_models import RunStatus
from hardware_limits import check_rotor_hardware_limits
from Meanline.solve_meanline import Inputs, solve_meanline
from run_meanline import load_meanline_inputs, main, run_meanline


class HardwareLimitTests(unittest.TestCase):
    def test_limit_boundaries_and_each_violation(self) -> None:
        cases = (
            (500.0, 500.0, 0.32, True, (True, True, True)),
            (500.1, 500.0, 0.32, False, (False, True, True)),
            (500.0, 500.1, 0.32, False, (True, False, True)),
            (500.0, 500.0, 0.3201, False, (True, True, False)),
            (501.0, 501.0, 0.33, False, (False, False, False)),
        )
        for inlet_speed, exit_speed, area, passed, check_results in cases:
            with self.subTest(inlet_speed=inlet_speed, exit_speed=exit_speed, area=area):
                result = check_rotor_hardware_limits(
                    stage_index=2,
                    shaft_speed_rpm=10_000,
                    inlet_blade_speed_m_s=inlet_speed,
                    exit_blade_speed_m_s=exit_speed,
                    exit_annulus_area_m2=area,
                    boundary=boundary_conditions.as_si(),
                )
                self.assertEqual(result["passed"], passed)
                checks = result["checks"]
                self.assertEqual(tuple(check["passed"] for check in checks), check_results)
                self.assertTrue(all(check["stage_index"] == 2 for check in checks))
                self.assertEqual(checks[2]["value"], area * 10_000**2)
                self.assertEqual(checks[2]["unit"], "m^2 rpm^2")

    def test_invalid_result_is_not_treated_as_passing(self) -> None:
        for area in (math.nan, math.inf, 0.0, -1.0):
            with self.subTest(area=area):
                with self.assertRaisesRegex(ValueError, "exit_annulus_area_m2"):
                    check_rotor_hardware_limits(
                        stage_index=1,
                        shaft_speed_rpm=10_000,
                        inlet_blade_speed_m_s=400,
                        exit_blade_speed_m_s=400,
                        exit_annulus_area_m2=area,
                        boundary=boundary_conditions.as_si(),
                    )

    def test_overflowed_an2_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rotor exit AN2 must be finite"):
            check_rotor_hardware_limits(
                stage_index=1,
                shaft_speed_rpm=1e200,
                inlet_blade_speed_m_s=400,
                exit_blade_speed_m_s=400,
                exit_annulus_area_m2=1,
                boundary=boundary_conditions.as_si(),
            )


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
        hardware_limits = saved_record["result"]["hardware_limits"]
        self.assertFalse(hardware_limits["passed"])
        self.assertEqual(
            [check["passed"] for check in hardware_limits["checks"]],
            [False, False, True],
        )
        self.assertAlmostEqual(
            hardware_limits["checks"][2]["value"],
            saved_record["result"]["rotor_exit_annulus_area_m2"] * 13_000**2,
        )

    def test_cli_rejects_default_but_saves_converged_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "default.json"
            exit_code = main(["--output", str(output_path)])
            saved_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertNotEqual(exit_code, 0)
        self.assertEqual(saved_record["status"], RunStatus.SUCCEEDED.value)
        self.assertTrue(saved_record["converged"])
        self.assertFalse(saved_record["result"]["hardware_limits"]["passed"])

    def test_cli_accepts_hardware_limit_passing_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            output_path = Path(temporary_directory) / "passing.json"
            config_path.write_text(
                json.dumps(
                    {"shaft_speed_rpm": 8000, "radius_2_m": 0.35, "radius_3_m": 0.4}
                ),
                encoding="utf-8",
            )
            exit_code = main(["--config", str(config_path), "--output", str(output_path)])
            saved_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(saved_record["status"], RunStatus.SUCCEEDED.value)
        self.assertTrue(saved_record["result"]["hardware_limits"]["passed"])

    def test_nonconverged_run_does_not_claim_hardware_feasibility(self) -> None:
        result = replace(solve_meanline(Inputs()), residual=1.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nonconverged.json"
            with patch("run_meanline.solve_meanline", return_value=result):
                exit_code = main(["--output", str(output_path)])
            saved_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertNotEqual(exit_code, 0)
        self.assertEqual(saved_record["status"], RunStatus.NON_CONVERGED.value)
        self.assertIsNone(saved_record["result"]["hardware_limits"])

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
