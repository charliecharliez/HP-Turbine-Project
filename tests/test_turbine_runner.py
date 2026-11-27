import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from Helpers.turbine_solver import TurbineDesign
from Runner.main import (
    SELECTED_DESIGN,
    default_output_directory,
    load_turbine_design,
    main,
)


class TurbineRunnerTests(unittest.TestCase):
    def test_config_selects_stage_count_before_solving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "two-stage.json"
            config_path.write_text(
                json.dumps(
                    {
                        "stage_count": 2,
                        "expansion_shares": [0.4, 0.6],
                        "stages": [
                            {"degree_of_reaction": 0.35},
                            {"degree_of_reaction": 0.45},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            design = load_turbine_design(config_path)

        self.assertIsInstance(design, TurbineDesign)
        self.assertEqual(len(design.stages), 2)
        self.assertEqual(design.expansion_shares, (0.4, 0.6))
        self.assertEqual(
            tuple(stage.inputs.degree_of_reaction for stage in design.stages),
            (0.35, 0.45),
        )

    def test_config_requires_one_or_two_matching_stage_entries(self) -> None:
        configurations = (
            {"stage_count": True, "stages": [{}]},
            {"stage_count": 0, "stages": []},
            {"stage_count": 3, "stages": [{}, {}, {}]},
            {"stage_count": 1.0, "stages": [{}]},
            {"stage_count": 2, "stages": [{}]},
        )
        for configuration in configurations:
            with self.subTest(configuration=configuration), tempfile.TemporaryDirectory() as temporary_directory:
                config_path = Path(temporary_directory) / "invalid.json"
                config_path.write_text(json.dumps(configuration), encoding="utf-8")

                with self.assertRaises(ValueError):
                    load_turbine_design(config_path)

    def test_config_rejects_unknown_fields_and_invalid_expansion_shares(self) -> None:
        configurations = (
            {"stage_count": 1, "stages": [{}], "unexpected": 1},
            {"stage_count": 1, "stages": [{"unexpected": 1}]},
            {"stage_count": 1, "stages": [{}], "expansion_shares": [0.9]},
        )
        for configuration in configurations:
            with self.subTest(configuration=configuration), tempfile.TemporaryDirectory() as temporary_directory:
                config_path = Path(temporary_directory) / "invalid.json"
                config_path.write_text(json.dumps(configuration), encoding="utf-8")

                with self.assertRaises(ValueError):
                    load_turbine_design(config_path)

    def test_default_outputs_are_separate_for_each_stage_count(self) -> None:
        one_stage = default_output_directory(1)
        two_stage = default_output_directory(2)

        self.assertEqual(one_stage.name, "one-stage")
        self.assertEqual(two_stage.name, "two-stage")
        self.assertNotEqual(one_stage, two_stage)

    def test_cli_writes_selected_design_to_its_stage_count_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "two-stage.json"
            config_path.write_text(
                json.dumps({"stage_count": 2, "stages": [{}, {}]}),
                encoding="utf-8",
            )
            stage = SimpleNamespace(final=SimpleNamespace(meanline=object()))
            solution = SimpleNamespace(feasible=True, stages=(stage, stage))

            with patch("Runner.main.solve_turbine", return_value=solution) as solve, patch(
                "Runner.main.write_turbine_reports", return_value=(Path("report.csv"), Path("report.json"))
            ) as write_reports, patch("Runner.main.print_turbine_report"), patch(
                "Runner.main.plot_velocity_triangles"
            ):
                exit_code = main(["--config", str(config_path)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(solve.call_args.args[0].stages), 2)
        self.assertEqual(write_reports.call_args.args[1], default_output_directory(2))
        self.assertIs(write_reports.call_args.args[0], solution)

    def test_cli_writes_infeasible_results_before_returning_failure(self) -> None:
        stage = SimpleNamespace(final=SimpleNamespace(meanline=object()))
        solution = SimpleNamespace(feasible=False, stages=(stage,))

        with patch("Runner.main.solve_turbine", return_value=solution), patch(
            "Runner.main.write_turbine_reports", return_value=(Path("report.csv"), Path("report.json"))
        ) as write_reports, patch("Runner.main.print_turbine_report"):
            exit_code = main([])

        self.assertEqual(exit_code, 2)
        write_reports.assert_called_once()

    def test_missing_config_uses_existing_default_design(self) -> None:
        self.assertIs(load_turbine_design(None), SELECTED_DESIGN)


if __name__ == "__main__":
    unittest.main()
