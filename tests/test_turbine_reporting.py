import csv
import json
import math
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from Helpers.stage_solver import StageDesign
from Helpers.turbine_reporting import StageReport, build_stage_reports
from Helpers.turbine_solver import TurbineDesign, TurbineSolution, print_turbine_report, solve_turbine
from Pipeline.Meanline.solve_meanline import Inputs, plot_velocity_triangles
from Runner.turbine_reports import CSV_FIELDS, RUNS_DIRECTORY, write_turbine_reports


class TurbineReportingTests(unittest.TestCase):
    solution: TurbineSolution
    stage_reports: tuple[StageReport, ...]

    @classmethod
    def setUpClass(cls) -> None:
        stage = StageDesign(
            inputs=Inputs(
                degree_of_reaction=0.4,
                N_shaft=11000,
                radius_2=0.4339,
                radius_3=0.4339,
                alpha_2=65.0,
                beta_3=-60.0,
                loss_kinetic_stator=0.08,
                loss_kinetic_rotor=0.1,
            )
        )
        cls.solution = solve_turbine(
            TurbineDesign(stages=(stage, stage), expansion_shares=(0.5, 0.5))
        )
        cls.stage_reports = build_stage_reports(cls.solution.stages)

    def test_station_records_are_stage_qualified_and_close_velocity_triangles(self) -> None:
        self.assertEqual(
            tuple(report.stage_index for report in self.stage_reports),
            (1, 2),
        )

        for stage_report in self.stage_reports:
            stage = self.solution.stages[stage_report.stage_index - 1]
            meanline = stage.final.meanline
            inputs = stage.final.inputs
            station_2, station_3 = stage_report.stations
            self.assertEqual(station_2.station_id, f"({stage_report.stage_index}, 2)")
            self.assertEqual(station_2.location, "stator_exit_rotor_inlet")
            self.assertEqual(station_3.station_id, f"({stage_report.stage_index}, 3)")
            self.assertEqual(station_3.location, "rotor_exit")

            for station in stage_report.stations:
                absolute = station.absolute_velocity
                relative = station.relative_velocity
                with self.subTest(station=station.station_id):
                    self.assertAlmostEqual(
                        absolute.speed_m_s**2,
                        absolute.axial_velocity_m_s**2 + absolute.tangential_velocity_m_s**2,
                    )
                    self.assertAlmostEqual(
                        relative.speed_m_s**2,
                        relative.axial_velocity_m_s**2 + relative.tangential_velocity_m_s**2,
                    )
                    self.assertAlmostEqual(
                        station.annulus.tip_radius_m - station.annulus.hub_radius_m,
                        station.annulus.blade_height_m,
                    )

            self.assertAlmostEqual(station_2.absolute_velocity.mach, meanline.M_2)
            self.assertAlmostEqual(station_2.relative_velocity.mach, meanline.M_2rel)
            self.assertAlmostEqual(station_3.absolute_velocity.mach, meanline.M_3)
            self.assertAlmostEqual(station_3.relative_velocity.mach, meanline.M_3rel)
            self.assertAlmostEqual(
                station_2.annulus.area_m2,
                2 * math.pi * inputs.radius_2 * meanline.blade_height_2,
            )
            self.assertAlmostEqual(station_3.annulus.area_m2, meanline.area_3)
            self.assertAlmostEqual(station_2.annulus.hub_diameter_m, meanline.D_hub_2)
            self.assertAlmostEqual(station_2.annulus.tip_diameter_m, meanline.D_tip_2)
            self.assertAlmostEqual(station_3.annulus.hub_diameter_m, meanline.D_hub_3)
            self.assertAlmostEqual(station_3.annulus.tip_diameter_m, meanline.D_tip_3)

    def test_single_stage_report_uses_stage_one_station_ids(self) -> None:
        single_stage_reports = build_stage_reports((self.solution.stages[0],))

        self.assertEqual(
            tuple(station.station_id for station in single_stage_reports[0].stations),
            ("(1, 2)", "(1, 3)"),
        )

    def test_console_report_identifies_stations_units_and_frames(self) -> None:
        report = StringIO()

        with patch("sys.stdout", report):
            print_turbine_report(self.solution)

        output = report.getvalue()
        self.assertIn("Stage 1 of 2", output)
        self.assertIn("Stage 2 of 2", output)
        self.assertIn("(1, 2)", output)
        self.assertIn("(2, 3)", output)
        self.assertIn("Hub radius [m]", output)
        self.assertIn("C absolute; W rotor-relative", output)
        self.assertIn("M_C abs", output)
        self.assertIn("M_W rel", output)
        self.assertIn("M_2rel rotor-relative", output)

    def test_triangle_plot_title_includes_stage_label(self) -> None:
        meanline = self.solution.stages[1].final.meanline

        with patch("Pipeline.Meanline.solve_meanline.VelocityTriangle.plot_combined") as plot_combined, patch(
            "Pipeline.Meanline.solve_meanline.plt.show"
        ):
            plot_velocity_triangles(meanline, stage_label="Stage 2 of 2")

        self.assertIn("Stage 2 of 2", plot_combined.call_args.kwargs["title"])

    def test_json_and_csv_exports_are_nested_flattened_and_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            csv_path, json_path = write_turbine_reports(self.solution, output_directory)

            self.assertEqual(csv_path.name, "turbine_report.csv")
            self.assertEqual(json_path.name, "turbine_report.json")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["stages"]), 2)
            self.assertEqual(
                payload["velocity_sign_convention"]["axial_positive"],
                "downstream",
            )
            json_stations = [
                station
                for stage in payload["stages"]
                for station in stage["stations"]
            ]
            self.assertEqual(
                [station["station_id"] for station in json_stations],
                ["(1, 2)", "(1, 3)", "(2, 2)", "(2, 3)"],
            )
            self.assertIn("hub_radius_m", json_stations[0]["annulus"])
            self.assertIn("mach", json_stations[0]["absolute_velocity"])
            self.assertIn("mach", json_stations[0]["relative_velocity"])

            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames), CSV_FIELDS)
            self.assertEqual(
                [row["station_id"] for row in rows],
                ["(1, 2)", "(1, 3)", "(2, 2)", "(2, 3)"],
            )
            self.assertIn("absolute_axial_velocity_m_s", CSV_FIELDS)
            self.assertIn("relative_flow_angle_deg", CSV_FIELDS)
            self.assertIn("annulus_area_m2", CSV_FIELDS)
            self.assertTrue(math.isfinite(float(rows[0]["absolute_mach"])))

            csv_path.write_text("stale CSV\n", encoding="utf-8")
            json_path.write_text('{"stale": true}\n', encoding="utf-8")
            write_turbine_reports(self.solution, output_directory)

            refreshed_payload = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                refreshed_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(refreshed_payload["stages"]), 2)
            self.assertEqual(len(refreshed_rows), 4)

    def test_default_export_directory_is_repository_runs(self) -> None:
        self.assertEqual(RUNS_DIRECTORY, Path(__file__).resolve().parents[1] / "runs")


if __name__ == "__main__":
    unittest.main()
