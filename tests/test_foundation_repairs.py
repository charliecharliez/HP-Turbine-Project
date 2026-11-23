import math
import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from boundary_conditions import boundary_conditions
from Losses.losses_solver import AXIAL_INLET_PROFILE_LOSS, load_chart
from Meanline import gas_props


class FoundationRepairTests(unittest.TestCase):
    def test_all_required_ko_charts_load(self) -> None:
        chart_names = (
            "ko_fig01_yp_nozzle.csv",
            "ko_fig02_yp_impulse.csv",
            "ko_fig14_te_nozzle.csv",
            "ko_fig14_te_impulse.csv",
        )

        for chart_name in chart_names:
            with self.subTest(chart_name=chart_name):
                chart = load_chart(chart_name)
                self.assertGreater(chart.size, 0)
                self.assertTrue(np.isfinite(chart).all())

    def test_profile_loss_interpolates_at_chart_data_point(self) -> None:
        warnings: list[str] = []

        loss = AXIAL_INLET_PROFILE_LOSS.profile_loss(
            pitch_chord=1.03642,
            exit_angle=40.0,
            warnings=warnings,
        )

        self.assertAlmostEqual(loss, 0.0178378)
        self.assertEqual(warnings, [])

    def test_coolprop_pressures_are_in_pascals(self) -> None:
        expected_inlet_pressure = boundary_conditions.P_01 * 1e5
        expected_exit_pressure = (
            expected_inlet_pressure / boundary_conditions.pressure_ratio_TS
        )

        self.assertEqual(gas_props.P_IN, expected_inlet_pressure)
        self.assertAlmostEqual(gas_props.P_OUT, expected_exit_pressure)
        self.assertTrue(math.isfinite(gas_props.T_OUT_S))
        self.assertGreater(gas_props.T_OUT_S, 0.0)


if __name__ == "__main__":
    unittest.main()
