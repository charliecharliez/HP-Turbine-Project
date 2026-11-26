import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from boundary_conditions import boundary_conditions
from Losses.losses_solver import (
    AXIAL_INLET_PROFILE_LOSS,
    IMPULSE_PROFILE_LOSS,
    RowFlow,
    RowGeometry,
    compressibility_correction,
    hub_mach_factor,
    hub_shock_loss,
    load_chart,
    reynolds_correction,
    secondary_loss,
    stagnation_pressure_ratio,
    supersonic_profile_factor,
    tip_efficiency_debit,
    trailing_edge_loss,
)
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

    def test_profile_charts_track_aungier_reference_curves(self) -> None:
        nozzle_loss = AXIAL_INLET_PROFILE_LOSS.profile_loss(0.762497, 40.0, [])
        impulse_loss = IMPULSE_PROFILE_LOSS.profile_loss(0.890887, 40.0, [])
        nozzle_reference = 0.0202667605
        impulse_reference = 0.0679960989

        self.assertLess(abs(nozzle_loss - nozzle_reference) / nozzle_reference, 0.07)
        self.assertLess(abs(impulse_loss - impulse_reference) / impulse_reference, 0.04)

    def test_chart_extrapolation_is_reported(self) -> None:
        warnings: list[str] = []

        AXIAL_INLET_PROFILE_LOSS.profile_loss(1.5, 90.0, warnings)

        self.assertEqual(len(warnings), 2)

    def test_hub_mach_factor_uses_digitized_reference_points(self) -> None:
        warnings: list[str] = []

        self.assertAlmostEqual(hub_mach_factor(0.65, "stator", warnings), 1.115)
        self.assertAlmostEqual(hub_mach_factor(0.7, "rotor", warnings), 1.35)
        self.assertEqual(warnings, [])

    def test_hub_shock_matches_reference_pressure_ratio(self) -> None:
        geometry = RowGeometry(
            pitch_chord=0.8,
            chord=0.4,
            axial_chord=0.3,
            height=0.2,
            max_thickness_chord=0.2,
            trailing_edge_throat=0.08,
            inlet_metal_angle=20.0,
            hub_tip_ratio_inlet=0.7,
            hub_radius_exit=0.3,
            tip_radius_exit=0.4,
        )
        flow = RowFlow(
            kind="rotor",
            inlet_angle=20.0,
            exit_angle=-40.0,
            inlet_mach=0.8,
            exit_mach=1.1,
            inlet_static_pressure=4.0,
            exit_static_pressure=3.0,
            reynolds=5e5,
            gamma=1.4,
        )
        warnings: list[str] = []
        pressure_loss = hub_shock_loss(geometry, flow, warnings)
        inlet_total_pressure = flow.inlet_static_pressure * stagnation_pressure_ratio(
            flow.inlet_mach, flow.gamma
        )
        exit_total_pressure = flow.exit_static_pressure * stagnation_pressure_ratio(
            flow.exit_mach, flow.gamma
        )
        hub_inlet_mach = 1.35 * flow.inlet_mach
        hub_loss = 0.75 * (hub_inlet_mach - 0.4) ** 1.75
        expected_loss = (
            geometry.hub_tip_ratio_inlet
            * hub_loss
            * (inlet_total_pressure - flow.inlet_static_pressure)
            / (exit_total_pressure - flow.exit_static_pressure)
        )

        self.assertAlmostEqual(pressure_loss, expected_loss)
        self.assertEqual(warnings, [])

    def test_hub_mach_factor_clamps_with_a_range_warning(self) -> None:
        warnings: list[str] = []

        factor = hub_mach_factor(0.4, "stator", warnings)

        self.assertEqual(factor, 1.4)
        self.assertEqual(len(warnings), 1)

    def test_secondary_loss_matches_reference_loading_relation(self) -> None:
        geometry = RowGeometry(
            pitch_chord=0.8,
            chord=0.4,
            axial_chord=0.3,
            height=0.2,
            max_thickness_chord=0.2,
            trailing_edge_throat=0.08,
            inlet_metal_angle=15.0,
            hub_tip_ratio_inlet=0.7,
            hub_radius_exit=0.3,
            tip_radius_exit=0.4,
        )
        inlet_angle = 23.0
        exit_angle = 58.0
        mach_correction = 0.85
        source_exit_angle = -exit_angle
        inlet_radians = math.radians(inlet_angle)
        exit_radians = math.radians(source_exit_angle)
        mean_angle = math.atan((math.tan(inlet_radians) + math.tan(exit_radians)) / 2)
        lift_over_pitch_chord = (
            2 * math.cos(mean_angle) * (math.tan(inlet_radians) - math.tan(exit_radians))
        )
        loading = (
            lift_over_pitch_chord**2
            * math.cos(exit_radians) ** 2
            / math.cos(mean_angle) ** 3
        )
        aspect_ratio = geometry.height / geometry.chord
        aspect_ratio_factor = (1 - 0.25 * math.sqrt(2 - aspect_ratio)) / aspect_ratio
        angle_factor = math.cos(exit_radians) / math.cos(math.radians(geometry.inlet_metal_angle))
        compressibility_factor = (
            1 - (geometry.axial_chord / geometry.height) ** 2 * (1 - mach_correction)
        )
        expected_loss = (
            1.2 * 0.0334 * aspect_ratio_factor * angle_factor * loading * compressibility_factor
        )

        self.assertAlmostEqual(
            secondary_loss(geometry, inlet_angle, exit_angle, geometry.inlet_metal_angle, mach_correction),
            expected_loss,
        )

    def test_loss_correction_factors_match_reference_limits(self) -> None:
        self.assertAlmostEqual(reynolds_correction(1e5), 2**0.4)
        self.assertEqual(reynolds_correction(2e5), 1.0)
        self.assertEqual(reynolds_correction(1e6), 1.0)
        self.assertAlmostEqual(reynolds_correction(2e6), 2**-0.2)
        self.assertAlmostEqual(compressibility_correction(0.3, 0.6), 0.875)
        self.assertEqual(compressibility_correction(0.5, 0.2), 1.0)
        self.assertEqual(supersonic_profile_factor(1.0), 1.0)
        self.assertAlmostEqual(supersonic_profile_factor(1.2), 3.4)
        with self.assertRaises(ValueError):
            reynolds_correction(0.0)
        with self.assertRaises(ValueError):
            compressibility_correction(0.2, 0.0)

    def test_tip_efficiency_debit_matches_unshrouded_rotor_correlation(self) -> None:
        geometry = RowGeometry(
            pitch_chord=0.8,
            chord=0.4,
            axial_chord=0.3,
            height=0.2,
            max_thickness_chord=0.2,
            trailing_edge_throat=0.08,
            inlet_metal_angle=20.0,
            hub_tip_ratio_inlet=0.7,
            hub_radius_exit=0.3,
            tip_radius_exit=0.4,
            tip_clearance=0.002,
        )
        flow = RowFlow(
            kind="rotor",
            inlet_angle=20.0,
            exit_angle=-40.0,
            inlet_mach=0.8,
            exit_mach=1.1,
            inlet_static_pressure=4.0,
            exit_static_pressure=3.0,
            reynolds=5e5,
            gamma=1.4,
        )
        expected_debit = (
            0.93
            * geometry.tip_clearance
            / (geometry.height * math.cos(math.radians(40.0)))
            * geometry.tip_radius_exit
            / ((geometry.hub_radius_exit + geometry.tip_radius_exit) / 2)
        )

        self.assertAlmostEqual(tip_efficiency_debit(geometry, flow, 40.0), expected_debit)
        self.assertEqual(tip_efficiency_debit(geometry, replace(flow, kind="stator"), 40.0), 0.0)

    def test_trailing_edge_range_and_zero_angle_are_explicit(self) -> None:
        warnings: list[str] = []

        trailing_edge_loss(0.5, 0.0, 40.0, 0.6, 1.4, warnings)

        self.assertEqual(len(warnings), 1)
        with self.assertRaises(ValueError):
            trailing_edge_loss(0.08, 0.0, 0.0, 0.6, 1.4, [])

    def test_trailing_edge_loss_uses_mach_dependent_pressure_conversion(self) -> None:
        low_mach_loss = trailing_edge_loss(0.2, 0.0, 40.0, 0.2, 1.4, [])
        higher_mach_loss = trailing_edge_loss(0.2, 0.0, 40.0, 0.6, 1.4, [])

        self.assertAlmostEqual(low_mach_loss, 0.04776609112)
        self.assertAlmostEqual(higher_mach_loss, 0.05511130830)
        self.assertGreater(higher_mach_loss, low_mach_loss)

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
