import math
import sys
import unittest
from pathlib import Path

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from Airfoil.airfoil import rotor_row, stator_row
from Losses.losses_solver import row_geometry, rotor_losses, stage_efficiency, stator_losses
from Meanline.gas_props import GAS
from Meanline.solve_meanline import Inputs, StageInlet, solve_meanline
from RadialEquilibrium.radial_equilibrium_solver import solve_radial_equilibrium
from stage_solver import LOSS_TOLERANCE, StageDesign, solve_stage
from boundary_conditions import boundary_conditions
from design_sweep import SweepGrid, screen_design
from main import SELECTED_DESIGN

DOWNSTREAM_INLET = StageInlet(T_0=1500.0, P_0=8.0, m_dot=30.0)
DOWNSTREAM_INPUTS = Inputs(pressure_ratio=2.0, alpha_2=68.0, beta_3=-58.0, N_shaft=10000)


class MeanlineTests(unittest.TestCase):
    def test_averaged_gas_properties_preserve_the_perfect_gas_identity(self) -> None:
        self.assertAlmostEqual(
            GAS.C_p,
            GAS.gamma * GAS.R / (GAS.gamma - 1),
            places=8,
        )

    def test_continuity_and_energy_close(self) -> None:
        meanline = solve_meanline()

        self.assertLess(abs(meanline.residual), 1e-9)
        self.assertAlmostEqual(meanline.work_euler, meanline.work_thermo, delta=1e-6 * meanline.work_thermo)

    def test_stator_total_temperature_and_mass_flow_close(self) -> None:
        cases = (
            (Inputs(), StageInlet()),
            (DOWNSTREAM_INPUTS, DOWNSTREAM_INLET),
        )
        for inputs, inlet in cases:
            meanline = solve_meanline(inputs, inlet)
            axial_velocity = meanline.v_2 * math.cos(math.radians(meanline.alpha_2))
            density = meanline.P_s2 * 1e5 / (GAS.R * meanline.T_2)
            area = 2 * math.pi * inputs.radius_2 * meanline.blade_height_2
            with self.subTest(inputs=inputs):
                self.assertAlmostEqual(
                    meanline.T_2 + meanline.v_2**2 / (2 * GAS.C_p),
                    meanline.T_01,
                    places=10,
                )
                self.assertAlmostEqual(density * axial_velocity * area, inlet.m_dot, places=8)

    def test_rotor_rothalpy_and_euler_work_close_at_unequal_radii(self) -> None:
        cases = (
            (Inputs(), StageInlet()),
            (DOWNSTREAM_INPUTS, DOWNSTREAM_INLET),
        )
        for inputs, inlet in cases:
            meanline = solve_meanline(inputs, inlet)
            c_theta_2 = meanline.v_2 * math.sin(math.radians(meanline.alpha_2))
            c_theta_3 = meanline.u_3 + meanline.w_3 * math.sin(math.radians(meanline.beta_3))
            euler_work = meanline.u_2 * c_theta_2 - meanline.u_3 * c_theta_3
            rothalpy_in = GAS.C_p * meanline.T_2 + (
                meanline.w_2**2 - meanline.u_2**2
            ) / 2
            rothalpy_out = GAS.C_p * meanline.T_3 + (
                meanline.w_3**2 - meanline.u_3**2
            ) / 2
            with self.subTest(inputs=inputs):
                self.assertAlmostEqual(meanline.work_euler, euler_work, places=8)
                self.assertAlmostEqual(meanline.work_thermo, euler_work, places=7)
                self.assertAlmostEqual(rothalpy_in, rothalpy_out, places=7)

    def test_reaction_and_efficiencies_match_isentropic_pressure_relations(self) -> None:
        inputs = Inputs()
        meanline = solve_meanline(inputs)
        exponent = (GAS.gamma - 1) / GAS.gamma
        isentropic_exit_temperature = meanline.T_01 * (
            meanline.P_s3 / meanline.P_01
        ) ** exponent
        isentropic_stator_exit_temperature = meanline.T_01 * (
            meanline.P_s2 / meanline.P_01
        ) ** exponent
        reaction = (
            isentropic_stator_exit_temperature - isentropic_exit_temperature
        ) / (meanline.T_01 - isentropic_exit_temperature)
        expected_eta_ts = (meanline.T_01 - meanline.T_03) / (
            meanline.T_01
            * (1 - (meanline.P_s3 / meanline.P_01) ** exponent)
        )
        expected_eta_tt = (meanline.T_01 - meanline.T_03) / (
            meanline.T_01
            * (1 - (meanline.P_03 / meanline.P_01) ** exponent)
        )
        expected_exit_total_pressure = meanline.P_s3 * (
            meanline.T_03 / meanline.T_3
        ) ** (GAS.gamma / (GAS.gamma - 1))

        self.assertAlmostEqual(reaction, inputs.degree_of_reaction, places=12)
        self.assertAlmostEqual(meanline.eta_ts, expected_eta_ts, places=12)
        self.assertAlmostEqual(meanline.eta_tt, expected_eta_tt, places=12)
        self.assertAlmostEqual(meanline.P_03, expected_exit_total_pressure, places=12)

    def test_signed_angles_and_mach_numbers(self) -> None:
        inputs = Inputs()
        meanline = solve_meanline(inputs)
        speed_of_sound_3 = math.sqrt(GAS.gamma * GAS.R * meanline.T_3)
        c_x_2 = meanline.v_2 * math.cos(math.radians(meanline.alpha_2))
        c_theta_2 = meanline.v_2 * math.sin(math.radians(meanline.alpha_2))
        c_x_3 = meanline.w_3 * math.cos(math.radians(meanline.beta_3))
        c_theta_3 = meanline.u_3 + meanline.w_3 * math.sin(math.radians(meanline.beta_3))

        self.assertEqual(meanline.beta_3, inputs.beta_3)
        self.assertGreater(meanline.beta_2, 0.0)
        self.assertAlmostEqual(
            meanline.beta_2,
            math.degrees(math.atan2(c_theta_2 - meanline.u_2, c_x_2)),
        )
        self.assertAlmostEqual(
            meanline.alpha_3,
            math.degrees(math.atan2(c_theta_3, c_x_3)),
        )
        self.assertAlmostEqual(meanline.M_3rel, meanline.w_3 / speed_of_sound_3)


class InterfaceTests(unittest.TestCase):
    def test_free_vortex_midspan_reproduces_a_downstream_stage(self) -> None:
        meanline = solve_meanline(DOWNSTREAM_INPUTS, DOWNSTREAM_INLET)
        midspan = solve_radial_equilibrium(meanline).midspan

        self.assertAlmostEqual(midspan.stator_exit.static_pressure, meanline.P_s2)
        self.assertAlmostEqual(midspan.rotor_exit.static_pressure, meanline.P_s3)
        self.assertAlmostEqual(midspan.stator_exit.relative_angle, meanline.beta_2)
        self.assertAlmostEqual(midspan.rotor_exit.relative_angle, meanline.beta_3)
        self.assertAlmostEqual(midspan.degree_of_reaction, DOWNSTREAM_INPUTS.degree_of_reaction)

    def test_blade_rows_feed_losses_the_rounded_pitch(self) -> None:
        meanline = solve_meanline()
        stator = stator_row(solve_radial_equilibrium(meanline))
        geometry = row_geometry(stator.airfoil_params(), meanline, meanline.blade_height_2, 0.0,
                                meanline.alpha_2, meanline.D_hub_2, meanline.D_tip_2)

        self.assertAlmostEqual(geometry.pitch_chord, stator.midspan.pitch_chord)
        self.assertLess(stator.midspan.throat, stator.midspan.pitch)

    def test_exit_metal_angles_set_the_throat_to_the_exit_flow_angle(self) -> None:
        flow = solve_radial_equilibrium(solve_meanline())

        for row in (stator_row(flow), rotor_row(flow)):
            for name, section in row.sections.items():
                with self.subTest(row=row.kind, span=name):
                    self.assertAlmostEqual(section.throat_exit_angle, abs(section.flow.exit_angle), places=5)

    def test_losses_run_on_a_downstream_stage(self) -> None:
        meanline = solve_meanline(DOWNSTREAM_INPUTS, DOWNSTREAM_INLET)
        flow = solve_radial_equilibrium(meanline)
        stator = stator_losses(meanline, stator_row(flow).airfoil_params())
        rotor = rotor_losses(meanline, rotor_row(flow).airfoil_params())

        for loss in (stator, rotor):
            with self.subTest(row=loss.kind):
                self.assertTrue(0.0 < loss.kinetic_loss < 1.0)

    def test_stage_efficiency_applies_the_tip_debit_to_the_meanline(self) -> None:
        meanline = solve_meanline()
        rotor = rotor_losses(meanline)
        efficiency = stage_efficiency(meanline, rotor)
        tip_factor = 1 - rotor.tip_efficiency_debit

        self.assertAlmostEqual(efficiency.total_to_static, meanline.eta_ts * tip_factor)
        self.assertAlmostEqual(efficiency.specific_work, meanline.work_thermo * tip_factor)


class StageLoopTests(unittest.TestCase):
    def test_loop_converges_to_self_consistent_losses(self) -> None:
        solution = solve_stage(StageDesign())
        final = solution.final

        self.assertTrue(solution.converged)
        self.assertLess(abs(final.inputs.loss_kinetic_stator - final.stator_loss.kinetic_loss), LOSS_TOLERANCE)
        self.assertLess(abs(final.inputs.loss_kinetic_rotor - final.rotor_loss.kinetic_loss), LOSS_TOLERANCE)
        self.assertAlmostEqual(solution.efficiency.total_to_static,
                               final.meanline.eta_ts * (1 - final.rotor_loss.tip_efficiency_debit))

    def test_current_default_design_violates_the_blade_speed_limit(self) -> None:
        solution = solve_stage(StageDesign())
        violated = [constraint.name for constraint in solution.constraints if not constraint.satisfied]

        self.assertEqual(violated, ["Rotor mean blade speed [m/s]"])
        self.assertFalse(solution.feasible)

    def test_selected_design_satisfies_every_project_constraint(self) -> None:
        solution = solve_stage(SELECTED_DESIGN)

        self.assertTrue(solution.converged)
        for constraint in solution.constraints:
            with self.subTest(constraint=constraint.name):
                self.assertTrue(constraint.satisfied, f"{constraint.value} vs limit {constraint.limit}")


class DesignSweepTests(unittest.TestCase):
    def test_grid_designs_never_exceed_the_blade_speed_limit(self) -> None:
        for design in SweepGrid().designs(StageDesign()):
            omega = design.inputs.N_shaft * 2 * math.pi / 60
            blade_speed = omega * max(design.inputs.radius_2, design.inputs.radius_3)
            self.assertLessEqual(blade_speed, boundary_conditions.max_mean_blade_speed)

    def test_screening_rejects_designs_outside_the_stagger_chart(self) -> None:
        design = StageDesign(inputs=Inputs(alpha_2=45.0))
        screened = screen_design(design)

        self.assertFalse(screened.feasible)
        self.assertTrue(screened.error)


if __name__ == "__main__":
    unittest.main()
