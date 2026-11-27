import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from Pipeline.Airfoil.airfoil import rotor_row, stator_row
from Constraints.boundary_conditions import boundary_conditions
from Helpers.design_optimizer import MODEL_FAILURE_PENALTY, DesignSpace, penalized_objective, screen_turbine
from Pipeline.Losses.losses_solver import row_geometry, rotor_losses, stage_efficiency, stator_losses
from Runner.main import SELECTED_DESIGN
from Pipeline.Meanline.gas_props import AirState, GAS
from Pipeline.Meanline.solve_meanline import Inputs, StageInlet, solve_meanline
from Pipeline.RadialEquilibrium.radial_equilibrium_solver import solve_radial_equilibrium
from Helpers.stage_solver import LOSS_TOLERANCE, StageDesign, solve_stage
from Helpers.turbine_solver import TurbineDesign, solve_turbine

DOWNSTREAM_INLET = StageInlet(T_0=1500.0, P_0=8.0, m_dot=30.0, alpha=-30.0, mach=0.4)
DOWNSTREAM_INPUTS = Inputs(pressure_ratio=2.0, alpha_2=68.0, beta_3=-58.0, N_shaft=10000)
TWO_STAGE_INPUTS = Inputs(degree_of_reaction=0.4, N_shaft=11000, radius_2=0.4339, radius_3=0.4339, alpha_2=65.0,
                          beta_3=-60.0, loss_kinetic_stator=0.08, loss_kinetic_rotor=0.1)


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
            density = meanline.stator_exit_air_state.density_kg_m3
            area = 2 * math.pi * inputs.radius_2 * meanline.blade_height_2
            with self.subTest(inputs=inputs):
                self.assertAlmostEqual(
                    meanline.stator_exit_air_state.enthalpy_j_kg + meanline.v_2**2 / 2,
                    meanline.inlet_air_state.enthalpy_j_kg,
                    places=7,
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
            rothalpy_in = meanline.stator_exit_air_state.enthalpy_j_kg + (
                meanline.w_2**2 - meanline.u_2**2
            ) / 2
            rothalpy_out = meanline.rotor_exit_air_state.enthalpy_j_kg + (
                meanline.w_3**2 - meanline.u_3**2
            ) / 2
            with self.subTest(inputs=inputs):
                self.assertAlmostEqual(meanline.work_euler, euler_work, places=8)
                self.assertAlmostEqual(meanline.work_thermo, euler_work, places=7)
                self.assertAlmostEqual(rothalpy_in, rothalpy_out, places=7)

    def test_reaction_and_efficiencies_match_isentropic_pressure_relations(self) -> None:
        inputs = Inputs()
        meanline = solve_meanline(inputs)
        overall_isentropic = AirState.isentropic_from_temperature_pressure(
            meanline.T_01,
            meanline.P_01 * 1e5,
            meanline.P_s3 * 1e5,
        )
        stator_isentropic = AirState.isentropic_from_temperature_pressure(
            meanline.T_01,
            meanline.P_01 * 1e5,
            meanline.P_s2 * 1e5,
        )
        stator_drop = meanline.inlet_air_state.enthalpy_j_kg - stator_isentropic.enthalpy_j_kg
        overall_drop = meanline.inlet_air_state.enthalpy_j_kg - overall_isentropic.enthalpy_j_kg
        reaction = 1 - stator_drop / overall_drop
        ideal_static_drop = meanline.inlet_air_state.enthalpy_j_kg - overall_isentropic.enthalpy_j_kg
        ideal_total = AirState.isentropic_from_temperature_pressure(
            meanline.T_01,
            meanline.P_01 * 1e5,
            meanline.P_03 * 1e5,
        )
        ideal_total_drop = meanline.inlet_air_state.enthalpy_j_kg - ideal_total.enthalpy_j_kg
        expected_eta_ts = meanline.work_thermo / ideal_static_drop
        expected_eta_tt = meanline.work_thermo / ideal_total_drop
        expected_exit_total_pressure = meanline.rotor_exit_stagnation_air_state.pressure_pa / 1e5

        self.assertAlmostEqual(reaction, inputs.degree_of_reaction, places=12)
        self.assertAlmostEqual(meanline.eta_ts, expected_eta_ts, places=12)
        self.assertAlmostEqual(meanline.eta_tt, expected_eta_tt, places=12)
        self.assertAlmostEqual(meanline.P_03, expected_exit_total_pressure, places=12)

    def test_signed_angles_and_mach_numbers(self) -> None:
        inputs = Inputs()
        meanline = solve_meanline(inputs)
        speed_of_sound_3 = math.sqrt(
            meanline.rotor_exit_air_state.gamma
            * meanline.rotor_exit_air_state.gas_constant_j_kg_k
            * meanline.T_3
        )
        c_x_2 = meanline.v_2 * math.cos(math.radians(meanline.alpha_2))
        c_theta_2 = meanline.v_2 * math.sin(math.radians(meanline.alpha_2))
        c_x_3 = meanline.w_3 * math.cos(math.radians(meanline.beta_3))
        c_theta_3 = meanline.u_3 + meanline.w_3 * math.sin(math.radians(meanline.beta_3))

        self.assertEqual(meanline.beta_3, inputs.beta_3)
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
        solution = solve_turbine(SELECTED_DESIGN)

        self.assertTrue(solution.converged)
        for constraint in solution.constraints:
            with self.subTest(constraint=constraint.name):
                self.assertTrue(constraint.satisfied, f"{constraint.value} vs limit {constraint.limit}")


class TurbineTests(unittest.TestCase):
    def test_one_stage_turbine_matches_the_stage_loop(self) -> None:
        stage = StageDesign(inputs=TWO_STAGE_INPUTS)
        turbine = solve_turbine(TurbineDesign(stages=(stage,)))
        alone = solve_stage(stage)

        self.assertAlmostEqual(turbine.performance.total_to_static, alone.efficiency.total_to_static)
        self.assertAlmostEqual(turbine.performance.total_to_total, alone.efficiency.total_to_total)

    def test_second_stage_starts_from_the_first_stage_exit(self) -> None:
        stage = StageDesign(inputs=TWO_STAGE_INPUTS)
        turbine = solve_turbine(TurbineDesign(stages=(stage, stage), expansion_shares=(0.5, 0.5)))
        first, second = (stage.final.meanline for stage in turbine.stages)
        overall_exit_pressure = boundary_conditions.P_01 / boundary_conditions.pressure_ratio_TS

        self.assertAlmostEqual(second.T_01, first.T_03)
        self.assertAlmostEqual(second.P_01, first.P_03)
        self.assertAlmostEqual(second.alpha_1, first.alpha_3)
        self.assertAlmostEqual(second.M_1, first.M_3)
        self.assertAlmostEqual(second.P_s3, overall_exit_pressure)
        self.assertAlmostEqual(turbine.performance.specific_work,
                               sum(stage.efficiency.specific_work for stage in turbine.stages))

    def test_rig_limits_the_first_rotor_hub_to_tip_ratio(self) -> None:
        short_blades = StageDesign(inputs=replace(TWO_STAGE_INPUTS, alpha_2=70.0, beta_3=-70.0,
                                                  degree_of_reaction=0.25))
        solution = solve_turbine(TurbineDesign(stages=(short_blades,)))
        violated = {constraint.name for constraint in solution.constraints if not constraint.satisfied}
        first_rotor = solution.stages[0].final.meanline

        self.assertGreater(first_rotor.D_hub_2 / first_rotor.D_tip_2, boundary_conditions.rig_max_hub_tip_ratio)
        self.assertEqual(violated, {"AMBRA rotor inlet hub/tip"})
        self.assertFalse(solution.feasible)

    def test_invalid_turbines_are_rejected(self) -> None:
        stage = StageDesign(inputs=TWO_STAGE_INPUTS)
        other_speed = replace(stage, inputs=replace(TWO_STAGE_INPUTS, N_shaft=12000))

        with self.assertRaises(ValueError):
            TurbineDesign(stages=(stage, stage), expansion_shares=(0.6, 0.6))
        with self.assertRaises(ValueError):
            TurbineDesign(stages=(stage, other_speed), expansion_shares=(0.5, 0.5))


class OptimizerTests(unittest.TestCase):
    def test_decoded_first_rotor_never_exceeds_its_blade_speed(self) -> None:
        random = np.random.default_rng(seed=3)
        for stage_count in (1, 2):
            space = DesignSpace(stage_count)
            lower, upper = np.array(space.scipy_bounds()).T
            for variables in random.uniform(lower, upper, size=(200, len(lower))):
                design = space.decode(variables)
                first = design.stages[0].inputs
                omega = first.N_shaft * 2 * math.pi / 60
                self.assertLessEqual(omega * max(first.radius_2, first.radius_3), variables[1])

    def test_mean_line_runs_on_between_stages(self) -> None:
        space = DesignSpace(2)
        lower, upper = np.array(space.scipy_bounds()).T
        design = space.decode((lower + upper) / 2)

        self.assertEqual(design.stages[1].inputs.radius_2, design.stages[0].inputs.radius_3)
        self.assertAlmostEqual(sum(design.expansion_shares), 1.0)

    def test_screening_rejects_designs_outside_the_stagger_chart(self) -> None:
        design = TurbineDesign(stages=(StageDesign(inputs=Inputs(alpha_2=45.0)),))
        screened = screen_turbine(design)

        self.assertFalse(screened.feasible)
        self.assertTrue(screened.error)

    def test_unusable_expansion_shares_get_the_failure_penalty(self) -> None:
        space = DesignSpace(3)
        lower, upper = np.array(space.scipy_bounds()).T
        variables = (lower + upper) / 2
        variables[-2:] = 0.6

        self.assertEqual(penalized_objective(variables, space), MODEL_FAILURE_PENALTY)


if __name__ == "__main__":
    unittest.main()
