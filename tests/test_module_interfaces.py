import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from Airfoil.airfoil import rotor_row, stator_row
from boundary_conditions import boundary_conditions
from design_optimizer import MODEL_FAILURE_PENALTY, DesignSpace, penalized_objective, screen_turbine
from Losses.losses_solver import row_geometry, rotor_losses, stage_efficiency, stator_losses
from main import SELECTED_DESIGN
from Meanline.gas_props import GAS
from Meanline.solve_meanline import Inputs, StageInlet, solve_meanline
from RadialEquilibrium.radial_equilibrium_solver import solve_radial_equilibrium
from stage_solver import LOSS_TOLERANCE, StageDesign, solve_stage
from turbine_solver import TurbineDesign, solve_turbine

DOWNSTREAM_INLET = StageInlet(T_0=1500.0, P_0=8.0, m_dot=30.0, alpha=-30.0, mach=0.4)
DOWNSTREAM_INPUTS = Inputs(pressure_ratio=2.0, alpha_2=68.0, beta_3=-58.0, N_shaft=10000)
TWO_STAGE_INPUTS = Inputs(degree_of_reaction=0.4, N_shaft=11000, radius_2=0.4339, radius_3=0.4339, alpha_2=65.0,
                          beta_3=-60.0, loss_kinetic_stator=0.08, loss_kinetic_rotor=0.1)


class MeanlineTests(unittest.TestCase):
    def test_continuity_and_energy_close(self) -> None:
        meanline = solve_meanline()

        self.assertLess(abs(meanline.residual), 1e-9)
        self.assertAlmostEqual(meanline.work_euler, meanline.work_thermo, delta=1e-6 * meanline.work_thermo)

    def test_signed_angles_and_mach_numbers(self) -> None:
        inputs = Inputs()
        meanline = solve_meanline(inputs)
        speed_of_sound_3 = math.sqrt(GAS.gamma * GAS.R * meanline.T_3)

        self.assertEqual(meanline.beta_3, inputs.beta_3)
        self.assertGreater(meanline.beta_2, 0.0)
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
