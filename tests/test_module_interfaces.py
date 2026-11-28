import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from Pipeline.Airfoil.airfoil import rotor_row, stator_row
from Constraints.boundary_conditions import boundary_conditions
from Helpers.design_optimizer import (
    CandidateStatus,
    export_final_selection_record,
    MODEL_FAILURE_PENALTY,
    DesignSpace,
    OptimizerCandidate,
    final_selection_record,
    export_optimizer_run_record,
    MultiModeOptimizationResult,
    pareto_frontier,
    penalized_objective,
    select_final_candidate,
    screen_turbine,
)
from Helpers.design_space import ExitFlowLimits, VariableBounds
from Helpers.design_sweep import SweepGrid
from Pipeline.Losses.losses_solver import (
    RowFlow,
    assess_transonic,
    classify_mach,
    rotor_losses,
    rotor_row_inputs,
    row_geometry,
    stage_efficiency,
    stator_losses,
    stator_row_inputs,
)
from Runner.main import SELECTED_DESIGN
from Pipeline.Meanline.gas_props import AirState, GAS
from Pipeline.Meanline.solve_meanline import Inputs, StageInlet, solve_meanline
from Pipeline.RadialEquilibrium.radial_equilibrium_solver import solve_radial_equilibrium
from Helpers.stage_solver import (
    ENERGY_ABSOLUTE_TOLERANCE,
    ENERGY_RELATIVE_TOLERANCE,
    LOSS_TOLERANCE,
    MASS_FLOW_TOLERANCE,
    MAX_PASSES,
    StageDesign,
    evaluate_stage,
    print_stage_report,
    solve_stage,
    stage_constraints,
)
from Helpers.turbine_solver import TurbineDesign, print_turbine_report, solve_turbine

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

    def test_kinetic_loss_inputs_match_stator_and_rotor_states(self) -> None:
        inputs = Inputs(loss_kinetic_stator=0.12, loss_kinetic_rotor=0.16)
        meanline = solve_meanline(inputs)
        stator_isentropic_drop = (
            meanline.inlet_air_state.enthalpy_j_kg
            - meanline.stator_isentropic_air_state.enthalpy_j_kg
        )
        rotor_relative_inlet_enthalpy = meanline.stator_exit_air_state.enthalpy_j_kg + (
            meanline.w_2**2 + meanline.u_3**2 - meanline.u_2**2
        ) / 2
        rotor_isentropic_drop = (
            rotor_relative_inlet_enthalpy
            - meanline.rotor_isentropic_air_state.enthalpy_j_kg
        )

        self.assertAlmostEqual(
            1 - meanline.v_2**2 / (2 * stator_isentropic_drop),
            inputs.loss_kinetic_stator,
        )
        self.assertAlmostEqual(
            1 - meanline.w_3**2 / (2 * rotor_isentropic_drop),
            inputs.loss_kinetic_rotor,
        )

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
        cases = (
            (Inputs(), StageInlet()),
            (DOWNSTREAM_INPUTS, DOWNSTREAM_INLET),
        )
        for inputs, inlet in cases:
            meanline = solve_meanline(inputs, inlet)
            speed_of_sound_2 = math.sqrt(
                meanline.stator_exit_air_state.gamma
                * meanline.stator_exit_air_state.gas_constant_j_kg_k
                * meanline.T_2
            )
            speed_of_sound_3 = math.sqrt(
                meanline.rotor_exit_air_state.gamma
                * meanline.rotor_exit_air_state.gas_constant_j_kg_k
                * meanline.T_3
            )
            c_x_2 = meanline.v_2 * math.cos(math.radians(meanline.alpha_2))
            c_theta_2 = meanline.v_2 * math.sin(math.radians(meanline.alpha_2))
            c_x_3 = meanline.w_3 * math.cos(math.radians(meanline.beta_3))
            c_theta_3 = meanline.u_3 + meanline.w_3 * math.sin(math.radians(meanline.beta_3))

            with self.subTest(inputs=inputs):
                self.assertEqual(meanline.beta_3, inputs.beta_3)
                self.assertAlmostEqual(
                    meanline.beta_2,
                    math.degrees(math.atan2(c_theta_2 - meanline.u_2, c_x_2)),
                )
                self.assertAlmostEqual(
                    meanline.alpha_3,
                    math.degrees(math.atan2(c_theta_3, c_x_3)),
                )
                self.assertAlmostEqual(meanline.M_2, meanline.v_2 / speed_of_sound_2)
                self.assertAlmostEqual(meanline.M_2rel, meanline.w_2 / speed_of_sound_2)
                self.assertAlmostEqual(meanline.M_3, meanline.v_3 / speed_of_sound_3)
                self.assertAlmostEqual(meanline.M_3rel, meanline.w_3 / speed_of_sound_3)


class InterfaceTests(unittest.TestCase):
    def test_transonic_threshold_and_span_assessment_are_explicit(self) -> None:
        self.assertEqual(classify_mach(0.899), "subsonic")
        self.assertEqual(classify_mach(0.900), "transonic")
        self.assertEqual(classify_mach(1.000), "transonic")
        self.assertEqual(classify_mach(1.001), "supersonic")

        meanline = solve_meanline()
        spanwise = solve_radial_equilibrium(meanline)
        stator = stator_row(spanwise)
        rotor = rotor_row(spanwise)
        assessment = assess_transonic(
            meanline,
            spanwise,
            stator.airfoil_params(),
            rotor.airfoil_params(),
            stage_index=2,
        )

        self.assertEqual(assessment.stage_index, 2)
        self.assertEqual(
            tuple(span.span_location for span in assessment.spans),
            ("hub", "midspan", "tip"),
        )
        self.assertEqual(
            tuple(span.span_fraction for span in assessment.spans),
            (0.0, 0.5, 1.0),
        )
        for span in assessment.spans:
            mach_by_frame = {
                (mach.station_id, mach.frame): mach.mach
                for mach in span.mach_values
            }
            span_flow = {
                "hub": spanwise.hub,
                "midspan": spanwise.midspan,
                "tip": spanwise.tip,
            }[span.span_location]
            self.assertAlmostEqual(
                mach_by_frame["(2, 2)", "absolute"],
                span_flow.stator_exit.mach,
            )
            self.assertAlmostEqual(
                mach_by_frame["(2, 2)", "rotor_relative"],
                span_flow.stator_exit.relative_mach,
            )
            self.assertAlmostEqual(
                mach_by_frame["(2, 3)", "absolute"],
                span_flow.rotor_exit.mach,
            )
            self.assertAlmostEqual(
                mach_by_frame["(2, 3)", "rotor_relative"],
                span_flow.rotor_exit.relative_mach,
            )
            self.assertAlmostEqual(span.stator_proxy.loss.accounting_residual, 0.0, places=12)
            self.assertAlmostEqual(span.rotor_proxy.loss.accounting_residual, 0.0, places=12)

    def test_row_loss_adapters_preserve_frame_and_midspan_identity(self) -> None:
        meanline = solve_meanline()
        spanwise = solve_radial_equilibrium(meanline)
        stator = stator_row(spanwise)
        rotor = rotor_row(spanwise)

        stator_inputs = stator_row_inputs(meanline, stator.airfoil_params(), stage_index=2)
        rotor_inputs = rotor_row_inputs(meanline, rotor.airfoil_params(), stage_index=2)

        self.assertEqual(stator_inputs.stage_index, 2)
        self.assertEqual(stator_inputs.row_kind, "stator")
        self.assertEqual(stator_inputs.reference_frame, "absolute")
        self.assertEqual(stator_inputs.span_location, "midspan")
        self.assertAlmostEqual(stator_inputs.flow.inlet_angle, meanline.alpha_1)
        self.assertAlmostEqual(stator_inputs.flow.exit_angle, meanline.alpha_2)
        self.assertAlmostEqual(stator_inputs.flow.inlet_mach, meanline.M_1)
        self.assertAlmostEqual(stator_inputs.flow.exit_mach, meanline.M_2)

        self.assertEqual(rotor_inputs.stage_index, 2)
        self.assertEqual(rotor_inputs.row_kind, "rotor")
        self.assertEqual(rotor_inputs.reference_frame, "rotor_relative")
        self.assertEqual(rotor_inputs.span_location, "midspan")
        self.assertAlmostEqual(rotor_inputs.flow.inlet_angle, meanline.beta_2)
        self.assertAlmostEqual(rotor_inputs.flow.exit_angle, meanline.beta_3)
        self.assertAlmostEqual(rotor_inputs.flow.inlet_mach, meanline.M_2rel)
        self.assertAlmostEqual(rotor_inputs.flow.exit_mach, meanline.M_3rel)

    def test_row_loss_adapters_require_positive_reynolds_and_mach(self) -> None:
        meanline = solve_meanline()
        spanwise = solve_radial_equilibrium(meanline)
        stator_inputs = stator_row_inputs(meanline, stator_row(spanwise).airfoil_params())

        for field_name, invalid_value in (
            ("reynolds", 0.0),
            ("exit_mach", 0.0),
            ("exit_static_pressure", 0.0),
            ("gamma", 1.0),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    values = {
                        "kind": stator_inputs.flow.kind,
                        "inlet_angle": stator_inputs.flow.inlet_angle,
                        "exit_angle": stator_inputs.flow.exit_angle,
                        "inlet_mach": stator_inputs.flow.inlet_mach,
                        "exit_mach": stator_inputs.flow.exit_mach,
                        "inlet_static_pressure": stator_inputs.flow.inlet_static_pressure,
                        "exit_static_pressure": stator_inputs.flow.exit_static_pressure,
                        "reynolds": stator_inputs.flow.reynolds,
                        "gamma": stator_inputs.flow.gamma,
                    }
                    values[field_name] = invalid_value
                    RowFlow(**values)

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
        self.assertTrue(final.balances_converged)
        self.assertLess(abs(final.mass_flow_residual), MASS_FLOW_TOLERANCE)
        self.assertTrue(
            math.isclose(
                final.meanline.work_euler,
                final.meanline.work_thermo,
                rel_tol=ENERGY_RELATIVE_TOLERANCE,
                abs_tol=ENERGY_ABSOLUTE_TOLERANCE,
            )
        )
        self.assertLess(abs(final.inputs.loss_kinetic_stator - final.stator_loss.kinetic_loss), LOSS_TOLERANCE)
        self.assertLess(abs(final.inputs.loss_kinetic_rotor - final.rotor_loss.kinetic_loss), LOSS_TOLERANCE)
        self.assertAlmostEqual(solution.efficiency.total_to_static,
                               final.meanline.eta_ts * (1 - final.rotor_loss.tip_efficiency_debit))

    def test_balance_residual_failures_prevent_converged_status(self) -> None:
        solution = solve_stage(StageDesign())
        final = solution.final
        invalid_meanlines = (
            ("mass", replace(final.meanline, residual=2 * MASS_FLOW_TOLERANCE)),
            ("energy", replace(final.meanline, work_thermo=final.meanline.work_thermo + 1.0)),
        )

        for residual_type, meanline in invalid_meanlines:
            with self.subTest(residual_type=residual_type):
                invalid_final = replace(final, meanline=meanline)
                invalid_solution = replace(
                    solution,
                    evaluations=solution.evaluations[:-1] + (invalid_final,),
                )
                self.assertFalse(invalid_solution.converged)
                self.assertFalse(invalid_solution.feasible)

    def test_max_pass_exhaustion_is_explicitly_non_converged(self) -> None:
        design = StageDesign()
        evaluation = evaluate_stage(design.inputs, design)
        report = StringIO()

        with patch("Helpers.stage_solver.MAX_PASSES", 2), patch(
            "Helpers.stage_solver.evaluate_stage",
            return_value=evaluation,
        ), patch("Helpers.stage_solver.loss_change", return_value=LOSS_TOLERANCE), patch(
            "sys.stdout",
            report,
        ):
            solution = solve_stage(design)
            print_stage_report(solution)

        self.assertEqual(len(solution.evaluations), 2)
        self.assertFalse(solution.converged)
        self.assertFalse(solution.feasible)
        self.assertIn("NOT converged after 2 passes", report.getvalue())
        self.assertIn("(maximum 2;", report.getvalue())

    def test_stage_report_prints_converged_residuals(self) -> None:
        report = StringIO()
        solution = solve_stage(StageDesign())

        with patch("sys.stdout", report):
            print_stage_report(solution)

        self.assertIn("Final coupled residuals", report.getvalue())
        self.assertIn("Mass-flow residual [kg/s]", report.getvalue())
        self.assertIn("Euler-energy residual [J/kg]", report.getvalue())
        self.assertIn("Stator loss-coefficient residual", report.getvalue())
        self.assertIn("Rotor loss-coefficient residual", report.getvalue())

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

    def test_overall_work_power_efficiencies_and_exit_energy_reconcile(self) -> None:
        stage = StageDesign(inputs=TWO_STAGE_INPUTS)
        designs = (
            TurbineDesign(stages=(stage,)),
            TurbineDesign(stages=(stage, stage), expansion_shares=(0.5, 0.5)),
        )

        for design in designs:
            turbine = solve_turbine(design)
            performance = turbine.performance
            exit_meanline = turbine.stages[-1].final.meanline
            inlet_state = AirState.from_temperature_pressure(
                design.inlet.T_0,
                design.inlet.P_0 * 1e5,
            )
            ideal_static = AirState.isentropic_from_temperature_pressure(
                design.inlet.T_0,
                design.inlet.P_0 * 1e5,
                exit_meanline.P_s3 * 1e5,
            )
            ideal_total = AirState.isentropic_from_temperature_pressure(
                design.inlet.T_0,
                design.inlet.P_0 * 1e5,
                exit_meanline.P_03 * 1e5,
            )
            expected_work = math.fsum(stage.efficiency.specific_work for stage in turbine.stages)
            exit_swirl_velocity = exit_meanline.v_3 * math.sin(math.radians(exit_meanline.alpha_3))

            with self.subTest(stage_count=len(design.stages)):
                self.assertAlmostEqual(performance.specific_work, expected_work)
                self.assertAlmostEqual(performance.power_w, expected_work * design.inlet.m_dot)
                self.assertAlmostEqual(
                    performance.exit_kinetic_energy_j_kg,
                    exit_meanline.rotor_exit_stagnation_air_state.enthalpy_j_kg
                    - exit_meanline.rotor_exit_air_state.enthalpy_j_kg,
                )
                self.assertAlmostEqual(
                    performance.exit_kinetic_energy_j_kg,
                    0.5
                    * (
                        (exit_meanline.v_3 * math.cos(math.radians(exit_meanline.alpha_3))) ** 2
                        + exit_swirl_velocity**2
                    ),
                )
                self.assertAlmostEqual(
                    performance.exit_swirl_kinetic_energy_j_kg,
                    0.5 * exit_swirl_velocity**2,
                )
                self.assertAlmostEqual(
                    performance.total_to_static,
                    expected_work
                    / (inlet_state.enthalpy_j_kg - ideal_static.enthalpy_j_kg),
                )
                self.assertAlmostEqual(
                    performance.total_to_total,
                    expected_work
                    / (inlet_state.enthalpy_j_kg - ideal_total.enthalpy_j_kg),
                )
                self.assertAlmostEqual(math.fsum(performance.work_shares), 1.0)

            if len(design.stages) == 1:
                report = StringIO()
                with patch("sys.stdout", report):
                    print_turbine_report(turbine)
                self.assertIn("Power [MW]", report.getvalue())
                self.assertIn("Exit kinetic energy [kJ/kg]", report.getvalue())
                self.assertIn("Exit swirl kinetic energy [kJ/kg]", report.getvalue())

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
        with self.assertRaisesRegex(ValueError, "one or two stages"):
            TurbineDesign(
                stages=(stage, stage, stage),
                expansion_shares=(1 / 3, 1 / 3, 1 / 3),
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            TurbineDesign(stages=(stage,), expansion_shares=(float("nan"),))


class OptimizerTests(unittest.TestCase):
    def test_design_space_only_supports_one_or_two_stages(self) -> None:
        for stage_count in (0, 3, 1.0, True):
            with self.subTest(stage_count=stage_count), self.assertRaises(ValueError):
                DesignSpace(stage_count)

    def test_decode_rejects_out_of_range_or_malformed_vectors(self) -> None:
        space = DesignSpace(1)
        lower, upper = np.array(space.scipy_bounds()).T
        valid = (lower + upper) / 2
        outside = valid.copy()
        outside[2] = upper[2] + 0.01

        with self.assertRaisesRegex(ValueError, "design variable 2"):
            space.decode(outside)
        with self.assertRaisesRegex(ValueError, "expected"):
            space.decode(valid[:-1])

    def test_variable_bounds_reject_physically_invalid_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "radius_ratio"):
            VariableBounds(radius_ratio=(0.9, 1.1))

    def test_default_exit_flow_limits_are_disabled_and_optional_limits_are_enforced(self) -> None:
        meanline = solve_meanline()
        shaft_speed = meanline.omega * 60 / (2 * math.pi)
        default_constraints = stage_constraints(meanline, shaft_speed, 0.2, 0)
        limited_constraints = stage_constraints(
            meanline,
            shaft_speed,
            0.2,
            0,
            ExitFlowLimits(
                max_absolute_exit_mach=meanline.M_3 / 2,
                max_absolute_exit_swirl_m_s=abs(
                    meanline.v_3 * math.sin(math.radians(meanline.alpha_3))
                )
                / 2,
            ),
        )

        self.assertNotIn("Rotor exit absolute Mach", {constraint.name for constraint in default_constraints})
        exit_constraints = {
            constraint.name: constraint
            for constraint in limited_constraints
            if constraint.name.startswith("Rotor exit absolute")
        }
        self.assertFalse(exit_constraints["Rotor exit absolute Mach"].satisfied)
        self.assertFalse(exit_constraints["Rotor exit absolute swirl [m/s]"].satisfied)
        space = DesignSpace(1, exit_flow_limits=ExitFlowLimits(max_absolute_exit_mach=1.0))
        lower, upper = np.array(space.scipy_bounds()).T
        decoded = space.decode((lower + upper) / 2)
        self.assertEqual(decoded.stages[0].exit_flow_limits.max_absolute_exit_mach, 1.0)

    def test_sweep_values_stay_within_canonical_optimizer_bounds(self) -> None:
        bounds = VariableBounds()
        grid = SweepGrid(bounds=bounds)
        for values, (lower, upper) in (
            (grid.blade_speeds, bounds.blade_speed),
            (grid.shaft_speeds, bounds.shaft_speed),
            (grid.radius_ratios, bounds.radius_ratio),
            (grid.reactions, bounds.reaction),
            (grid.stator_exit_angles, bounds.stator_exit_angle),
            (grid.rotor_exit_angles, bounds.rotor_exit_angle),
        ):
            self.assertTrue(all(lower <= value <= upper for value in values))

        with self.assertRaisesRegex(ValueError, "outside"):
            SweepGrid(radius_ratios=(1.0, 1.16), bounds=bounds)

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

    def test_pareto_frontier_keeps_nondominated_efficiency_power_candidates(self) -> None:
        candidates = (
            OptimizerCandidate("a", 1, CandidateStatus.FEASIBLE, 0.90, 30_000_000.0, None, (1.0,)),
            OptimizerCandidate("b", 1, CandidateStatus.FEASIBLE, 0.88, 31_000_000.0, None, (2.0,)),
            OptimizerCandidate("c", 2, CandidateStatus.FEASIBLE, 0.85, 32_000_000.0, None, (3.0,)),
            OptimizerCandidate("d", 1, CandidateStatus.INFEASIBLE, 0.95, 33_000_000.0, None, (4.0,)),
        )

        frontier = pareto_frontier(candidates)

        self.assertEqual(tuple(candidate.candidate_id for candidate in frontier), ("a", "b", "c"))

    def test_optimizer_candidate_status_and_cfd_status_are_explicit(self) -> None:
        candidate = OptimizerCandidate(
            "one-stage-1",
            1,
            CandidateStatus.FAILED,
            math.nan,
            math.nan,
            None,
            (1.0,),
            error="model failure",
        )

        self.assertEqual(candidate.status, CandidateStatus.FAILED)
        self.assertEqual(candidate.cfd_status, "not_requested")

    def test_final_selection_uses_each_run_start_objective_mode(self) -> None:
        candidates = (
            OptimizerCandidate(
                "one-eff",
                1,
                CandidateStatus.FEASIBLE,
                0.90,
                30_000_000.0,
                None,
                (1.0,),
                converged=True,
            ),
            OptimizerCandidate(
                "one-power",
                1,
                CandidateStatus.FEASIBLE,
                0.88,
                32_000_000.0,
                None,
                (2.0,),
                converged=True,
            ),
        )

        efficiency_selection = select_final_candidate(candidates, "efficiency", stage_count=1)
        power_selection = select_final_candidate(candidates, "power", stage_count=1)
        pareto_selection = select_final_candidate(candidates, "pareto", stage_count=1)

        self.assertEqual(efficiency_selection.candidate.candidate_id, "one-eff")
        self.assertEqual(power_selection.candidate.candidate_id, "one-power")
        self.assertIn(pareto_selection.candidate.candidate_id, ("one-eff", "one-power"))
        self.assertEqual(efficiency_selection.validation_status, "provisional_meanline")
        self.assertEqual(efficiency_selection.cfd_status, "not_requested")

    def test_final_selection_rejects_unverified_or_infeasible_candidates(self) -> None:
        infeasible = OptimizerCandidate(
            "infeasible",
            1,
            CandidateStatus.INFEASIBLE,
            0.95,
            33_000_000.0,
            None,
            (1.0,),
            converged=True,
        )
        nonconverged = OptimizerCandidate(
            "nonconverged",
            2,
            CandidateStatus.NON_CONVERGED,
            0.94,
            32_000_000.0,
            None,
            (2.0,),
            converged=False,
        )

        with self.assertRaisesRegex(ValueError, "no feasible converged candidate"):
            select_final_candidate((infeasible, nonconverged), "efficiency")

    def test_final_selection_export_marks_cfd_pending_and_lists_alternatives(self) -> None:
        candidates = (
            OptimizerCandidate(
                "one",
                1,
                CandidateStatus.FEASIBLE,
                0.90,
                30_000_000.0,
                None,
                (1.0,),
                converged=True,
            ),
            OptimizerCandidate(
                "two",
                2,
                CandidateStatus.FEASIBLE,
                0.89,
                31_000_000.0,
                None,
                (2.0,),
                converged=True,
            ),
        )
        selection = select_final_candidate(candidates, "power")

        record = final_selection_record(selection)
        self.assertEqual(record["validation_status"], "provisional_meanline")
        self.assertEqual(record["cfd_status"], "not_requested")
        self.assertEqual(record["selected_candidate"]["candidate_id"], "two")
        self.assertEqual(len(record["alternative_candidates"]), 1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = export_final_selection_record(
                selection,
                f"{temporary_directory}/final-selection.json",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("Cooling and metal-temperature feasibility", payload["limitations"][1])

    def test_optimizer_run_record_exports_modes_candidates_and_cfd_boundary(self) -> None:
        candidate = OptimizerCandidate(
            "two-stage-1",
            2,
            CandidateStatus.FEASIBLE,
            0.90,
            30_000_000.0,
            None,
            (1.0,),
        )
        result = MultiModeOptimizationResult(
            efficiency=(),
            power=(),
            pareto=(candidate,),
            candidates=(candidate,),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = export_optimizer_run_record(
                result,
                f"{temporary_directory}/optimizer.json",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["modes"], ["efficiency", "power", "pareto"])
        self.assertEqual(payload["cfd_status"], "preparation_only")
        self.assertEqual(payload["candidates"][0]["status"], "feasible")

    def test_unusable_expansion_shares_get_the_failure_penalty(self) -> None:
        space = DesignSpace(2)
        lower, upper = np.array(space.scipy_bounds()).T
        variables = (lower + upper) / 2
        variables[-1] = upper[-1] + 0.1

        self.assertEqual(penalized_objective(variables, space), MODEL_FAILURE_PENALTY)


if __name__ == "__main__":
    unittest.main()
