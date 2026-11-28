"""Single-stage design loop: meanline, radial equilibrium, blade rows and losses iterated to consistent loss coefficients."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from Pipeline.Airfoil.airfoil import (ROTOR_DESIGN, STATOR_DESIGN, BladeRow, BladeRowDesign, print_blade_row, rotor_row,
                             stator_row)
from Constraints.boundary_conditions import boundary_conditions
from Pipeline.Losses.losses_solver import (RowLoss, StageEfficiency, print_breakdown, rotor_losses, stage_efficiency,
                                  stator_losses, TransonicAssessment, assess_transonic, print_transonic_assessment)
from Pipeline.Meanline.solve_meanline import Inputs, MeanlineResult, StageInlet, solve_meanline
from Pipeline.RadialEquilibrium.radial_equilibrium_solver import (FREE_VORTEX, SpanwiseFlow, print_spanwise_table,
                                                         solve_radial_equilibrium)
from Helpers.design_space import ExitFlowLimits

LOSS_TOLERANCE = 1e-6  # largest change of either kinetic loss coefficient between passes
MASS_FLOW_TOLERANCE = 1e-8  # kg/s
ENERGY_RELATIVE_TOLERANCE = 1e-9
ENERGY_ABSOLUTE_TOLERANCE = 1e-6  # J/kg
MAX_PASSES = 50


# ---------------------------------------------------------------------------
# Design choices and records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageDesign:
    """Every design choice of one stage; the loss coefficients in inputs are only the first guess."""

    stage_index: int = 1
    inputs: Inputs = field(default_factory=Inputs)
    inlet: StageInlet = field(default_factory=StageInlet)
    vortex_exponent: float = FREE_VORTEX
    stator: BladeRowDesign = STATOR_DESIGN
    rotor: BladeRowDesign = ROTOR_DESIGN
    exit_flow_limits: ExitFlowLimits = field(default_factory=ExitFlowLimits)

    def __post_init__(self) -> None:
        if isinstance(self.stage_index, bool) or not isinstance(self.stage_index, int) or self.stage_index < 1:
            raise ValueError(f"stage_index must be a positive integer, got {self.stage_index}")
        if not isinstance(self.exit_flow_limits, ExitFlowLimits):
            raise ValueError("exit_flow_limits must be an ExitFlowLimits instance")


@dataclass(frozen=True)
class StageEvaluation:
    """One pass through every module at fixed loss coefficients."""

    inputs: Inputs
    meanline: MeanlineResult
    spanwise_flow: SpanwiseFlow
    stator: BladeRow
    rotor: BladeRow
    stator_loss: RowLoss
    rotor_loss: RowLoss
    transonic_assessment: TransonicAssessment

    @property
    def loss_change(self) -> float:
        return loss_change(self.inputs, self.stator_loss, self.rotor_loss)

    @property
    def mass_flow_residual(self) -> float:
        return self.meanline.residual

    @property
    def energy_residual(self) -> float:
        return self.meanline.work_euler - self.meanline.work_thermo

    @property
    def balances_converged(self) -> bool:
        return (
            abs(self.mass_flow_residual) < MASS_FLOW_TOLERANCE
            and math.isclose(
                self.meanline.work_euler,
                self.meanline.work_thermo,
                rel_tol=ENERGY_RELATIVE_TOLERANCE,
                abs_tol=ENERGY_ABSOLUTE_TOLERANCE,
            )
        )

    def inputs_with_predicted_losses(self) -> Inputs:
        return inputs_with_losses(self.inputs, self.stator_loss, self.rotor_loss)


def loss_change(inputs: Inputs, stator_loss: RowLoss, rotor_loss: RowLoss) -> float:
    """Largest difference between the loss coefficients used and those predicted."""
    stator_change = abs(stator_loss.kinetic_loss - inputs.loss_kinetic_stator)
    rotor_change = abs(rotor_loss.kinetic_loss - inputs.loss_kinetic_rotor)
    return max(stator_change, rotor_change)


def inputs_with_losses(inputs: Inputs, stator_loss: RowLoss, rotor_loss: RowLoss) -> Inputs:
    return replace(inputs, loss_kinetic_stator=stator_loss.kinetic_loss, loss_kinetic_rotor=rotor_loss.kinetic_loss)


@dataclass(frozen=True)
class Constraint:
    name: str
    value: float
    limit: float
    is_upper_limit: bool = True

    @property
    def satisfied(self) -> bool:
        return self.value <= self.limit if self.is_upper_limit else self.value >= self.limit


@dataclass(frozen=True)
class StageSolution:
    design: StageDesign
    evaluations: tuple[StageEvaluation, ...]  # every pass in order; the last one is the solution
    efficiency: StageEfficiency
    constraints: tuple[Constraint, ...]

    @property
    def final(self) -> StageEvaluation:
        return self.evaluations[-1]

    @property
    def converged(self) -> bool:
        return self.final.loss_change < LOSS_TOLERANCE and self.final.balances_converged

    @property
    def feasible(self) -> bool:
        return self.converged and all(constraint.satisfied for constraint in self.constraints)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
def evaluate_stage(inputs: Inputs, design: StageDesign) -> StageEvaluation:
    meanline = solve_meanline(inputs, design.inlet)
    spanwise_flow = solve_radial_equilibrium(meanline, design.vortex_exponent)
    stator = stator_row(spanwise_flow, design.stator)
    rotor = rotor_row(spanwise_flow, design.rotor)
    return StageEvaluation(
        inputs=inputs,
        meanline=meanline,
        spanwise_flow=spanwise_flow,
        stator=stator,
        rotor=rotor,
        stator_loss=stator_losses(meanline, stator.airfoil_params(), design.stage_index),
        rotor_loss=rotor_losses(meanline, rotor.airfoil_params(), design.stage_index),
        transonic_assessment=assess_transonic(
            meanline,
            spanwise_flow,
            stator.airfoil_params(),
            rotor.airfoil_params(),
            design.stage_index,
        ),
    )


def stage_constraints(meanline: MeanlineResult, shaft_speed: float, hub_reaction: float,
                      chart_warning_count: int,
                      exit_flow_limits: ExitFlowLimits = ExitFlowLimits()) -> tuple[Constraint, ...]:
    """Uncooled limits from the boundary conditions; blade speed is the larger rotor mean-radius value; shaft speed in rpm."""
    constraints = [
        Constraint("Rotor mean blade speed [m/s]", max(meanline.u_2, meanline.u_3),
                   boundary_conditions.max_mean_blade_speed),
        Constraint("Rotor exit AN^2 [m^2 rpm^2]", meanline.area_3 * shaft_speed**2, boundary_conditions.max_AN2),
        Constraint("Hub degree of reaction", hub_reaction, 0.0, is_upper_limit=False),
        Constraint("KO chart extrapolations", chart_warning_count, 0),
    ]
    if exit_flow_limits.max_absolute_exit_mach is not None:
        constraints.append(
            Constraint(
                "Rotor exit absolute Mach",
                meanline.M_3,
                exit_flow_limits.max_absolute_exit_mach,
            )
        )
    if exit_flow_limits.max_absolute_exit_swirl_m_s is not None:
        exit_swirl_m_s = abs(meanline.v_3 * math.sin(math.radians(meanline.alpha_3)))
        constraints.append(
            Constraint(
                "Rotor exit absolute swirl [m/s]",
                exit_swirl_m_s,
                exit_flow_limits.max_absolute_exit_swirl_m_s,
            )
        )
    return tuple(constraints)


def solve_stage(design: StageDesign = StageDesign()) -> StageSolution:
    """Repeat every module until the loss coefficients the meanline uses match the Kacker-Okapuu prediction."""
    evaluations = [evaluate_stage(design.inputs, design)]
    while evaluations[-1].loss_change >= LOSS_TOLERANCE and len(evaluations) < MAX_PASSES:
        evaluations.append(evaluate_stage(evaluations[-1].inputs_with_predicted_losses(), design))

    final = evaluations[-1]
    chart_warnings = final.stator_loss.chart_warnings + final.rotor_loss.chart_warnings
    return StageSolution(
        design=design,
        evaluations=tuple(evaluations),
        efficiency=stage_efficiency(final.meanline, final.rotor_loss),
        constraints=stage_constraints(final.meanline, final.inputs.N_shaft,
                                      final.spanwise_flow.hub.degree_of_reaction, len(chart_warnings),
                                      design.exit_flow_limits),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
STAGE_SUMMARY_ROWS: tuple[tuple[str, Callable[[StageSolution], float], str], ...] = (
    ("Shaft speed [rpm]", lambda solution: solution.final.inputs.N_shaft, ".0f"),
    ("u_2 [m/s]", lambda solution: solution.final.meanline.u_2, ".1f"),
    ("u_3 [m/s]", lambda solution: solution.final.meanline.u_3, ".1f"),
    ("Flow coefficient", lambda solution: solution.final.meanline.flow_coefficient, ".3f"),
    ("Stage loading", lambda solution: solution.final.meanline.stage_loading, ".3f"),
    ("Degree of reaction", lambda solution: solution.final.inputs.degree_of_reaction, ".3f"),
    ("Pressure ratio P_01/P_s3", lambda solution: solution.final.inputs.pressure_ratio, ".3f"),
    ("alpha_1 absolute [deg]", lambda solution: solution.final.meanline.alpha_1, ".2f"),
    ("M_1 absolute", lambda solution: solution.final.meanline.M_1, ".3f"),
    ("alpha_2 absolute [deg]", lambda solution: solution.final.meanline.alpha_2, ".2f"),
    ("beta_2 rotor-relative [deg]", lambda solution: solution.final.meanline.beta_2, ".2f"),
    ("alpha_3 absolute [deg]", lambda solution: solution.final.meanline.alpha_3, ".2f"),
    ("beta_3 rotor-relative [deg]", lambda solution: solution.final.meanline.beta_3, ".2f"),
    ("M_2 absolute", lambda solution: solution.final.meanline.M_2, ".3f"),
    ("M_2rel rotor-relative", lambda solution: solution.final.meanline.M_2rel, ".3f"),
    ("M_3 absolute", lambda solution: solution.final.meanline.M_3, ".3f"),
    ("M_3rel rotor-relative", lambda solution: solution.final.meanline.M_3rel, ".3f"),
    ("Blade height 2 [mm]", lambda solution: solution.final.meanline.blade_height_2 * 1e3, ".2f"),
    ("Blade height 3 [mm]", lambda solution: solution.final.meanline.blade_height_3 * 1e3, ".2f"),
    ("Specific work [kJ/kg]", lambda solution: solution.efficiency.specific_work / 1e3, ".1f"),
    ("eta_tt", lambda solution: solution.efficiency.total_to_total, ".4f"),
    ("eta_ts", lambda solution: solution.efficiency.total_to_static, ".4f"),
    ("Tip clearance debit", lambda solution: solution.efficiency.tip_clearance_debit, ".4f"),
)


def print_stage_report(solution: StageSolution) -> None:
    status = "converged" if solution.converged else "NOT converged"
    print(f"Single-stage loop {status} after {len(solution.evaluations)} passes "
          f"(maximum {MAX_PASSES}; loss tolerance {LOSS_TOLERANCE:g}; "
          f"mass-flow and energy balances checked)")
    print(f"{'Pass':>4}{'xi_stator used':>16}{'predicted':>12}{'xi_rotor used':>16}{'predicted':>12}{'change':>11}")
    for number, evaluation in enumerate(solution.evaluations, start=1):
        print(f"{number:>4}{evaluation.inputs.loss_kinetic_stator:>16.6f}{evaluation.stator_loss.kinetic_loss:>12.6f}"
              f"{evaluation.inputs.loss_kinetic_rotor:>16.6f}{evaluation.rotor_loss.kinetic_loss:>12.6f}"
              f"{evaluation.loss_change:>11.1e}")

    final = solution.final
    print("\nFinal coupled residuals")
    print(f"  Mass-flow residual [kg/s] {final.mass_flow_residual:.3e} "
          f"(limit {MASS_FLOW_TOLERANCE:.1e})")
    print(f"  Euler-energy residual [J/kg] {final.energy_residual:.3e} "
          f"(absolute limit {ENERGY_ABSOLUTE_TOLERANCE:.1e}; "
          f"relative limit {ENERGY_RELATIVE_TOLERANCE:.1e})")
    stator_loss_residual = abs(final.inputs.loss_kinetic_stator - final.stator_loss.kinetic_loss)
    rotor_loss_residual = abs(final.inputs.loss_kinetic_rotor - final.rotor_loss.kinetic_loss)
    print(f"  Stator loss-coefficient residual {stator_loss_residual:.3e}")
    print(f"  Rotor loss-coefficient residual {rotor_loss_residual:.3e}")

    print("\nStage")
    for label, value_of, number_format in STAGE_SUMMARY_ROWS:
        print(f"  {label:<24}{value_of(solution):>14{number_format}}")

    print()
    print_breakdown([solution.final.stator_loss, solution.final.rotor_loss])
    print()
    print_spanwise_table(solution.final.spanwise_flow)
    print()
    print_transonic_assessment(solution.final.transonic_assessment)
    print()
    print_blade_row(solution.final.stator)
    print()
    print_blade_row(solution.final.rotor)

    print()
    print_constraints(solution.constraints)
    print(f"Stage is {'feasible' if solution.feasible else 'NOT feasible'} (cooling and metal temperature not checked)")


def print_constraints(constraints: Sequence[Constraint]) -> None:
    print(f"{'Constraint':<36}{'value':>12}{'limit':>14}  status")
    for constraint in constraints:
        comparison = "<=" if constraint.is_upper_limit else ">="
        status = "ok" if constraint.satisfied else "VIOLATED"
        print(f"{constraint.name:<36}{constraint.value:>12.4g}{comparison:>4}{constraint.limit:>10.4g}  {status}")
