"""Differential-evolution optimization of one- and two-stage turbines: screened fast, then verified with the full loop."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import differential_evolution

from Pipeline.Airfoil.airfoil import estimated_airfoil_params, stagger_chart_covers
from Pipeline.Losses.losses_solver import StageEfficiency, rotor_losses, stage_efficiency, stator_losses
from Pipeline.Meanline.solve_meanline import Inputs, MeanlineResult, solve_meanline
from Pipeline.RadialEquilibrium.radial_equilibrium_solver import solve_radial_equilibrium
from Helpers.design_space import ExitFlowLimits, VariableBounds
from Helpers.stage_solver import (LOSS_TOLERANCE, MAX_PASSES, Constraint, StageDesign, inputs_with_losses, loss_change,
                          stage_constraints)
from Helpers.turbine_solver import (TurbineDesign, TurbinePerformance, TurbineSolution, rig_constraints,
                            solve_stages_in_sequence, solve_turbine, turbine_performance)

MODEL_ERRORS = (ValueError, ArithmeticError)
RADIUS_DECIMALS = 4  # m, radii rounded down so blade speeds never exceed their targets
SHAFT_SPEED_DECIMALS = -1  # rpm
REACTION_DECIMALS = 2
ANGLE_DECIMALS = 1  # deg
SHARE_DECIMALS = 2
VARIABLES_PER_STAGE = 4  # radius ratio, reaction, stator exit angle, rotor exit angle
MODEL_FAILURE_PENALTY = 10.0
INFEASIBLE_OFFSET = 1.0  # ranks every infeasible design below every feasible one
POPULATION_SIZE = 15  # multiplier of the number of variables
MAX_GENERATIONS = 300
CONVERGENCE_TOLERANCE = 1e-4
CANDIDATES_TO_VERIFY = 6
FIRST_GUESS = Inputs(loss_kinetic_stator=0.08, loss_kinetic_rotor=0.10)


@dataclass(frozen=True)
class DesignSpace:
    """Maps a vector of design variables to a turbine whose mean line runs on from stage to stage."""

    stage_count: int
    bounds: VariableBounds = field(default_factory=VariableBounds)
    base_stage: StageDesign = field(default_factory=lambda: StageDesign(inputs=FIRST_GUESS))
    exit_flow_limits: ExitFlowLimits | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, VariableBounds):
            raise ValueError("bounds must be a VariableBounds instance")
        if self.exit_flow_limits is not None and not isinstance(self.exit_flow_limits, ExitFlowLimits):
            raise ValueError("exit_flow_limits must be None or an ExitFlowLimits instance")
        if (
            isinstance(self.stage_count, bool)
            or not isinstance(self.stage_count, int)
            or self.stage_count not in (1, 2)
        ):
            raise ValueError(f"stage_count must be 1 or 2, got {self.stage_count!r}")
        if self.stage_count == 2:
            lower, upper = self.bounds.expansion_share
            feasible_lower = max(lower, 1.0 - upper)
            feasible_upper = min(upper, 1.0 - lower)
            if feasible_lower >= feasible_upper:
                raise ValueError("expansion_share bounds cannot produce a valid two-stage split")

    def scipy_bounds(self) -> list[tuple[float, float]]:
        bounds = self.bounds
        per_stage = [bounds.radius_ratio, bounds.reaction, bounds.stator_exit_angle, bounds.rotor_exit_angle]
        share_bounds = bounds.expansion_share
        if self.stage_count == 2:
            lower, upper = share_bounds
            share_bounds = (max(lower, 1.0 - upper), min(upper, 1.0 - lower))
        shares = [share_bounds] * (self.stage_count - 1)
        return [bounds.shaft_speed, bounds.blade_speed, *per_stage * self.stage_count, *shares]

    def decode(self, variables: Sequence[float]) -> TurbineDesign:
        """Build a turbine from an in-range variable vector at practical resolution."""
        expected_length = 2 + VARIABLES_PER_STAGE * self.stage_count + self.stage_count - 1
        if len(variables) != expected_length:
            raise ValueError(f"expected {expected_length} design variables, got {len(variables)}")
        candidate_bounds = self.scipy_bounds()
        values = []
        for index, value in enumerate(variables):
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"design variable {index} must be a finite number")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"design variable {index} must be a finite number")
            lower, upper = candidate_bounds[index]
            if not lower <= numeric_value <= upper:
                raise ValueError(
                    f"design variable {index} must be within [{lower}, {upper}], got {numeric_value}"
                )
            values.append(numeric_value)

        shaft_speed = round(values[0], SHAFT_SPEED_DECIMALS)
        blade_speed = values[1]
        stage_variables = np.reshape(
            values[2 : 2 + VARIABLES_PER_STAGE * self.stage_count],
            (self.stage_count, VARIABLES_PER_STAGE),
        )
        leading_shares = [
            round(share, SHARE_DECIMALS)
            for share in values[2 + VARIABLES_PER_STAGE * self.stage_count :]
        ]
        omega = shaft_speed * 2 * math.pi / 60

        stages = []
        rotor_exit_radius = round_down(blade_speed / omega, RADIUS_DECIMALS)
        for index, (radius_ratio, reaction, alpha_2, beta_3) in enumerate(stage_variables):
            if index == 0:
                stator_exit_radius = round_down(rotor_exit_radius / radius_ratio, RADIUS_DECIMALS)
            else:
                stator_exit_radius = rotor_exit_radius
                rotor_exit_radius = round_down(stator_exit_radius * radius_ratio, RADIUS_DECIMALS)
            inputs = replace(self.base_stage.inputs, N_shaft=shaft_speed, radius_2=stator_exit_radius,
                             radius_3=rotor_exit_radius,
                             degree_of_reaction=round(float(reaction), REACTION_DECIMALS),
                             alpha_2=round(float(alpha_2), ANGLE_DECIMALS),
                             beta_3=round(float(beta_3), ANGLE_DECIMALS))
            stage = replace(self.base_stage, inputs=inputs)
            if self.exit_flow_limits is not None:
                stage = replace(stage, exit_flow_limits=self.exit_flow_limits)
            stages.append(stage)

        shares = (*leading_shares, round(1.0 - math.fsum(leading_shares), SHARE_DECIMALS))
        if self.stage_count == 2 and any(
            not self.bounds.expansion_share[0] <= share <= self.bounds.expansion_share[1]
            for share in shares
        ):
            raise ValueError(f"decoded expansion shares are outside their bounds: {shares}")
        return TurbineDesign(stages=tuple(stages), expansion_shares=shares)


def round_down(value: float, decimals: int) -> float:
    factor = 10**decimals
    return math.floor(value * factor + 1e-9) / factor


def design_key(design: TurbineDesign) -> tuple[float, ...]:
    stage_values = [(stage.inputs.N_shaft, stage.inputs.radius_2, stage.inputs.radius_3,
                     stage.inputs.degree_of_reaction, stage.inputs.alpha_2, stage.inputs.beta_3)
                    for stage in design.stages]
    return tuple(round(value, 6) for values in stage_values for value in values) + tuple(
        round(share, 6) for share in design.expansion_shares)


# ---------------------------------------------------------------------------
# Screening with estimated blade geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScreenedStage:
    inputs: Inputs  # with the loss coefficients converged against estimated blade geometry
    meanline: MeanlineResult
    efficiency: StageEfficiency
    constraints: tuple[Constraint, ...]


@dataclass(frozen=True)
class ScreenedTurbine:
    design: TurbineDesign
    stages: tuple[ScreenedStage, ...] = ()
    performance: TurbinePerformance | None = None
    error: str = ""

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        if not self.stages:
            return ()
        stage_constraints = tuple(constraint for stage in self.stages for constraint in stage.constraints)
        return stage_constraints + rig_constraints(self.stages[0].meanline)

    @property
    def feasible(self) -> bool:
        return not self.error and all(constraint.satisfied for constraint in self.constraints)


def screen_stage(design: StageDesign) -> ScreenedStage:
    """Meanline-loss loop with estimated blade geometry; raises ValueError on model failures and stagger-chart gaps."""
    inputs = design.inputs
    for _ in range(MAX_PASSES):
        meanline = solve_meanline(inputs, design.inlet)
        stator_params = estimated_airfoil_params(design.stator, meanline.alpha_1, meanline.alpha_2)
        rotor_params = estimated_airfoil_params(design.rotor, meanline.beta_2, meanline.beta_3)
        stator_loss = stator_losses(meanline, stator_params)
        rotor_loss = rotor_losses(meanline, rotor_params)
        if loss_change(inputs, stator_loss, rotor_loss) < LOSS_TOLERANCE:
            break
        inputs = inputs_with_losses(inputs, stator_loss, rotor_loss)
    else:
        raise ValueError("Screening loss loop did not converge")

    spanwise_flow = solve_radial_equilibrium(meanline, design.vortex_exponent)
    for span in (spanwise_flow.hub, spanwise_flow.tip):
        stator_covered = stagger_chart_covers(meanline.alpha_1, span.stator_exit.absolute_angle)
        rotor_covered = stagger_chart_covers(span.stator_exit.relative_angle, span.rotor_exit.relative_angle)
        if not (stator_covered and rotor_covered):
            raise ValueError(f"Stagger chart: span fraction {span.span_fraction:g} outside KO Fig. 5")

    chart_warning_count = len(stator_loss.chart_warnings + rotor_loss.chart_warnings)
    return ScreenedStage(
        inputs=inputs,
        meanline=meanline,
        efficiency=stage_efficiency(meanline, rotor_loss),
        constraints=stage_constraints(meanline, inputs.N_shaft, spanwise_flow.hub.degree_of_reaction,
                                      chart_warning_count, design.exit_flow_limits),
    )


def screen_turbine(design: TurbineDesign) -> ScreenedTurbine:
    """Fast estimate of a turbine; model failures are recorded instead of raised."""
    try:
        stages = solve_stages_in_sequence(design, screen_stage, lambda stage: stage.meanline)
    except MODEL_ERRORS as error:
        return ScreenedTurbine(design, error=f"{type(error).__name__}: {error}")
    performance = turbine_performance(design.inlet, [stage.meanline for stage in stages],
                                      [stage.efficiency.specific_work for stage in stages])
    return ScreenedTurbine(design, tuple(stages), performance)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def constraint_violation(constraint: Constraint) -> float:
    """Amount by which a constraint is exceeded, relative to its limit."""
    excess = constraint.value - constraint.limit if constraint.is_upper_limit else constraint.limit - constraint.value
    return max(0.0, excess) / max(abs(constraint.limit), 1.0)


def penalized_objective(variables: np.ndarray, space: DesignSpace) -> float:
    """Negative screened eta_ts for feasible designs; penalties above zero otherwise."""
    try:
        design = space.decode(variables)
    except ValueError:
        return MODEL_FAILURE_PENALTY
    screened = screen_turbine(design)
    if screened.error or screened.performance is None:
        return MODEL_FAILURE_PENALTY
    violation = math.fsum(constraint_violation(constraint) for constraint in screened.constraints)
    if violation > 0:
        return INFEASIBLE_OFFSET + violation
    return -screened.performance.total_to_static


@dataclass(frozen=True)
class OptimizationResult:
    space: DesignSpace
    evaluations: int
    generations: int
    message: str
    verified: tuple[TurbineSolution, ...]  # full-loop solutions of the best distinct screened-feasible designs
    verification_errors: tuple[tuple[TurbineDesign, str], ...]
    screened_efficiency: dict[tuple[float, ...], float]  # screened eta_ts of each verified design, by design key

    @property
    def selected(self) -> TurbineSolution | None:
        """Most efficient design the full loop confirms converged and within every constraint."""
        feasible = [solution for solution in self.verified if solution.feasible]
        return max(feasible, key=lambda solution: solution.performance.total_to_static, default=None)


def best_distinct_designs(space: DesignSpace, population: np.ndarray, energies: np.ndarray,
                          count: int) -> list[tuple[TurbineDesign, float]]:
    """Screened-feasible designs of the final population, best first, without repeats."""
    designs: list[tuple[TurbineDesign, float]] = []
    seen: set[tuple[float, ...]] = set()
    for index in np.argsort(energies):
        if energies[index] >= 0 or len(designs) == count:
            break
        design = space.decode(population[index])
        key = design_key(design)
        if key not in seen:
            seen.add(key)
            designs.append((design, -float(energies[index])))
    return designs


def optimize_turbine(space: DesignSpace, seed: int = 1, workers: int = 1,
                     max_generations: int = MAX_GENERATIONS) -> OptimizationResult:
    """Maximize screened eta_ts by differential evolution, then verify the best designs with the full loop."""
    search = differential_evolution(
        penalized_objective, space.scipy_bounds(), args=(space,), seed=seed, workers=workers,
        updating="deferred", popsize=POPULATION_SIZE, maxiter=max_generations, tol=CONVERGENCE_TOLERANCE,
        polish=False,
    )

    verified: list[TurbineSolution] = []
    verification_errors: list[tuple[TurbineDesign, str]] = []
    screened_efficiency: dict[tuple[float, ...], float] = {}
    for design, efficiency in best_distinct_designs(space, search.population, search.population_energies,
                                                    CANDIDATES_TO_VERIFY):
        screened_efficiency[design_key(design)] = efficiency
        try:
            verified.append(solve_turbine(design))
        except MODEL_ERRORS as error:
            verification_errors.append((design, f"{type(error).__name__}: {error}"))
    return OptimizationResult(space, search.nfev, search.nit, search.message, tuple(verified),
                              tuple(verification_errors), screened_efficiency)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
StageColumn = tuple[str, Callable[[TurbineSolution, int], float], str]
STAGE_COLUMNS: tuple[StageColumn, ...] = (
    ("u [m/s]", lambda solution, index: max(solution.stages[index].final.meanline.u_2,
                                            solution.stages[index].final.meanline.u_3), ".1f"),
    ("phi", lambda solution, index: solution.stages[index].final.meanline.flow_coefficient, ".3f"),
    ("psi", lambda solution, index: solution.stages[index].final.meanline.stage_loading, ".3f"),
    ("R", lambda solution, index: solution.stages[index].final.inputs.degree_of_reaction, ".3f"),
    ("alpha_2", lambda solution, index: solution.stages[index].final.meanline.alpha_2, ".2f"),
    ("beta_3", lambda solution, index: solution.stages[index].final.meanline.beta_3, ".2f"),
    ("alpha_3", lambda solution, index: solution.stages[index].final.meanline.alpha_3, ".2f"),
    ("M_2", lambda solution, index: solution.stages[index].final.meanline.M_2, ".3f"),
    ("M_3rel", lambda solution, index: solution.stages[index].final.meanline.M_3rel, ".3f"),
    ("work", lambda solution, index: solution.performance.work_shares[index], ".3f"),
    ("h_3 [mm]", lambda solution, index: solution.stages[index].final.meanline.blade_height_3 * 1e3, ".1f"),
    ("vanes", lambda solution, index: solution.stages[index].final.stator.blade_count, ".0f"),
    ("blades", lambda solution, index: solution.stages[index].final.rotor.blade_count, ".0f"),
)


def print_optimization_summary(result: OptimizationResult) -> None:
    stage_count = result.space.stage_count
    print(f"{stage_count}-stage optimization: {result.evaluations} screened designs over {result.generations} "
          f"generations ({result.message})")
    print(f"{'N [rpm]':>9}{'screened':>10}{'eta_ts':>9}{'eta_tt':>9}  status")
    for solution in result.verified:
        problems = [constraint.name for constraint in solution.constraints if not constraint.satisfied]
        if not solution.converged:
            problems.insert(0, "not converged")
        status = "feasible" if solution.feasible else "; ".join(problems)
        screened = result.screened_efficiency[design_key(solution.design)]
        print(f"{solution.design.stages[0].inputs.N_shaft:>9.0f}{screened:>10.4f}"
              f"{solution.performance.total_to_static:>9.4f}{solution.performance.total_to_total:>9.4f}  {status}")
    for design, message in result.verification_errors:
        print(f"{design.stages[0].inputs.N_shaft:>9.0f}{'':>28}  {message}")

    selected = result.selected
    if selected is None:
        print("No verified design satisfies every constraint")
        return
    print(f"Selected: expansion shares {tuple(round(share, 4) for share in selected.design.expansion_shares)}")
    for number, stage in enumerate(selected.design.stages, start=1):
        print(f"  stage {number}: {stage.inputs}")


def print_stage_count_comparison(results: Sequence[OptimizationResult]) -> None:
    print(f"{'Stages':>6}{'eta_ts':>9}{'eta_tt':>9}{'work [kJ/kg]':>14}{'N [rpm]':>9}")
    for result in results:
        selected = result.selected
        if selected is None:
            print(f"{result.space.stage_count:>6}  no feasible design")
            continue
        performance = selected.performance
        print(f"{result.space.stage_count:>6}{performance.total_to_static:>9.4f}{performance.total_to_total:>9.4f}"
              f"{performance.specific_work / 1e3:>14.1f}{selected.design.stages[0].inputs.N_shaft:>9.0f}")

    print(f"\n{'Stage':>7}" + "".join(f"{label:>10}" for label, _, _ in STAGE_COLUMNS))
    for result in results:
        selected = result.selected
        if selected is None:
            continue
        for index in range(len(selected.stages)):
            cells = "".join(f"{value_of(selected, index):>10{number_format}}"
                            for _, value_of, number_format in STAGE_COLUMNS)
            print(f"{f'{len(selected.stages)}.{index + 1}':>7}{cells}")


if __name__ == "__main__":
    results = []
    for stage_count in (1, 2):
        result = optimize_turbine(DesignSpace(stage_count), workers=-1)
        print_optimization_summary(result)
        print()
        results.append(result)
    print_stage_count_comparison(results)
