"""Single-stage design sweep: screen a grid with estimated blade geometry, then verify the best with the full loop."""
from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace

from Pipeline.Airfoil.airfoil import estimated_airfoil_params, stagger_chart_covers
from Pipeline.Losses.losses_solver import rotor_losses, stage_efficiency, stator_losses
from Pipeline.Meanline.solve_meanline import Inputs, solve_meanline
from Pipeline.RadialEquilibrium.radial_equilibrium_solver import solve_radial_equilibrium
from Helpers.design_space import VariableBounds
from Helpers.stage_solver import LOSS_TOLERANCE, MAX_PASSES, Constraint, StageDesign, StageSolution, solve_stage, stage_constraints

MODEL_ERRORS = (ValueError, ArithmeticError)
RADIUS_RESOLUTION = 1e-4  # m, radii rounded down so the blade speed never exceeds its target
REFINEMENT_STEPS = {"blade_speed": 10.0, "shaft_speed": 1000.0, "radius_ratio": 0.05, "reaction": 0.05,
                    "angle": 1.0}  # m/s, rpm, -, -, deg
MAX_REFINEMENTS = 8
MIN_IMPROVEMENT = 1e-3  # eta_ts gain below which a refinement is not adopted


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepGrid:
    """Candidate values of each swept variable; both mean radii follow from the blade speed and shaft speed."""

    blade_speeds: tuple[float, ...] = (450.0, 475.0, 500.0)  # m/s, at the larger rotor mean radius
    shaft_speeds: tuple[float, ...] = (14000.0, 16000.0, 18000.0, 20000.0)  # rpm
    radius_ratios: tuple[float, ...] = (1.0, 1.075, 1.15)  # radius_3 / radius_2
    reactions: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5)
    stator_exit_angles: tuple[float, ...] = (66.0, 69.0, 72.0, 75.0, 78.0)  # deg
    rotor_exit_angles: tuple[float, ...] = (-55.0, -59.0, -63.0, -67.0, -71.0)  # deg
    bounds: VariableBounds = field(default_factory=VariableBounds)

    def __post_init__(self) -> None:
        variable_ranges = (
            ("blade_speeds", self.blade_speeds, self.bounds.blade_speed),
            ("shaft_speeds", self.shaft_speeds, self.bounds.shaft_speed),
            ("radius_ratios", self.radius_ratios, self.bounds.radius_ratio),
            ("reactions", self.reactions, self.bounds.reaction),
            ("stator_exit_angles", self.stator_exit_angles, self.bounds.stator_exit_angle),
            ("rotor_exit_angles", self.rotor_exit_angles, self.bounds.rotor_exit_angle),
        )
        for name, values, (lower, upper) in variable_ranges:
            if not values:
                raise ValueError(f"{name} must contain at least one candidate value")
            for value in values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{name} values must be finite numbers")
                if not lower <= value <= upper:
                    raise ValueError(f"{name} value {value} is outside [{lower}, {upper}]")

    def designs(self, base: StageDesign) -> Iterator[StageDesign]:
        combinations = itertools.product(self.blade_speeds, self.shaft_speeds, self.radius_ratios, self.reactions,
                                         self.stator_exit_angles, self.rotor_exit_angles)
        for blade_speed, shaft_speed, radius_ratio, reaction, alpha_2, beta_3 in combinations:
            radius_3 = round_down(blade_speed / (shaft_speed * 2 * math.pi / 60), RADIUS_RESOLUTION)
            radius_2 = round_down(radius_3 / radius_ratio, RADIUS_RESOLUTION)
            inputs = replace(base.inputs, N_shaft=shaft_speed, radius_2=radius_2, radius_3=radius_3,
                             degree_of_reaction=reaction, alpha_2=alpha_2, beta_3=beta_3)
            yield replace(base, inputs=inputs)

    @classmethod
    def around(cls, inputs: Inputs, bounds: VariableBounds = VariableBounds()) -> SweepGrid:
        """Grid of one step either side of a design, with the blade speed held at or below its limit."""
        omega = inputs.N_shaft * 2 * math.pi / 60
        blade_speed = omega * max(inputs.radius_2, inputs.radius_3)
        radius_ratio = inputs.radius_3 / inputs.radius_2
        return cls(
            blade_speeds=bounded_neighbors(blade_speed, REFINEMENT_STEPS["blade_speed"], bounds.blade_speed),
            shaft_speeds=bounded_neighbors(inputs.N_shaft, REFINEMENT_STEPS["shaft_speed"], bounds.shaft_speed),
            radius_ratios=bounded_neighbors(radius_ratio, REFINEMENT_STEPS["radius_ratio"], bounds.radius_ratio),
            reactions=bounded_neighbors(inputs.degree_of_reaction, REFINEMENT_STEPS["reaction"], bounds.reaction),
            stator_exit_angles=bounded_neighbors(
                inputs.alpha_2, REFINEMENT_STEPS["angle"], bounds.stator_exit_angle
            ),
            rotor_exit_angles=bounded_neighbors(
                inputs.beta_3, REFINEMENT_STEPS["angle"], bounds.rotor_exit_angle
            ),
            bounds=bounds,
        )


def round_down(value: float, resolution: float) -> float:
    return math.floor(value / resolution + 1e-9) * resolution


def bounded_neighbors(center: float, step: float, bounds: tuple[float, float]) -> tuple[float, ...]:
    lower, upper = bounds
    return tuple(sorted({max(lower, min(upper, value)) for value in (center - step, center, center + step)}))


# ---------------------------------------------------------------------------
# Screening and verification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScreenedDesign:
    """Fast estimate of one candidate; its design carries the loss coefficients converged with estimated geometry."""

    design: StageDesign
    total_to_static: float = math.nan
    constraints: tuple[Constraint, ...] = ()
    error: str = ""

    @property
    def feasible(self) -> bool:
        return not self.error and all(constraint.satisfied for constraint in self.constraints)


@dataclass(frozen=True)
class SweepResult:
    screened: tuple[ScreenedDesign, ...]
    verified: tuple[StageSolution, ...]  # full-loop solutions of the best screened candidates, in screening order
    verification_errors: tuple[tuple[StageDesign, str], ...] = field(default=())

    @property
    def selected(self) -> StageSolution | None:
        """Most efficient design that the full loop confirms converged and satisfies every constraint."""
        feasible = [solution for solution in self.verified if solution.feasible]
        return max(feasible, key=lambda solution: solution.efficiency.total_to_static, default=None)


def screen_design(design: StageDesign) -> ScreenedDesign:
    """Meanline-loss loop with estimated blade geometry; model failures are recorded instead of raised."""
    inputs = design.inputs
    try:
        for _ in range(MAX_PASSES):
            meanline = solve_meanline(inputs, design.inlet)
            stator_params = estimated_airfoil_params(design.stator, meanline.alpha_1, meanline.alpha_2)
            rotor_params = estimated_airfoil_params(design.rotor, meanline.beta_2, meanline.beta_3)
            stator_loss = stator_losses(meanline, stator_params)
            rotor_loss = rotor_losses(meanline, rotor_params)
            stator_change = abs(stator_loss.kinetic_loss - inputs.loss_kinetic_stator)
            rotor_change = abs(rotor_loss.kinetic_loss - inputs.loss_kinetic_rotor)
            if max(stator_change, rotor_change) < LOSS_TOLERANCE:
                break
            inputs = replace(inputs, loss_kinetic_stator=stator_loss.kinetic_loss,
                             loss_kinetic_rotor=rotor_loss.kinetic_loss)
        else:
            return ScreenedDesign(design, error="loss loop did not converge")
        spanwise_flow = solve_radial_equilibrium(meanline, design.vortex_exponent)
        hub, tip = spanwise_flow.hub, spanwise_flow.tip
    except MODEL_ERRORS as error:
        return ScreenedDesign(design, error=f"{type(error).__name__}: {error}")

    for span in (hub, tip):
        stator_covered = stagger_chart_covers(meanline.alpha_1, span.stator_exit.absolute_angle)
        rotor_covered = stagger_chart_covers(span.stator_exit.relative_angle, span.rotor_exit.relative_angle)
        if not (stator_covered and rotor_covered):
            return ScreenedDesign(design, error=f"Stagger chart: span fraction {span.span_fraction:g} outside KO Fig. 5")
    hub_reaction = hub.degree_of_reaction

    chart_warning_count = len(stator_loss.chart_warnings + rotor_loss.chart_warnings)
    return ScreenedDesign(
        design=replace(design, inputs=inputs),
        total_to_static=stage_efficiency(meanline, rotor_loss).total_to_static,
        constraints=stage_constraints(
            meanline,
            inputs.N_shaft,
            hub_reaction,
            chart_warning_count,
            design.exit_flow_limits,
        ),
    )


def run_sweep(grid: SweepGrid, base: StageDesign, candidates_to_verify: int = 10) -> SweepResult:
    """Screen every grid design, then run the full loop on the most efficient screened-feasible candidates."""
    screened = [screen_design(design) for design in grid.designs(base)]
    ranked = sorted((candidate for candidate in screened if candidate.feasible),
                    key=lambda candidate: candidate.total_to_static, reverse=True)

    verified: list[StageSolution] = []
    verification_errors: list[tuple[StageDesign, str]] = []
    for candidate in ranked[:candidates_to_verify]:
        try:
            verified.append(solve_stage(candidate.design))
        except MODEL_ERRORS as error:
            verification_errors.append((candidate.design, f"{type(error).__name__}: {error}"))
    return SweepResult(tuple(screened), tuple(verified), tuple(verification_errors))


def design_key(inputs: Inputs) -> tuple[float, ...]:
    return (inputs.N_shaft, round(inputs.radius_2, 6), round(inputs.radius_3, 6), round(inputs.degree_of_reaction, 6),
            round(inputs.alpha_2, 6), round(inputs.beta_3, 6))


def refine_sweep(
    result: SweepResult,
    base: StageDesign,
    bounds: VariableBounds = VariableBounds(),
) -> list[SweepResult]:
    """Sweep one step around the selected design while that raises eta_ts by at least MIN_IMPROVEMENT."""
    sweeps = [result]
    for _ in range(MAX_REFINEMENTS):
        selected = sweeps[-1].selected
        if selected is None:
            break
        refined = run_sweep(
            SweepGrid.around(selected.final.inputs, bounds),
            replace(base, inputs=selected.final.inputs),
        )
        if refined.selected is None or design_key(refined.selected.final.inputs) == design_key(selected.final.inputs):
            break
        gain = refined.selected.efficiency.total_to_static - selected.efficiency.total_to_static
        if gain < MIN_IMPROVEMENT:
            break
        sweeps.append(refined)
    return sweeps


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
DESIGN_COLUMNS: tuple[tuple[str, Callable[[Inputs], float], str], ...] = (
    ("N [rpm]", lambda inputs: inputs.N_shaft, ".0f"),
    ("r_2 [m]", lambda inputs: inputs.radius_2, ".4f"),
    ("r_3 [m]", lambda inputs: inputs.radius_3, ".4f"),
    ("R", lambda inputs: inputs.degree_of_reaction, ".2f"),
    ("alpha_2", lambda inputs: inputs.alpha_2, ".1f"),
    ("beta_3", lambda inputs: inputs.beta_3, ".1f"),
)


def design_cells(inputs: Inputs) -> str:
    return "".join(f"{value_of(inputs):>9{number_format}}" for _, value_of, number_format in DESIGN_COLUMNS)


def print_sweep_summary(result: SweepResult, rows: int = 10) -> None:
    screened = result.screened
    errors = [candidate for candidate in screened if candidate.error]
    violations = Counter(constraint.name for candidate in screened for constraint in candidate.constraints
                         if not constraint.satisfied)
    feasible = sorted((candidate for candidate in screened if candidate.feasible),
                      key=lambda candidate: candidate.total_to_static, reverse=True)

    print(f"Screened {len(screened)} designs: {len(feasible)} feasible, {len(errors)} failed in a model")
    for name, count in violations.most_common():
        print(f"  {count:>6} violate {name}")
    for message, count in Counter(candidate.error.split(":")[0] for candidate in errors).most_common():
        print(f"  {count:>6} raised {message}")

    header = "".join(f"{label:>9}" for label, _, _ in DESIGN_COLUMNS)
    print(f"\nBest screened designs (estimated blade geometry)\n{header}{'eta_ts':>9}")
    for candidate in feasible[:rows]:
        print(f"{design_cells(candidate.design.inputs)}{candidate.total_to_static:>9.4f}")

    print(f"\nFull-loop verification\n{header}{'eta_ts':>9}{'passes':>8}  status")
    for solution in result.verified:
        problems = [constraint.name for constraint in solution.constraints if not constraint.satisfied]
        if not solution.converged:
            problems.insert(0, "not converged")
        status = "feasible" if solution.feasible else "; ".join(problems)
        print(f"{design_cells(solution.final.inputs)}{solution.efficiency.total_to_static:>9.4f}"
              f"{len(solution.evaluations):>8}  {status}")
    for design, message in result.verification_errors:
        print(f"{design_cells(design.inputs)}{'':>17}  {message}")

    selected = result.selected
    if selected is None:
        print("\nNo verified design satisfies every constraint")
    else:
        print(f"\nSelected design: {selected.final.inputs}")


if __name__ == "__main__":
    sweep_base = StageDesign(inputs=Inputs(loss_kinetic_stator=0.08, loss_kinetic_rotor=0.08))
    coarse = run_sweep(SweepGrid(), sweep_base)
    print_sweep_summary(coarse)
    sweeps = refine_sweep(coarse, sweep_base)
    for number, sweep in enumerate(sweeps[1:], start=1):
        print(f"\n=== Refinement {number} ===")
        print_sweep_summary(sweep, rows=5)
    print(f"\nRefinement stopped after {len(sweeps) - 1} adopted steps (next gain below {MIN_IMPROVEMENT:g} in eta_ts)")
