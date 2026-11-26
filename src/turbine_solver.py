"""One- or multi-stage turbine: stages solved in sequence, each starting from the previous stage's exit."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TypeVar

from boundary_conditions import boundary_conditions
from Meanline.gas_props import GAS
from Meanline.solve_meanline import MeanlineResult, StageInlet
from stage_solver import Constraint, StageDesign, StageSolution, print_constraints, print_stage_report, solve_stage

StageResult = TypeVar("StageResult")
SHARE_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Design choices and records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TurbineDesign:
    """Every stage's design choices; each stage's inlet and pressure ratio are set here, overriding its own."""

    stages: tuple[StageDesign, ...] = (StageDesign(),)
    expansion_shares: tuple[float, ...] = (1.0,)  # each stage's share of the overall isentropic enthalpy drop
    inlet: StageInlet = field(default_factory=StageInlet)
    pressure_ratio: float = boundary_conditions.pressure_ratio_TS  # P_01 / P_s3 of the whole turbine

    def __post_init__(self) -> None:
        if len(self.expansion_shares) != len(self.stages):
            raise ValueError(f"{len(self.stages)} stages need as many expansion shares, "
                             f"got {len(self.expansion_shares)}")
        if min(self.expansion_shares) <= 0 or abs(sum(self.expansion_shares) - 1) > SHARE_TOLERANCE:
            raise ValueError(f"Expansion shares must be positive and sum to 1, got {self.expansion_shares}")
        shaft_speeds = {stage.inputs.N_shaft for stage in self.stages}
        if len(shaft_speeds) > 1:
            raise ValueError(f"Every stage turns on the same shaft, got speeds {sorted(shaft_speeds)} rpm")

    def stage_exit_pressures(self) -> list[float]:  # bar
        """Static pressure after each stage once the stages up to it have taken their shares of the expansion."""
        exponent = (GAS.gamma - 1) / GAS.gamma
        overall_drop = 1 - self.pressure_ratio ** -exponent
        exit_pressures = []
        cumulative_share = 0.0
        for share in self.expansion_shares:
            cumulative_share += share
            exit_pressures.append(self.inlet.P_0 * (1 - cumulative_share * overall_drop) ** (1 / exponent))
        return exit_pressures


@dataclass(frozen=True)
class TurbinePerformance:
    specific_work: float  # J/kg, every stage's work after its tip-clearance debit
    total_to_total: float
    total_to_static: float
    work_shares: tuple[float, ...]  # each stage's fraction of the specific work


@dataclass(frozen=True)
class TurbineSolution:
    design: TurbineDesign
    stages: tuple[StageSolution, ...]
    performance: TurbinePerformance

    @property
    def converged(self) -> bool:
        return all(stage.converged for stage in self.stages)

    @property
    def rig_constraints(self) -> tuple[Constraint, ...]:
        return rig_constraints(self.stages[0].final.meanline)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        stage_constraints = tuple(constraint for stage in self.stages for constraint in stage.constraints)
        return stage_constraints + self.rig_constraints

    @property
    def feasible(self) -> bool:
        return self.converged and all(constraint.satisfied for constraint in self.constraints)


def rig_constraints(tested_rotor: MeanlineResult) -> tuple[Constraint, ...]:
    """AMBRA hub-to-tip range at the inlet and exit of the rotor it tests, the first one."""
    hub_tip_ratios = {
        "inlet": tested_rotor.D_hub_2 / tested_rotor.D_tip_2,
        "exit": tested_rotor.D_hub_3 / tested_rotor.D_tip_3,
    }
    constraints = []
    for location, ratio in hub_tip_ratios.items():
        constraints.append(Constraint(f"AMBRA rotor {location} hub/tip", ratio,
                                      boundary_conditions.rig_min_hub_tip_ratio, is_upper_limit=False))
        constraints.append(Constraint(f"AMBRA rotor {location} hub/tip", ratio,
                                      boundary_conditions.rig_max_hub_tip_ratio))
    return tuple(constraints)


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------
def stage_in_turbine(stage: StageDesign, inlet: StageInlet, exit_pressure: float) -> StageDesign:
    inputs = replace(stage.inputs, pressure_ratio=inlet.P_0 / exit_pressure)
    return replace(stage, inlet=inlet, inputs=inputs)


def solve_stages_in_sequence(design: TurbineDesign, solve: Callable[[StageDesign], StageResult],
                             meanline_of: Callable[[StageResult], MeanlineResult]) -> list[StageResult]:
    """Solve each stage from the previous stage's exit state down to its own exit pressure."""
    inlet = design.inlet
    results = []
    for stage, exit_pressure in zip(design.stages, design.stage_exit_pressures()):
        result = solve(stage_in_turbine(stage, inlet, exit_pressure))
        results.append(result)
        inlet = meanline_of(result).exit_state()
    return results


def turbine_performance(inlet: StageInlet, meanlines: Sequence[MeanlineResult],
                        stage_work: Sequence[float]) -> TurbinePerformance:
    """Overall efficiencies from the turbine inlet to the last stage's exit; stage work in J/kg."""
    exponent = (GAS.gamma - 1) / GAS.gamma
    exit_state = meanlines[-1]
    specific_work = math.fsum(stage_work)
    ideal_work_static = GAS.C_p * inlet.T_0 * (1 - (exit_state.P_s3 / inlet.P_0) ** exponent)
    ideal_work_total = GAS.C_p * inlet.T_0 * (1 - (exit_state.P_03 / inlet.P_0) ** exponent)
    return TurbinePerformance(
        specific_work=specific_work,
        total_to_total=specific_work / ideal_work_total,
        total_to_static=specific_work / ideal_work_static,
        work_shares=tuple(work / specific_work for work in stage_work),
    )


def solve_turbine(design: TurbineDesign = TurbineDesign()) -> TurbineSolution:
    stages = solve_stages_in_sequence(design, solve_stage, lambda solution: solution.final.meanline)
    performance = turbine_performance(design.inlet, [stage.final.meanline for stage in stages],
                                      [stage.efficiency.specific_work for stage in stages])
    return TurbineSolution(design, tuple(stages), performance)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_turbine_report(solution: TurbineSolution) -> None:
    stage_count = len(solution.stages)
    for number, stage in enumerate(solution.stages, start=1):
        print(f"{'=' * 30} Stage {number} of {stage_count} {'=' * 30}")
        print_stage_report(stage)
        print()

    performance = solution.performance
    shares = ", ".join(f"{share:.3f}" for share in performance.work_shares)
    expansion = ", ".join(f"{share:.3f}" for share in solution.design.expansion_shares)
    print(f"{'=' * 30} Turbine {'=' * 30}")
    print(f"  {'Stages':<36}{stage_count:>12}")
    print(f"  {'Expansion shares (design)':<36}{expansion:>12}")
    print(f"  {'Work shares (result)':<36}{shares:>12}")
    print(f"  {'Specific work [kJ/kg]':<36}{performance.specific_work / 1e3:>12.1f}")
    print(f"  {'Power [MW]':<36}{performance.specific_work * solution.design.inlet.m_dot / 1e6:>12.2f}")
    print(f"  {'eta_tt':<36}{performance.total_to_total:>12.4f}")
    print(f"  {'eta_ts':<36}{performance.total_to_static:>12.4f}")
    print()
    print_constraints(solution.rig_constraints)
    status = "feasible" if solution.feasible else "NOT feasible"
    print(f"Turbine is {status} (cooling and metal temperature not checked)")
