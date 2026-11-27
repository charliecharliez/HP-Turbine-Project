"""Parametric study: best uncooled efficiency at each flow coefficient and degree of reaction, for one and two stages."""
from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from design_optimizer import (MAX_GENERATIONS, POPULATION_SIZE, DesignSpace, MapFunction, ScreenedTurbine,
                              VariableBounds, screen_turbine, search_designs)

Quantity = Literal["flow_coefficient", "degree_of_reaction"]

STAGE_COUNTS = (1, 2)
SWEEPS: dict[Quantity, tuple[float, ...]] = {
    "flow_coefficient": (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
    "degree_of_reaction": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
}
QUANTITY_LABELS: dict[Quantity, str] = {"flow_coefficient": "Flow coefficient  φ",
                                        "degree_of_reaction": "Degree of reaction  R"}
TARGET_WINDOW = 0.004  # width of a swept variable's bounds above its target; rounding to 0.01 returns the target
POINT_POPULATION_SIZE = 10
POINT_GENERATIONS = 150
FIGURE_DIRECTORY = Path(__file__).resolve().parents[1] / "figures"
FIGURE_RESOLUTION = 200  # dpi

# Chart style
SERIES_COLORS = {1: "#2a78d6", 2: "#eb6834"}  # categorical slots 1 and 2, by stage count
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudyPoint:
    stage_count: int
    quantity: Quantity
    target: float
    best: ScreenedTurbine | None  # None when no design met every constraint


@dataclass(frozen=True)
class ParametricStudy:
    optima: dict[int, ScreenedTurbine | None]  # unconstrained optimum of each stage count
    points: tuple[StudyPoint, ...]

    def series(self, stage_count: int, quantity: Quantity) -> list[ScreenedTurbine]:
        """Best designs of one sweep, in order of the swept quantity."""
        designs = [point.best for point in self.points
                   if point.stage_count == stage_count and point.quantity == quantity and point.best]
        return sorted(designs, key=lambda design: stage_average(design, quantity))


def stage_values(turbine: ScreenedTurbine, quantity: Quantity) -> list[float]:
    if quantity == "flow_coefficient":
        return [stage.meanline.flow_coefficient for stage in turbine.stages]
    return [stage.inputs.degree_of_reaction for stage in turbine.stages]


def stage_average(turbine: ScreenedTurbine, quantity: Quantity) -> float:
    values = stage_values(turbine, quantity)
    return sum(values) / len(values)


def efficiency(turbine: ScreenedTurbine) -> float:
    if turbine.performance is None:
        raise ValueError("A screened turbine without performance has no efficiency")
    return turbine.performance.total_to_static


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------
def space_at(stage_count: int, quantity: Quantity, target: float) -> DesignSpace:
    """Design space with every stage held at the target flow coefficient or degree of reaction."""
    target_bounds = (target, target + TARGET_WINDOW)
    if quantity == "flow_coefficient":
        bounds = replace(VariableBounds(), flow_coefficient=target_bounds)
        return DesignSpace(stage_count, bounds=bounds, rotor_exit_setting="flow_coefficient")
    bounds = replace(VariableBounds(), reaction=target_bounds)
    return DesignSpace(stage_count, bounds=bounds)


def best_screened(space: DesignSpace, workers: MapFunction, start: ScreenedTurbine | None,
                  max_generations: int, population_size: int) -> ScreenedTurbine | None:
    """Most efficient screened design meeting every constraint, or None."""
    start_vector = None
    if start is not None:
        start_vector = space.encode(start.design, [stage.meanline for stage in start.stages])
    search = search_designs(space, workers=workers, max_generations=max_generations,
                            population_size=population_size, start=start_vector)
    if search.fun >= 0:
        return None
    return screen_turbine(space.decode(search.x))


def nearest_start(solved: Sequence[StudyPoint], target: float,
                  fallback: ScreenedTurbine | None) -> ScreenedTurbine | None:
    feasible = [point for point in solved if point.best is not None]
    if not feasible:
        return fallback
    return min(feasible, key=lambda point: abs(point.target - target)).best


def sweep(stage_count: int, quantity: Quantity, targets: Sequence[float], optimum: ScreenedTurbine | None,
          workers: MapFunction) -> list[StudyPoint]:
    """Re-optimize at each target, outward from the optimum, each search starting from its nearest solved neighbour."""
    if optimum is None:
        ordered_targets = list(targets)
    else:
        optimum_value = stage_average(optimum, quantity)
        ordered_targets = sorted(targets, key=lambda target: abs(target - optimum_value))

    solved: list[StudyPoint] = []
    for target in ordered_targets:
        best = best_screened(space_at(stage_count, quantity, target), workers,
                             nearest_start(solved, target, optimum), POINT_GENERATIONS, POINT_POPULATION_SIZE)
        solved.append(StudyPoint(stage_count, quantity, target, best))
        outcome = "no feasible design" if best is None else f"eta_ts {efficiency(best):.4f}"
        print(f"  {stage_count} stage(s), {quantity} = {target:.2f}: {outcome}", flush=True)
    return sorted(solved, key=lambda point: point.target)


def run_parametric_study(workers: MapFunction, stage_counts: Sequence[int] = STAGE_COUNTS,
                         sweeps: dict[Quantity, tuple[float, ...]] = SWEEPS) -> ParametricStudy:
    optima = {count: best_screened(DesignSpace(count), workers, None, MAX_GENERATIONS, POPULATION_SIZE)
              for count in stage_counts}
    points = [point
              for count in stage_counts
              for quantity, targets in sweeps.items()
              for point in sweep(count, quantity, targets, optima[count], workers)]
    return ParametricStudy(optima, tuple(points))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def joined_values(turbine: ScreenedTurbine, quantity: Quantity) -> str:
    return "/".join(f"{value:.3f}" for value in stage_values(turbine, quantity))


def print_study(study: ParametricStudy) -> None:
    for count, optimum in study.optima.items():
        summary = "no feasible design" if optimum is None else (
            f"eta_ts {efficiency(optimum):.4f} at phi {joined_values(optimum, 'flow_coefficient')}, "
            f"R {joined_values(optimum, 'degree_of_reaction')}")
        print(f"{count}-stage unconstrained optimum: {summary}")

    print(f"\n{'Stages':>6}  {'Swept':<20}{'target':>8}{'phi':>14}{'R':>14}{'eta_ts':>9}{'eta_tt':>9}{'N [rpm]':>9}")
    for point in study.points:
        best = point.best
        if best is None or best.performance is None:
            print(f"{point.stage_count:>6}  {point.quantity:<20}{point.target:>8.2f}  no design met every constraint")
            continue
        print(f"{point.stage_count:>6}  {point.quantity:<20}{point.target:>8.2f}"
              f"{joined_values(best, 'flow_coefficient'):>14}{joined_values(best, 'degree_of_reaction'):>14}"
              f"{best.performance.total_to_static:>9.4f}{best.performance.total_to_total:>9.4f}"
              f"{best.design.stages[0].inputs.N_shaft:>9.0f}")


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK, labelcolor=SECONDARY_INK, labelsize=9)
    ax.margins(x=0.14)


def plot_series(ax: Axes, study: ParametricStudy, stage_count: int, quantity: Quantity) -> None:
    color = SERIES_COLORS[stage_count]
    designs = study.series(stage_count, quantity)
    if designs:
        x_values = [stage_average(design, quantity) for design in designs]
        y_values = [efficiency(design) for design in designs]
        ax.plot(x_values, y_values, color=color, linewidth=1.6, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        name = f"{stage_count} stage" + ("s" if stage_count > 1 else "")
        ax.annotate(name, (x_values[-1], y_values[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=SECONDARY_INK, fontsize=9)

    optimum = study.optima[stage_count]
    if optimum is not None:
        ax.plot(stage_average(optimum, quantity), efficiency(optimum), linestyle="none", marker="*",
                markersize=14, color=color, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)


def plot_parametric_study(study: ParametricStudy) -> Figure:
    """Best eta_ts against flow coefficient and against degree of reaction, one series per stage count."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, layout="constrained", facecolor=SURFACE)
    for ax, quantity in zip(axes, QUANTITY_LABELS):
        style_axes(ax)
        for stage_count in study.optima:
            plot_series(ax, study, stage_count, quantity)
        ax.set_xlabel(QUANTITY_LABELS[quantity], color=SECONDARY_INK)
    axes[0].set_ylabel("Total-to-static efficiency  η_ts", color=SECONDARY_INK)

    handles = [Line2D([], [], color=SERIES_COLORS[count], linewidth=1.6, marker="o", markersize=6,
                      markeredgecolor=SURFACE, label=f"{count} stage" + ("s" if count > 1 else ""))
               for count in study.optima]
    handles.append(Line2D([], [], linestyle="none", marker="*", markersize=12, color=MUTED_INK,
                          label="Unconstrained optimum (stage average)"))
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    figure.suptitle("Best uncooled η_ts at each flow coefficient and degree of reaction, other variables re-optimized",
                    x=0.01, ha="left", color=INK, fontsize=12)
    return figure


if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        study = run_parametric_study(pool.map)
    print()
    print_study(study)
    FIGURE_DIRECTORY.mkdir(exist_ok=True)
    parametric_figure = plot_parametric_study(study)
    parametric_figure.savefig(FIGURE_DIRECTORY / "parametric_study.png", dpi=FIGURE_RESOLUTION, facecolor=SURFACE)
    plt.show()
