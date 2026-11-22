"""Suggested blade stagger angle from Kacker & Okapuu (1982) Fig. 5 for given inlet and outlet blade angles."""
from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from Empirical.stagger_serializer import JSON_PATH, read_digitized_curves, serialize_polynomials

__all__ = ["BETA_1_MIN", "beta_1_max", "beta_2_range", "plot_stagger_chart", "stagger_angle"]

BETA_1_MIN = -30.0  # deg

_CURVE_COLOR = "#2b2b2b"
_POINT_COLOR = "#d1495b"


@dataclass(frozen=True)
class StaggerCurve:
    beta_2: float  # deg
    beta_1_endpoint: float  # deg, largest inlet blade angle covered by the digitized curve
    coefficients: tuple[float, ...]  # highest power first

    def stagger(self, beta_1: float | np.ndarray) -> float | np.ndarray:
        return np.polyval(self.coefficients, beta_1)


@cache
def load_curves() -> tuple[StaggerCurve, ...]:
    """Return the fitted curves in ascending beta_2, serializing the CSV first if the JSON is missing."""
    if not JSON_PATH.exists():
        serialize_polynomials()
    contents = json.loads(JSON_PATH.read_text())

    curves = [
        StaggerCurve(curve["beta_2"], curve["beta_1_endpoint"], tuple(curve["coefficients"]))
        for curve in contents["curves"]
    ]
    return tuple(sorted(curves, key=lambda curve: curve.beta_2))


def beta_2_range() -> tuple[float, float]:
    """Return the (min, max) outlet blade angle [deg] covered by the chart."""
    curves = load_curves()
    return curves[0].beta_2, curves[-1].beta_2


def _require_within(name: str, value: float, lower: float, upper: float) -> None:
    if not lower <= value <= upper:
        raise ValueError(f"{name} = {value} deg is outside the valid range [{lower:.2f}, {upper:.2f}] deg")


def _weighted_curves(beta_2: float) -> list[tuple[StaggerCurve, float]]:
    """Return the curves bracketing beta_2 with their nonzero linear interpolation weights."""
    _require_within("beta_2", beta_2, *beta_2_range())
    curves = load_curves()
    beta_2_values = [curve.beta_2 for curve in curves]

    upper_index = max(bisect_left(beta_2_values, beta_2), 1)
    lower = curves[upper_index - 1]
    upper = curves[upper_index]
    upper_weight = (beta_2 - lower.beta_2) / (upper.beta_2 - lower.beta_2)

    weighted_curves = [(lower, 1.0 - upper_weight), (upper, upper_weight)]
    return [(curve, weight) for curve, weight in weighted_curves if weight > 0.0]


def beta_1_max(beta_2: float) -> float:
    """Return the largest valid inlet blade angle [deg] for the outlet blade angle beta_2 [deg]."""
    return min(curve.beta_1_endpoint for curve, _ in _weighted_curves(beta_2))


def stagger_angle(beta_1: float, beta_2: float) -> float:
    """Return the suggested stagger angle lambda [deg] for inlet and outlet blade angles beta_1, beta_2 [deg]."""
    _require_within("beta_1", beta_1, BETA_1_MIN, beta_1_max(beta_2))
    weighted_staggers = (weight * curve.stagger(beta_1) for curve, weight in _weighted_curves(beta_2))
    return float(sum(weighted_staggers))


def plot_stagger_chart(show_digitized_points: bool = True) -> Figure:
    """Reconstruct K&O Fig. 5 from the fitted polynomials, optionally overlaying the digitized points."""
    curves = load_curves()
    digitized_curves = read_digitized_curves()

    figure, axes = plt.subplots(figsize=(7.0, 5.0), layout="constrained")
    for curve in curves:
        beta_1 = np.linspace(BETA_1_MIN, curve.beta_1_endpoint, 200)
        stagger = [stagger_angle(value, curve.beta_2) for value in beta_1]
        axes.plot(beta_1, stagger, color=_CURVE_COLOR, linewidth=1.5)
        axes.annotate(
            f"{curve.beta_2:g}",
            xy=(beta_1[0], stagger[0]),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
        )

        if show_digitized_points:
            points = digitized_curves[curve.beta_2]
            axes.scatter(points[:, 0], points[:, 1], s=16, color=_POINT_COLOR, zorder=3)

    axes.text(BETA_1_MIN - 6.0, 76.0, r"$\beta_2$ [deg]", ha="center", fontsize=9)
    axes.set_xlim(-45.0, 75.0)
    axes.set_ylim(0.0, 80.0)
    axes.set_xlabel(r"Blade section inlet angle $\beta_1$ [deg]")
    axes.set_ylabel(r"Stagger angle $\lambda$ [deg]")
    axes.set_title("Stagger angle for typical turbine blade sections (Kacker & Okapuu, 1982)", fontsize=10)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(True)
    return figure


if __name__ == "__main__":
    plot_stagger_chart()
    plt.show()
