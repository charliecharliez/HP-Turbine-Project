from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["bezier_curve", "Camberline"]


def bezier_curve(control_points: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Evaluate a Bezier curve of any degree at n_points evenly spaced parameter values."""
    control_points = np.asarray(control_points, dtype=float)
    degree = len(control_points) - 1
    t = np.linspace(0.0, 1.0, n_points)[:, np.newaxis]
    index = np.arange(degree + 1)
    binomials = np.array([math.comb(degree, i) for i in index])
    bernstein_basis = binomials * t**index * (1.0 - t) ** (degree - index)
    return bernstein_basis @ control_points


# Coordinates: x axial, y tangential, leading edge at the origin.
# Angles: deg from axial, + toward +y.
@dataclass(frozen=True)
class Camberline:
    """Quadratic Bezier camberline set by the axial chord, stagger and metal angles."""

    axial_chord: float
    stagger: float  # deg
    inlet_metal_angle: float  # deg
    exit_metal_angle: float  # deg

    def __post_init__(self) -> None:
        if self.axial_chord <= 0:
            raise ValueError(f"Axial chord must be positive, got {self.axial_chord}")
        for name in ("stagger", "inlet_metal_angle", "exit_metal_angle"):
            angle = getattr(self, name)
            if not -90 < angle < 90:
                raise ValueError(f"{name} must lie strictly between -90 and 90 deg, got {angle}")

        lower_bound = min(self.inlet_metal_angle, self.exit_metal_angle)
        upper_bound = max(self.inlet_metal_angle, self.exit_metal_angle)
        if not lower_bound < self.stagger < upper_bound:
            raise ValueError(
                f"Stagger ({self.stagger}°) must lie strictly between the inlet "
                f"({self.inlet_metal_angle}°) and exit ({self.exit_metal_angle}°) metal angles, "
                f"otherwise the metal-angle tangents do not meet within the axial chord"
            )

    @property
    def leading_edge(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    @property
    def trailing_edge(self) -> np.ndarray:
        tangential_offset = self.axial_chord * _tan(self.stagger)
        return np.array([self.axial_chord, tangential_offset])

    @property
    def tangent_intersection(self) -> np.ndarray:
        """Middle control point, where the inlet and exit metal-angle tangents meet."""
        inlet_slope = _tan(self.inlet_metal_angle)
        exit_slope = _tan(self.exit_metal_angle)
        stagger_slope = _tan(self.stagger)
        axial_position = self.axial_chord * (exit_slope - stagger_slope) / (exit_slope - inlet_slope)
        return np.array([axial_position, axial_position * inlet_slope])

    @property
    def control_points(self) -> np.ndarray:
        return np.vstack((self.leading_edge, self.tangent_intersection, self.trailing_edge))

    @property
    def chord(self) -> float:
        return self.axial_chord / math.cos(math.radians(self.stagger))

    @property
    def camber_angle(self) -> float:  # deg
        return abs(self.exit_metal_angle - self.inlet_metal_angle)

    def coordinates(self, n_points: int = 200) -> np.ndarray:
        return bezier_curve(self.control_points, n_points)


def _tan(angle: float) -> float:  # angle in deg
    return math.tan(math.radians(angle))
