from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from Pipeline.Airfoil.bezier import bezier_curve, bezier_derivative, bezier_points

__all__ = ["Camberline"]


# Coordinates: x axial, y tangential, leading edge at the origin.
# Angles: deg from axial, + toward +y.
# Chord ratio: x/c, projection onto the chord from the leading edge over the chord length.
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
        for name in ("inlet_metal_angle", "exit_metal_angle"):
            if abs(getattr(self, name) - self.stagger) >= 90:
                raise ValueError(
                    f"{name} must lie within 90° of the stagger ({self.stagger}°), otherwise the "
                    f"camberline doubles back along the chord"
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

    def points_at_chordratios(self, chord_ratios: np.ndarray | float) -> np.ndarray:
        return bezier_points(self.control_points, self.bezier_parameters(chord_ratios))

    def unit_tangents(self, chord_ratios: np.ndarray | float) -> np.ndarray:
        parameters = self.bezier_parameters(chord_ratios)
        derivatives = bezier_derivative(self.control_points, parameters)
        return derivatives / np.linalg.norm(derivatives, axis=1, keepdims=True)

    def suction_normals(self, chord_ratios: np.ndarray | float) -> np.ndarray:
        """Unit normals pointing to the convex (suction) side of the camberline."""
        tangents = self.unit_tangents(chord_ratios)
        left_normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        turning_direction = math.copysign(1.0, self.exit_metal_angle - self.inlet_metal_angle)
        return -turning_direction * left_normals

    def bezier_parameters(self, chord_ratios: np.ndarray | float) -> np.ndarray:
        """Bezier parameters of the camber points whose projections on the chord sit at x/c."""
        chord_ratios = np.atleast_1d(np.asarray(chord_ratios, dtype=float))
        chord_direction = self.trailing_edge / self.chord
        middle_ratio = float(self.tangent_intersection @ chord_direction) / self.chord
        discriminant = middle_ratio**2 + (1.0 - 2.0 * middle_ratio) * chord_ratios
        return chord_ratios / (middle_ratio + np.sqrt(discriminant))


def _tan(angle: float) -> float:  # angle in deg
    return math.tan(math.radians(angle))
