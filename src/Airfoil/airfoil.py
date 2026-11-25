from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from matplotlib.path import Path
from scipy.optimize import minimize, root
from scipy.spatial import cKDTree

from Airfoil.bezier import bezier_derivative, bezier_points
from Airfoil.camberline import Camberline

__all__ = ["Airfoil", "AirfoilSide", "OffsetPoint", "TrailingEdgeCircle"]

_SUCTION_SIDE = 1.0  # sign along the camber suction normals
_PRESSURE_SIDE = -1.0
_OUTLINE_POINTS_PER_SIDE = 2000  # resolution of the outline for the inscribed-circle search
_SEARCH_GRID_SIZE = 60  # grid points per axis seeding the inscribed-circle search


# Lengths in the units of the camberline's axial chord; angles in deg, + away from the camberline.
# Chord ratio: x/c along the camberline's chord. Offsets d: distance along the camber normal.
@dataclass(frozen=True)
class OffsetPoint:
    """Control point on the camber normal at chord_ratio, offset by d to its side."""

    chord_ratio: float
    offset: float


@dataclass(frozen=True)
class AirfoilSide:
    """Control points of one side: alpha point, interior points and epsilon point."""

    alpha: float  # deg, camber LE tangent to the line from the LE point to the alpha point
    epsilon: float  # deg, camber TE tangent to the line from the TE apex to the epsilon point
    alpha_chord_ratio: float
    epsilon_chord_ratio: float
    interior_points: tuple[OffsetPoint, ...] = ()  # between the alpha and epsilon points

    def __post_init__(self) -> None:
        for name in ("alpha", "epsilon"):
            angle = getattr(self, name)
            if not -90 < angle < 90:
                raise ValueError(f"{name} must lie strictly between -90 and 90 deg, got {angle}")

        chord_ratios = self.chord_ratios
        if not (0 < chord_ratios[0] and chord_ratios[-1] < 1 and np.all(np.diff(chord_ratios) > 0)):
            raise ValueError(
                f"Chord ratios must increase strictly from the alpha point to the epsilon point "
                f"within (0, 1), got {[round(ratio, 4) for ratio in chord_ratios]}"
            )

    @property
    def chord_ratios(self) -> list[float]:
        """Chord ratios of the alpha, interior and epsilon points, in order."""
        interior_ratios = [point.chord_ratio for point in self.interior_points]
        return [self.alpha_chord_ratio, *interior_ratios, self.epsilon_chord_ratio]


@dataclass(frozen=True)
class TrailingEdgeCircle:
    """Circle tangent to both sides that rounds off the sharp TE apex, passing through the camber TE."""

    center: np.ndarray
    radius: float
    suction_parameter: float  # Bezier parameter of the suction-side tangent point
    pressure_parameter: float  # Bezier parameter of the pressure-side tangent point


# Each side is a Bezier curve from the camber LE to a TE apex common to both sides, which lies beyond
# the camber TE along the exit metal angle. Every control point in between lies on a camber normal,
# offset by its d to that side. A circle tangent to both sides and through the camber TE rounds off
# the apex; each side ends at its tangent point.
@dataclass(frozen=True)
class Airfoil:
    """Suction and pressure sides as Bezier curves offset orthogonally from a camberline."""

    camberline: Camberline
    le_radius: float  # d of both sides' control points beside the camber LE
    te_diam: float  # diameter of the trailing-edge circle
    suction: AirfoilSide
    pressure: AirfoilSide
    te_extension: float = field(init=False)  # TE apex distance beyond the camber TE
    te_circle: TrailingEdgeCircle = field(init=False)

    def __post_init__(self) -> None:
        for name in ("le_radius", "te_diam"):
            length = getattr(self, name)
            if length < 0:
                raise ValueError(f"{name} must not be negative, got {length}")
        te_extension, te_circle = self._solve_trailing_edge()
        object.__setattr__(self, "te_extension", te_extension)
        object.__setattr__(self, "te_circle", te_circle)

    @cached_property
    def chord_length(self) -> float:
        """Camberline chord, the reference length of every chord ratio."""
        return self.camberline.chord

    @cached_property
    def max_thickness(self) -> float:
        """Diameter of the largest circle inscribed in the profile."""
        _, radius = self._max_inscribed_circle
        return 2 * radius

    @cached_property
    def chord_ratio_at_max_thick(self) -> float:
        """Chord ratio x/c of the centre of the largest inscribed circle."""
        center, _ = self._max_inscribed_circle
        chord_direction = self.camberline.trailing_edge / self.chord_length
        return float(center @ chord_direction) / self.chord_length

    @cached_property
    def _max_inscribed_circle(self) -> tuple[np.ndarray, float]:
        outline = self.coordinates(_OUTLINE_POINTS_PER_SIDE)
        boundary = Path(outline)
        nearest_boundary = cKDTree(outline)

        def inscribed_radius(center: np.ndarray) -> float:
            distance, _ = nearest_boundary.query(center)
            return float(distance) if boundary.contains_point(center) else -float(distance)

        lower_corner = outline.min(axis=0)
        upper_corner = outline.max(axis=0)
        axes = [np.linspace(low, high, _SEARCH_GRID_SIZE) for low, high in zip(lower_corner, upper_corner)]
        candidates = np.column_stack([grid.ravel() for grid in np.meshgrid(*axes)])
        candidates = candidates[boundary.contains_points(candidates)]
        distances, _ = nearest_boundary.query(candidates)
        best_candidate = candidates[np.argmax(distances)]

        refined = minimize(lambda center: -inscribed_radius(center), best_candidate, method="Nelder-Mead",
                           options={"xatol": 1e-9, "fatol": 1e-12})
        return refined.x, inscribed_radius(refined.x)

    # ============================================================ control points
    @cached_property
    def suction_control_points(self) -> np.ndarray:
        return self._control_points(self.suction, _SUCTION_SIDE, self.te_extension)

    @cached_property
    def pressure_control_points(self) -> np.ndarray:
        return self._control_points(self.pressure, _PRESSURE_SIDE, self.te_extension)

    @property
    def te_apex(self) -> np.ndarray:
        return self._te_apex(self.te_extension)

    def _te_apex(self, te_extension: float) -> np.ndarray:
        te_tangent = self.camberline.unit_tangents(1.0)[0]
        return self.camberline.trailing_edge + te_extension * te_tangent

    def _control_points(self, side: AirfoilSide, side_sign: float,
                        te_extension: float) -> np.ndarray:
        camberline = self.camberline
        chord_ratios = np.array([0.0, *side.chord_ratios, 1.0])
        feet = camberline.points_at_chordratios(chord_ratios)  # camber points under each control point
        normals = side_sign * camberline.suction_normals(chord_ratios)
        le_tangent, te_tangent = camberline.unit_tangents(np.array([0.0, 1.0]))

        le_point = feet[0] + self.le_radius * normals[0]
        te_apex = self._te_apex(te_extension)
        alpha_line = _rotate_toward(le_tangent, normals[0], side.alpha)
        epsilon_line = _rotate_toward(-te_tangent, normals[-1], side.epsilon)
        alpha_point = _meet_camber_normal(le_point, alpha_line, feet[1], normals[1])
        epsilon_point = _meet_camber_normal(te_apex, epsilon_line, feet[-2], normals[-2])
        interior_points = [foot + point.offset * normal for point, foot, normal
                           in zip(side.interior_points, feet[2:-2], normals[2:-2])]
        return np.vstack((feet[0], le_point, alpha_point, *interior_points, epsilon_point,
                          te_apex))

    # ================================================================== surfaces
    def suction_side(self, n_points: int = 100) -> np.ndarray:
        """Suction side from the camber LE to its tangent point on the trailing-edge circle."""
        parameters = np.linspace(0.0, self.te_circle.suction_parameter, n_points)
        return bezier_points(self.suction_control_points, parameters)

    def pressure_side(self, n_points: int = 100) -> np.ndarray:
        """Pressure side from the camber LE to its tangent point on the trailing-edge circle."""
        parameters = np.linspace(0.0, self.te_circle.pressure_parameter, n_points)
        return bezier_points(self.pressure_control_points, parameters)

    def trailing_edge_arc(self, n_points: int = 30) -> np.ndarray:
        """Arc of the trailing-edge circle from the suction to the pressure tangent point, round the apex side."""
        circle = self.te_circle
        suction_end = self.suction_side(2)[-1]
        pressure_end = self.pressure_side(2)[-1]
        start_angle = _polar_angle(suction_end - circle.center)
        end_angle = _polar_angle(pressure_end - circle.center)
        apex_angle = _polar_angle(self.te_apex - circle.center)

        counterclockwise_sweep = (end_angle - start_angle) % (2 * math.pi)
        apex_on_counterclockwise_arc = (apex_angle - start_angle) % (2 * math.pi) < counterclockwise_sweep
        sweep = counterclockwise_sweep if apex_on_counterclockwise_arc else counterclockwise_sweep - 2 * math.pi
        angles = start_angle + np.linspace(0.0, sweep, n_points)
        return circle.center + circle.radius * np.column_stack((np.cos(angles), np.sin(angles)))

    def coordinates(self, n_points_per_side: int = 100) -> np.ndarray:
        """Closed loop: LE along the suction side, round the trailing-edge arc, back along the pressure side."""
        suction_side = self.suction_side(n_points_per_side)
        trailing_edge_arc = self.trailing_edge_arc()[1:-1]
        pressure_side_reversed = self.pressure_side(n_points_per_side)[::-1]
        return np.vstack((suction_side, trailing_edge_arc, pressure_side_reversed))

    # ============================================================= trailing edge
    def _solve_trailing_edge(self) -> tuple[float, TrailingEdgeCircle]:
        """TE apex distance and the circle tangent to both sides there that passes through the camber TE."""
        radius = self.te_diam / 2
        camber_te = self.camberline.trailing_edge
        if radius == 0:
            return 0.0, TrailingEdgeCircle(camber_te, 0.0, 1.0, 1.0)

        suction_at_zero, suction_rate = self._control_points_linear_in_extension(self.suction,
                                                                                 _SUCTION_SIDE)
        pressure_at_zero, pressure_rate = self._control_points_linear_in_extension(self.pressure,
                                                                                   _PRESSURE_SIDE)
        te_tangent = self.camberline.unit_tangents(1.0)[0]
        toward_suction = self.camberline.suction_normals(1.0)[0]

        def circle_center(lateral_parameter: float) -> np.ndarray:
            to_camber_te = _downstream_direction(te_tangent, toward_suction, lateral_parameter)
            return camber_te - radius * to_camber_te

        def residuals(unknowns: np.ndarray) -> np.ndarray:
            te_extension, suction_parameter, pressure_parameter, lateral_parameter = unknowns
            suction_points = suction_at_zero + te_extension * suction_rate
            pressure_points = pressure_at_zero + te_extension * pressure_rate
            suction_center = _inward_offset(suction_points, suction_parameter, -toward_suction, radius)
            pressure_center = _inward_offset(pressure_points, pressure_parameter, toward_suction, radius)
            center = circle_center(lateral_parameter)
            return np.concatenate((suction_center - center, pressure_center - center))

        initial_guess = self._straight_wedge_estimate(radius, suction_at_zero, suction_rate,
                                                      pressure_at_zero, pressure_rate)
        solution = root(residuals, initial_guess)
        te_extension, suction_parameter, pressure_parameter, lateral_parameter = solution.x
        if not (solution.success and te_extension >= 0 and 0 < suction_parameter < 1
                and 0 < pressure_parameter < 1):
            raise ValueError("No trailing-edge circle of this diameter fits between the suction and "
                             "pressure sides through the camber TE; reduce te_diam")
        circle = TrailingEdgeCircle(circle_center(lateral_parameter), radius, float(suction_parameter),
                                    float(pressure_parameter))
        return float(te_extension), circle

    def _control_points_linear_in_extension(self, side: AirfoilSide,
                                            side_sign: float) -> tuple[np.ndarray, np.ndarray]:
        """Control points at zero TE extension and their rate of change with it; both are exact."""
        unit_extension = self.camberline.axial_chord
        at_zero = self._control_points(side, side_sign, 0.0)
        at_unit = self._control_points(side, side_sign, unit_extension)
        return at_zero, (at_unit - at_zero) / unit_extension

    def _straight_wedge_estimate(self, radius: float, suction_at_zero: np.ndarray,
                                 suction_rate: np.ndarray, pressure_at_zero: np.ndarray,
                                 pressure_rate: np.ndarray) -> np.ndarray:
        """TE extension, tangent-point parameters and lateral parameter if both sides were their straight epsilon lines."""
        half_wedge = math.radians(self.suction.epsilon + self.pressure.epsilon) / 2
        bisector_skew = math.radians(self.suction.epsilon - self.pressure.epsilon) / 2
        if not abs(math.sin(bisector_skew)) < math.sin(half_wedge):
            raise ValueError("Both epsilon_ss and epsilon_ps must be positive to fit a trailing-edge "
                             "circle through the camber TE")

        apex_to_center = radius / math.sin(half_wedge)
        skew_ratio = math.sin(bisector_skew) / math.sin(half_wedge)
        te_extension = apex_to_center * math.cos(bisector_skew) - radius * math.sqrt(1 - skew_ratio**2)
        apex_to_tangent_point = radius / math.tan(half_wedge)
        suction_points = suction_at_zero + te_extension * suction_rate
        pressure_points = pressure_at_zero + te_extension * pressure_rate
        return np.array([te_extension,
                         _parameter_before_end(suction_points, apex_to_tangent_point),
                         _parameter_before_end(pressure_points, apex_to_tangent_point),
                         math.atanh(-skew_ratio)])


def _rotate_toward(direction: np.ndarray, normal: np.ndarray, angle: float) -> np.ndarray:
    radians = math.radians(angle)
    return math.cos(radians) * direction + math.sin(radians) * normal


def _meet_camber_normal(origin: np.ndarray, direction: np.ndarray, foot: np.ndarray,
                        normal: np.ndarray) -> np.ndarray:
    """Point where the line from origin along direction crosses the camber normal through foot."""
    line_and_normal = np.column_stack((direction, -normal))
    distance_along_line, _ = np.linalg.solve(line_and_normal, foot - origin)
    if distance_along_line <= 0:
        raise ValueError("An alpha or epsilon line points away from the camber normal at its "
                         "chord ratio; reduce the angle or move the point")
    return origin + distance_along_line * direction


def _inward_offset(control_points: np.ndarray, parameter: float, inward: np.ndarray,
                   distance: float) -> np.ndarray:
    """Point at distance along the curve normal that faces the inward direction."""
    point = bezier_points(control_points, parameter)[0]
    tangent = bezier_derivative(control_points, parameter)[0]
    normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    return point + distance * math.copysign(1.0, normal @ inward) * normal


def _parameter_before_end(control_points: np.ndarray, distance: float) -> float:
    """Bezier parameter roughly the given distance before the curve's end, along its last leg."""
    degree = len(control_points) - 1
    last_leg = float(np.linalg.norm(control_points[-1] - control_points[-2]))
    return float(np.clip(1.0 - distance / (degree * last_leg), 0.5, 0.999))


def _downstream_direction(tangent: np.ndarray, normal: np.ndarray,
                          lateral_parameter: float) -> np.ndarray:
    """Unit vector whose tangent component stays positive; tanh(lateral_parameter) is its normal component."""
    normal_component = math.tanh(lateral_parameter)
    return math.sqrt(1 - normal_component**2) * tangent + normal_component * normal


def _polar_angle(vector: np.ndarray) -> float:  # rad
    return math.atan2(vector[1], vector[0])


