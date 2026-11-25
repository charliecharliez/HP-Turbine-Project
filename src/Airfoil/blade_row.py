"""Blade-row sizing at hub, midspan and tip: Zweifel pitch, blade count, KO stagger, Bezier profile and measured throat."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from Airfoil.airfoil import Airfoil, AirfoilSide, OffsetPoint
from Airfoil.bezier import bezier_points
from Airfoil.camberline import Camberline
from Empirical.StaggerKO import get_stagger
from Losses.losses_solver import (STATOR_INLET_FLOW_ANGLE, AirfoilParams, RowKind, to_kacker_okapuu_angles,
                                  zweifel_pitch_chord)
from Meanline.gas_props import GAS, GasProps
from RadialEquilibrium.radial_equilibrium_solver import SpanwiseFlow

SPAN_FRACTIONS = {"hub": 0.0, "midspan": 0.5, "tip": 1.0}
THROAT_SAMPLES = 400
THICKNESS_SAMPLES = 200


# ---------------------------------------------------------------------------
# Design choices
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProfileShape:
    """Bezier profile parameterization, with lengths in axial chords."""

    le_radius: float = 0.05
    te_diam: float = 0.03
    suction: AirfoilSide = AirfoilSide(alpha=25.0, epsilon=8.0, alpha_chord_ratio=0.10, epsilon_chord_ratio=0.80,
                                       interior_points=(OffsetPoint(0.35, 0.35),))
    pressure: AirfoilSide = AirfoilSide(alpha=5.0, epsilon=2.0, alpha_chord_ratio=0.10, epsilon_chord_ratio=0.80,
                                        interior_points=(OffsetPoint(0.50, -0.08),))


@dataclass(frozen=True)
class BladeRowDesign:
    aspect_ratio: float  # h / c at midspan
    zweifel: float = 0.8  # target at midspan, before rounding the blade count
    shape: ProfileShape = field(default_factory=ProfileShape)


STATOR_DESIGN = BladeRowDesign(aspect_ratio=1.0)
ROTOR_DESIGN = BladeRowDesign(aspect_ratio=1.2)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionFlow:
    """Row-frame flow at one span; angles in deg signed positive along rotation."""

    span_fraction: float
    radius: float  # m, at row exit
    inlet_angle: float
    exit_angle: float
    exit_mach: float


@dataclass(frozen=True)
class BladeSection:
    """One spanwise blade section; lengths in m, airfoil coordinates in axial chords, metal angles at zero incidence and deviation."""

    flow: SectionFlow
    airfoil: Airfoil
    axial_chord: float
    pitch: float
    throat: float
    max_thickness: float
    throat_exit_angle: float  # deg, unsigned exit flow angle the throat sets

    @property
    def stagger(self) -> float:  # deg
        return self.airfoil.camberline.stagger

    @property
    def chord(self) -> float:
        return self.airfoil.camberline.chord * self.axial_chord

    @property
    def pitch_chord(self) -> float:
        return self.pitch / self.chord

    @property
    def throat_pitch(self) -> float:
        return self.throat / self.pitch

    @property
    def trailing_edge_throat(self) -> float:
        return self.airfoil.te_diam * self.axial_chord / self.throat

    @property
    def max_thickness_chord(self) -> float:
        return self.max_thickness / self.chord

    @property
    def zweifel(self) -> float:
        return zweifel_coefficient(self.flow.inlet_angle, self.flow.exit_angle, self.pitch / self.axial_chord)

    @property
    def deviation(self) -> float:  # deg, throat exit angle minus the metal exit angle magnitude
        return self.throat_exit_angle - abs(self.airfoil.camberline.exit_metal_angle)


@dataclass(frozen=True)
class BladeRow:
    kind: RowKind
    blade_count: int
    height: float  # m, at row exit
    hub: BladeSection
    midspan: BladeSection
    tip: BladeSection

    @property
    def sections(self) -> dict[str, BladeSection]:
        return {"hub": self.hub, "midspan": self.midspan, "tip": self.tip}

    def airfoil_params(self) -> AirfoilParams:
        """Midspan parameters in the form the Losses module reads, reproducing the rounded pitch."""
        section = self.midspan
        return AirfoilParams(
            aspect_ratio=self.height / section.chord,
            axial_chord_ratio=section.axial_chord / section.chord,
            zweifel=section.zweifel,
            max_thickness_chord=section.max_thickness_chord,
            trailing_edge_throat=section.trailing_edge_throat,
            inlet_metal_angle=section.airfoil.camberline.inlet_metal_angle,
        )


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def kacker_okapuu_stagger(inlet_metal_angle: float, exit_metal_angle: float) -> float:
    """Signed stagger [deg] from KO Fig. 5 for signed metal angles [deg]."""
    ko_inlet, ko_exit = to_kacker_okapuu_angles(inlet_metal_angle, exit_metal_angle)
    return math.copysign(get_stagger(ko_inlet, ko_exit), exit_metal_angle)


def zweifel_coefficient(inlet_angle: float, exit_angle: float, pitch_axial_chord: float) -> float:
    unit_zweifel_pitch_axial_chord = zweifel_pitch_chord(inlet_angle, exit_angle, axial_chord_ratio=1.0, zweifel=1.0)
    return pitch_axial_chord / unit_zweifel_pitch_axial_chord


def isentropic_area_ratio(mach: float, gamma: float) -> float:
    """A / A* at the given Mach number."""
    stagnation_term = 2 / (gamma + 1) * (1 + (gamma - 1) / 2 * mach**2)
    return stagnation_term ** ((gamma + 1) / (2 * (gamma - 1))) / mach


def throat_exit_angle(throat_pitch: float, exit_mach: float, gamma: float) -> float:
    """Unsigned exit flow angle [deg]: cos a = o/s subsonic, cos a = (o/s) A/A* past Mach 1."""
    area_ratio = isentropic_area_ratio(exit_mach, gamma) if exit_mach > 1 else 1.0
    return math.degrees(math.acos(min(1.0, throat_pitch * area_ratio)))


# ---------------------------------------------------------------------------
# Profile measurements (in axial chords)
# ---------------------------------------------------------------------------
def measure_throat(airfoil: Airfoil, pitch: float) -> float:
    """Shortest distance from the trailing-edge circle to the suction side of the neighbouring blade one pitch away."""
    camberline = airfoil.camberline
    turning_direction = math.copysign(1.0, camberline.exit_metal_angle - camberline.inlet_metal_angle)
    neighbour_suction_points = airfoil.suction_control_points + np.array([0.0, turning_direction * pitch])
    circle = airfoil.te_circle

    def gap(parameters: np.ndarray | float) -> np.ndarray:
        points = bezier_points(neighbour_suction_points, parameters)
        return np.linalg.norm(points - circle.center, axis=1) - circle.radius

    parameters = np.linspace(0.0, circle.suction_parameter, THROAT_SAMPLES)
    nearest = int(np.argmin(gap(parameters)))
    bracket = (parameters[max(nearest - 1, 0)], parameters[min(nearest + 1, THROAT_SAMPLES - 1)])
    refined = minimize_scalar(lambda parameter: float(gap(parameter)[0]), bounds=bracket, method="bounded")
    return float(refined.fun)


def measure_max_thickness(airfoil: Airfoil) -> float:
    """Largest distance from a suction-side point to its nearest pressure-side point."""
    suction_side = airfoil.suction_side(THICKNESS_SAMPLES)
    pressure_side = airfoil.pressure_side(THICKNESS_SAMPLES)
    distances = np.linalg.norm(suction_side[:, np.newaxis, :] - pressure_side[np.newaxis, :, :], axis=2)
    return float(distances.min(axis=1).max())


# ---------------------------------------------------------------------------
# Spanwise flow -> blade rows
# ---------------------------------------------------------------------------
def stator_section_flows(flow: SpanwiseFlow) -> list[SectionFlow]:
    sections = []
    for span_fraction in SPAN_FRACTIONS.values():
        stator_exit = flow.at_span(span_fraction).stator_exit
        sections.append(SectionFlow(span_fraction, stator_exit.radius, STATOR_INLET_FLOW_ANGLE,
                                    stator_exit.absolute_angle, stator_exit.mach))
    return sections


def rotor_section_flows(flow: SpanwiseFlow) -> list[SectionFlow]:
    sections = []
    for span_fraction in SPAN_FRACTIONS.values():
        span = flow.at_span(span_fraction)
        sections.append(SectionFlow(span_fraction, span.rotor_exit.radius, span.stator_exit.relative_angle,
                                    span.rotor_exit.relative_angle, span.rotor_exit.relative_mach))
    return sections


def blade_section(flow: SectionFlow, axial_chord: float, pitch: float, shape: ProfileShape,
                  gas: GasProps) -> BladeSection:
    stagger = kacker_okapuu_stagger(flow.inlet_angle, flow.exit_angle)
    camberline = Camberline(1.0, stagger, flow.inlet_angle, flow.exit_angle)
    airfoil = Airfoil(camberline, shape.le_radius, shape.te_diam, shape.suction, shape.pressure)
    throat = measure_throat(airfoil, pitch / axial_chord) * axial_chord
    return BladeSection(
        flow=flow,
        airfoil=airfoil,
        axial_chord=axial_chord,
        pitch=pitch,
        throat=throat,
        max_thickness=measure_max_thickness(airfoil) * axial_chord,
        throat_exit_angle=throat_exit_angle(throat / pitch, flow.exit_mach, gas.gamma),
    )


def blade_row(kind: RowKind, section_flows: list[SectionFlow], height: float, design: BladeRowDesign,
              gas: GasProps = GAS) -> BladeRow:
    """Axial chord from the midspan aspect ratio, blade count from the midspan Zweifel pitch, held across the span."""
    midspan = section_flows[1]
    midspan_stagger = kacker_okapuu_stagger(midspan.inlet_angle, midspan.exit_angle)
    chord = height / design.aspect_ratio
    axial_chord = chord * math.cos(math.radians(midspan_stagger))

    zweifel_pitch_axial_chord = zweifel_pitch_chord(midspan.inlet_angle, midspan.exit_angle, 1.0, design.zweifel)
    zweifel_pitch = zweifel_pitch_axial_chord * axial_chord
    blade_count = max(1, round(2 * math.pi * midspan.radius / zweifel_pitch))

    hub, mid, tip = (blade_section(section, axial_chord, 2 * math.pi * section.radius / blade_count,
                                   design.shape, gas) for section in section_flows)
    return BladeRow(kind, blade_count, height, hub, mid, tip)


def stator_row(flow: SpanwiseFlow, design: BladeRowDesign = STATOR_DESIGN, gas: GasProps = GAS) -> BladeRow:
    height = flow.annulus[2].tip_radius - flow.annulus[2].hub_radius
    return blade_row("stator", stator_section_flows(flow), height, design, gas)


def rotor_row(flow: SpanwiseFlow, design: BladeRowDesign = ROTOR_DESIGN, gas: GasProps = GAS) -> BladeRow:
    height = flow.annulus[3].tip_radius - flow.annulus[3].hub_radius
    return blade_row("rotor", rotor_section_flows(flow), height, design, gas)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
BLADE_TABLE_ROWS: tuple[tuple[str, Callable[[BladeSection], float], str], ...] = (
    ("Radius [mm]", lambda section: section.flow.radius * 1e3, ".2f"),
    ("Inlet metal [deg]", lambda section: section.airfoil.camberline.inlet_metal_angle, ".2f"),
    ("Exit metal [deg]", lambda section: section.airfoil.camberline.exit_metal_angle, ".2f"),
    ("Stagger [deg]", lambda section: section.stagger, ".2f"),
    ("Chord [mm]", lambda section: section.chord * 1e3, ".3f"),
    ("Axial chord [mm]", lambda section: section.axial_chord * 1e3, ".3f"),
    ("Pitch [mm]", lambda section: section.pitch * 1e3, ".3f"),
    ("Throat [mm]", lambda section: section.throat * 1e3, ".3f"),
    ("s/c", lambda section: section.pitch_chord, ".3f"),
    ("Zweifel", lambda section: section.zweifel, ".3f"),
    ("o/s", lambda section: section.throat_pitch, ".4f"),
    ("t_TE/o", lambda section: section.trailing_edge_throat, ".4f"),
    ("t_max/c", lambda section: section.max_thickness_chord, ".4f"),
    ("Exit Mach (row)", lambda section: section.flow.exit_mach, ".3f"),
    ("Throat exit [deg]", lambda section: section.throat_exit_angle, ".2f"),
    ("Deviation [deg]", lambda section: section.deviation, ".2f"),
)


def print_blade_row(row: BladeRow) -> None:
    sections = row.sections
    print(f"{row.kind}: {row.blade_count} blades, height {row.height * 1e3:.2f} mm")
    print(f"{'Quantity':<20}" + "".join(f"{name:>12}" for name in sections))
    print("-" * (20 + 12 * len(sections)))
    for label, value_of, number_format in BLADE_TABLE_ROWS:
        print(f"{label:<20}" + "".join(f"{value_of(section):>12{number_format}}" for section in sections.values()))
