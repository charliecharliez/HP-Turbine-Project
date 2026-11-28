"""Kacker-Okapuu (1982) stator and rotor losses."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from Constraints.boundary_conditions import boundary_conditions
from Pipeline.Meanline.gas_props import GAS
from Pipeline.Meanline.solve_meanline import MeanlineResult
from Pipeline.RadialEquilibrium.radial_equilibrium_solver import SpanPosition, SpanwiseFlow

RowKind = Literal["stator", "rotor"]

BAR = 1e5  # Pa


# ---------------------------------------------------------------------------
# Digitized Kacker-Okapuu charts (NASA turbo-design digitization of KO 1982)
# ---------------------------------------------------------------------------
CHART_DIRECTORY = Path(__file__).with_name("ko_charts")


def load_chart(file_name: str) -> np.ndarray:
    return np.loadtxt(CHART_DIRECTORY / file_name, delimiter=",", skiprows=1)


@dataclass(frozen=True)
class ProfileLossChart:
    """Profile loss Y_p against pitch/chord, one curve per exit angle."""

    name: str
    exit_angles: tuple[float, ...]  # deg, ascending
    curves: tuple[tuple[np.ndarray, np.ndarray], ...]  # (pitch/chord, Y_p) per exit angle
    lowest_pitch_chord: float  # lowest s/c covered by every curve
    highest_pitch_chord: float  # highest s/c covered by every curve

    @classmethod
    def from_file(cls, file_name: str, name: str) -> ProfileLossChart:
        table = load_chart(file_name)
        exit_angles = tuple(float(angle) for angle in np.unique(table[:, 2]))
        curves = []
        for exit_angle in exit_angles:
            rows = table[table[:, 2] == exit_angle]
            rows = rows[np.argsort(rows[:, 0])]
            curves.append((rows[:, 0], rows[:, 1]))
        lowest_pitch_chord = max(pitch_chords[0] for pitch_chords, _ in curves)
        highest_pitch_chord = min(pitch_chords[-1] for pitch_chords, _ in curves)
        return cls(name, exit_angles, tuple(curves), lowest_pitch_chord, highest_pitch_chord)

    def profile_loss(self, pitch_chord: float, exit_angle: float, warnings: list[str]) -> float:
        """Interpolate linearly in pitch/chord then exit angle; out-of-range inputs are clamped and noted."""
        if not self.exit_angles[0] <= exit_angle <= self.exit_angles[-1]:
            warnings.append(f"{self.name}: exit angle {exit_angle:.1f} deg outside "
                            f"{self.exit_angles[0]:g}-{self.exit_angles[-1]:g}")
        if not self.lowest_pitch_chord <= pitch_chord <= self.highest_pitch_chord:
            warnings.append(f"{self.name}: s/c = {pitch_chord:.3f} outside "
                            f"{self.lowest_pitch_chord:.2f}-{self.highest_pitch_chord:.2f}")

        losses_at_pitch_chord = [np.interp(pitch_chord, *curve) for curve in self.curves]
        return float(np.interp(exit_angle, self.exit_angles, losses_at_pitch_chord))


AXIAL_INLET_PROFILE_LOSS = ProfileLossChart.from_file("ko_fig01_yp_nozzle.csv", "Fig. 1")
IMPULSE_PROFILE_LOSS = ProfileLossChart.from_file("ko_fig02_yp_impulse.csv", "Fig. 2")
AXIAL_INLET_TRAILING_EDGE = load_chart("ko_fig14_te_nozzle.csv")  # Fig. 14, (t_TE/o, kinetic energy loss)
IMPULSE_TRAILING_EDGE = load_chart("ko_fig14_te_impulse.csv")  # Fig. 14, (t_TE/o, kinetic energy loss)

KO_DESIGN_LEVEL_FACTOR = 0.914  # KO scaling of profile losses to 1982 design standards
KO_PROFILE_REDUCTION = 2 / 3  # KO reduction of the Ainley-Mathieson profile loss
HUB_TIP_RATIOS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
HUB_TO_MEAN_MACH_RATIO: dict[RowKind, tuple[float, ...]] = {
    "stator": (1.4, 1.18, 1.05, 1.0, 1.0, 1.0),
    "rotor": (2.15, 1.7, 1.35, 1.12, 1.0, 1.0),
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AirfoilParams:
    """Airfoil parameters; placeholders until the Airfoil module supplies them."""

    aspect_ratio: float  # h / c
    axial_chord_ratio: float  # b_x / c
    zweifel: float = 0.8
    max_thickness_chord: float = 0.20  # t_max / c
    trailing_edge_throat: float = 0.08  # t_TE / o
    inlet_metal_angle: float | None = None  # deg, signed; None means zero incidence


STATOR_AIRFOIL = AirfoilParams(aspect_ratio=1.0, axial_chord_ratio=0.65)
ROTOR_AIRFOIL = AirfoilParams(aspect_ratio=1.2, axial_chord_ratio=0.85)


@dataclass(frozen=True)
class RowGeometry:
    """Blade-row geometry at midspan; lengths in m, angles in deg signed positive along rotation."""

    pitch_chord: float  # s / c
    chord: float
    axial_chord: float
    height: float  # blade height at row exit
    max_thickness_chord: float  # t_max / c
    trailing_edge_throat: float  # t_TE / o
    inlet_metal_angle: float
    hub_tip_ratio_inlet: float
    hub_radius_exit: float
    tip_radius_exit: float
    tip_clearance: float = 0.0  # m, unshrouded

    def __post_init__(self) -> None:
        for name in (
            "pitch_chord",
            "chord",
            "axial_chord",
            "height",
            "hub_radius_exit",
            "tip_radius_exit",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        for name in ("max_thickness_chord", "trailing_edge_throat", "hub_tip_ratio_inlet"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if not math.isfinite(self.inlet_metal_angle):
            raise ValueError(f"inlet_metal_angle must be finite, got {self.inlet_metal_angle}")
        if not math.isfinite(self.tip_clearance) or self.tip_clearance < 0:
            raise ValueError(f"tip_clearance must be finite and nonnegative, got {self.tip_clearance}")
        if self.tip_radius_exit <= self.hub_radius_exit:
            raise ValueError("tip_radius_exit must exceed hub_radius_exit")


@dataclass(frozen=True)
class RowFlow:
    """Midspan flow in the row frame; angles in deg signed positive along rotation, pressures in one consistent unit."""

    kind: RowKind
    inlet_angle: float
    exit_angle: float
    inlet_mach: float
    exit_mach: float
    inlet_static_pressure: float
    exit_static_pressure: float
    reynolds: float  # chord-based, at row exit
    gamma: float

    def __post_init__(self) -> None:
        for name in (
            "inlet_angle",
            "exit_angle",
            "inlet_mach",
            "exit_mach",
            "inlet_static_pressure",
            "exit_static_pressure",
            "reynolds",
            "gamma",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.inlet_mach < 0:
            raise ValueError(f"inlet_mach must be nonnegative, got {self.inlet_mach}")
        if self.exit_mach <= 0:
            raise ValueError(f"exit_mach must be positive, got {self.exit_mach}")
        if self.inlet_static_pressure <= 0 or self.exit_static_pressure <= 0:
            raise ValueError("row static pressures must be positive")
        if self.reynolds <= 0:
            raise ValueError(f"reynolds must be positive, got {self.reynolds}")
        if self.gamma <= 1:
            raise ValueError(f"gamma must exceed 1, got {self.gamma}")


@dataclass(frozen=True)
class RowLossInputs:
    """Validated row package passed from geometry and meanline modules to the loss model."""

    stage_index: int
    row_kind: RowKind
    reference_frame: Literal["absolute", "rotor_relative"]
    span_location: Literal["hub", "midspan", "tip"]
    geometry: RowGeometry
    flow: RowFlow

    def __post_init__(self) -> None:
        if isinstance(self.stage_index, bool) or not isinstance(self.stage_index, int) or self.stage_index < 1:
            raise ValueError(f"stage_index must be a positive integer, got {self.stage_index}")
        if self.row_kind not in ("stator", "rotor"):
            raise ValueError(f"row_kind must be 'stator' or 'rotor', got {self.row_kind}")
        if self.reference_frame not in ("absolute", "rotor_relative"):
            raise ValueError(f"unsupported reference_frame, got {self.reference_frame}")
        if self.span_location not in ("hub", "midspan", "tip"):
            raise ValueError(f"unsupported span_location, got {self.span_location}")
        if self.flow.kind != self.row_kind:
            raise ValueError("row kind must match the row flow")
        if self.reference_frame == "absolute" and self.row_kind != "stator":
            raise ValueError("absolute frame row inputs are valid only for a stator")
        if self.reference_frame == "rotor_relative" and self.row_kind != "rotor":
            raise ValueError("rotor-relative frame row inputs are valid only for a rotor")


@dataclass(frozen=True)
class RowLoss:
    """Total-pressure loss coefficients Y = (P01 - P02) / (P02 - P2) of one row, split by source."""

    kind: RowKind
    profile: float
    hub_shock: float
    secondary: float
    trailing_edge: float
    supersonic: float
    total: float
    kinetic_loss: float  # xi = 1 - V^2 / V_is^2, the meanline loss input
    tip_efficiency_debit: float  # rotor only
    mach_correction: float  # K_p
    reynolds_correction: float  # f_Re
    supersonic_correction: float  # f_Ma
    chart_warnings: tuple[str, ...] = ()
    profile_base: float = 0.0
    compressibility_increment: float = 0.0
    reynolds_increment: float = 0.0

    @property
    def incremental_pressure_components(self) -> tuple[tuple[str, float], ...]:
        profile_base = self.profile_base if self.profile_base else self.profile
        return (
            ("profile_base", profile_base),
            ("compressibility_increment", self.compressibility_increment),
            ("reynolds_increment", self.reynolds_increment),
            ("hub_shock", self.hub_shock),
            ("secondary", self.secondary),
            ("trailing_edge", self.trailing_edge),
            ("supersonic_increment", self.supersonic),
        )

    @property
    def incremental_pressure_total(self) -> float:
        return math.fsum(value for _, value in self.incremental_pressure_components)

    @property
    def accounting_residual(self) -> float:
        return self.incremental_pressure_total - self.total


@dataclass(frozen=True)
class StageEfficiency:
    """Uncooled stage performance with the unshrouded rotor tip-clearance debit applied."""

    specific_work: float  # J/kg
    total_to_total: float
    total_to_static: float
    tip_clearance_debit: float  # fractional efficiency reduction


MachRegime = Literal["subsonic", "transonic", "supersonic"]


@dataclass(frozen=True)
class MachAssessment:
    """Mach value and regime at a named station and reference frame."""

    station_id: str
    frame: Literal["absolute", "rotor_relative"]
    mach: float
    regime: MachRegime
    source: Literal["meanline_baseline", "radial_evidence"]


@dataclass(frozen=True)
class SpanLossProxy:
    """Preliminary empirical loss estimate using spanwise Mach and midspan geometry."""

    row_kind: RowKind
    span_location: Literal["hub", "midspan", "tip"]
    loss: RowLoss


@dataclass(frozen=True)
class SpanTransonicAssessment:
    """Transonic evidence and preliminary row-loss proxies at one span position."""

    span_location: Literal["hub", "midspan", "tip"]
    span_fraction: float
    mach_values: tuple[MachAssessment, ...]
    stator_proxy: SpanLossProxy
    rotor_proxy: SpanLossProxy


@dataclass(frozen=True)
class TransonicAssessment:
    """Complete hub/midspan/tip transonic assessment for one stage."""

    stage_index: int
    spans: tuple[SpanTransonicAssessment, ...]


# ---------------------------------------------------------------------------
# Gas and velocity-triangle relations
# ---------------------------------------------------------------------------
def stagnation_pressure_ratio(mach: float, gamma: float) -> float:
    """Isentropic P0 / P at the given Mach number."""
    return (1 + (gamma - 1) / 2 * mach**2) ** (gamma / (gamma - 1))


def static_density(pressure: float, temperature: float) -> float:  # kg/m^3, pressure in bar
    return pressure * BAR / (GAS.R * temperature)


def to_kacker_okapuu_angles(inlet_angle: float, exit_angle: float) -> tuple[float, float]:
    """Convert signed angles to KO's convention: exit angle positive, inlet angle positive when turning through axial."""
    exit_direction = 1.0 if exit_angle >= 0 else -1.0
    return -inlet_angle * exit_direction, abs(exit_angle)


def kinetic_loss_from_pressure_loss(pressure_loss: float, exit_mach: float, gamma: float) -> float:
    """Convert Y = (P01 - P02) / (P02 - P2) to xi = 1 - V^2 / V_is^2 at the same exit static pressure."""
    exponent = (gamma - 1) / gamma
    exit_total_to_static = stagnation_pressure_ratio(exit_mach, gamma)
    inlet_total_to_exit_static = exit_total_to_static * (1 + pressure_loss) - pressure_loss
    return 1 - (1 - exit_total_to_static**-exponent) / (1 - inlet_total_to_exit_static**-exponent)


def zweifel_pitch_chord(inlet_angle: float, exit_angle: float, axial_chord_ratio: float,
                        zweifel: float = 0.8) -> float:
    """Pitch-to-chord ratio from the Zweifel coefficient Zw = 2 (s/b_x) cos^2(a2) (tan a1 + tan a2)."""
    inlet, exit_ = map(math.radians, to_kacker_okapuu_angles(inlet_angle, exit_angle))
    pitch_axial_chord = zweifel / (2 * math.cos(exit_) ** 2 * (math.tan(inlet) + math.tan(exit_)))
    return pitch_axial_chord * axial_chord_ratio


# ---------------------------------------------------------------------------
# Kacker-Okapuu loss components (angles in KO convention, deg)
# ---------------------------------------------------------------------------
def blend_axial_inlet_and_impulse(axial_inlet_value: float, impulse_value: float, metal_exit_ratio: float) -> float:
    """KO interpolation between axial-inlet and impulse blading by the ratio of inlet metal to exit angle."""
    return axial_inlet_value + abs(metal_exit_ratio) * metal_exit_ratio * (impulse_value - axial_inlet_value)


def _metal_exit_ratio(inlet_metal_angle: float, exit_angle: float) -> float:
    if exit_angle == 0:
        raise ValueError("Kacker-Okapuu angle interpolation requires a nonzero exit angle.")
    return inlet_metal_angle / exit_angle


def ainley_mathieson_profile_loss(geometry: RowGeometry, inlet_metal_angle: float, exit_angle: float,
                                  warnings: list[str]) -> float:
    """Ainley-Mathieson profile loss Y_p,AMDC from the axial-inlet and impulse charts."""
    metal_exit_ratio = _metal_exit_ratio(inlet_metal_angle, exit_angle)
    axial_inlet_loss = AXIAL_INLET_PROFILE_LOSS.profile_loss(geometry.pitch_chord, exit_angle, warnings)
    impulse_loss = axial_inlet_loss
    if metal_exit_ratio != 0:
        impulse_loss = IMPULSE_PROFILE_LOSS.profile_loss(geometry.pitch_chord, exit_angle, warnings)

    thickness_chord = float(np.clip(geometry.max_thickness_chord, 0.15, 0.25))
    blended_loss = blend_axial_inlet_and_impulse(axial_inlet_loss, impulse_loss, metal_exit_ratio)
    return blended_loss * (thickness_chord / 0.2) ** metal_exit_ratio


def reynolds_correction(reynolds: float) -> float:
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive.")
    if reynolds <= 2e5:
        return (reynolds / 2e5) ** -0.4
    if reynolds < 1e6:
        return 1.0
    return (reynolds / 1e6) ** -0.2


def compressibility_correction(inlet_mach: float, exit_mach: float) -> float:
    """KO channel-acceleration factor K_p."""
    if inlet_mach < 0 or exit_mach <= 0:
        raise ValueError("Inlet Mach must be nonnegative and exit Mach must be positive.")
    exit_mach_factor = 1.0 if exit_mach <= 0.2 else max(0.0, 1 - 1.25 * (exit_mach - 0.2))
    inlet_exit_mach_ratio = inlet_mach / exit_mach
    return 1 - inlet_exit_mach_ratio**2 * (1 - exit_mach_factor)


def supersonic_profile_factor(exit_mach: float) -> float:
    """KO supersonic-exit factor f_Ma = 1 + 60 (M2 - 1)^2 on the profile loss (as implemented in TurboFlow)."""
    return 1.0 if exit_mach <= 1.0 else 1 + 60 * (exit_mach - 1) ** 2


def hub_mach_factor(hub_tip_ratio: float, kind: RowKind, warnings: list[str]) -> float:
    if not HUB_TIP_RATIOS[0] <= hub_tip_ratio <= HUB_TIP_RATIOS[-1]:
        warnings.append(
            f"Fig. 6: hub-to-tip ratio {hub_tip_ratio:.3f} outside "
            f"{HUB_TIP_RATIOS[0]:.1f}-{HUB_TIP_RATIOS[-1]:.1f}"
        )
    return float(np.interp(hub_tip_ratio, HUB_TIP_RATIOS, HUB_TO_MEAN_MACH_RATIO[kind]))


def hub_shock_loss(geometry: RowGeometry, flow: RowFlow, warnings: list[str]) -> float:
    """KO inlet shock loss driven by the hub inlet Mach number."""
    inlet_hub_mach = flow.inlet_mach * hub_mach_factor(geometry.hub_tip_ratio_inlet, flow.kind, warnings)
    if inlet_hub_mach <= 0.4:
        return 0.0

    hub_loss = 0.75 * (inlet_hub_mach - 0.4) ** 1.75
    inlet_dynamic_head = 1 - stagnation_pressure_ratio(flow.inlet_mach, flow.gamma)
    exit_dynamic_head = 1 - stagnation_pressure_ratio(flow.exit_mach, flow.gamma)
    pressure_ratio = flow.inlet_static_pressure / flow.exit_static_pressure
    return geometry.hub_tip_ratio_inlet * hub_loss * pressure_ratio * inlet_dynamic_head / exit_dynamic_head


def secondary_loss(geometry: RowGeometry, inlet_angle: float, exit_angle: float, inlet_metal_angle: float,
                   mach_correction: float) -> float:
    inlet, exit_ = math.radians(inlet_angle), math.radians(exit_angle)
    mean_angle = math.atan((math.tan(exit_) - math.tan(inlet)) / 2)
    lift_over_pitch_chord = 2 * (math.tan(inlet) + math.tan(exit_)) * math.cos(mean_angle)
    loading = lift_over_pitch_chord**2 * math.cos(exit_) ** 2 / math.cos(mean_angle) ** 3

    aspect_ratio = geometry.height / geometry.chord
    if aspect_ratio <= 2:
        aspect_ratio_factor = (1 - 0.25 * math.sqrt(2 - aspect_ratio)) / aspect_ratio
    else:
        aspect_ratio_factor = 1 / aspect_ratio

    angle_factor = math.cos(exit_) / math.cos(math.radians(inlet_metal_angle))
    ainley_mathieson_loss = 0.0334 * aspect_ratio_factor * angle_factor * loading
    compressibility_factor = 1 - (geometry.axial_chord / geometry.height) ** 2 * (1 - mach_correction)
    return 1.2 * ainley_mathieson_loss * compressibility_factor


def trailing_edge_loss(trailing_edge_throat: float, inlet_metal_angle: float, exit_angle: float,
                       exit_mach: float, gamma: float, warnings: list[str]) -> float:
    metal_exit_ratio = _metal_exit_ratio(inlet_metal_angle, exit_angle)
    if not 0 <= trailing_edge_throat <= 0.4:
        warnings.append(f"Fig. 14: t_TE/o = {trailing_edge_throat:.3f} outside 0-0.4")

    axial_inlet_energy_loss = max(0.0, float(np.interp(trailing_edge_throat, *AXIAL_INLET_TRAILING_EDGE.T)))
    impulse_energy_loss = max(0.0, float(np.interp(trailing_edge_throat, *IMPULSE_TRAILING_EDGE.T)))
    energy_loss = blend_axial_inlet_and_impulse(axial_inlet_energy_loss, impulse_energy_loss, metal_exit_ratio)

    exponent = -gamma / (gamma - 1)
    exit_dynamic_term = (gamma - 1) / 2 * exit_mach**2
    numerator = (1 - exit_dynamic_term * (1 / (1 - energy_loss) - 1)) ** exponent - 1
    denominator = 1 - (1 + exit_dynamic_term) ** exponent
    return numerator / denominator


def tip_efficiency_debit(geometry: RowGeometry, flow: RowFlow, exit_angle: float) -> float:
    """Unshrouded-rotor fractional efficiency debit from tip clearance."""
    if flow.kind != "rotor" or geometry.tip_clearance <= 0:
        return 0.0
    mean_radius = 0.5 * (geometry.hub_radius_exit + geometry.tip_radius_exit)
    clearance_height = geometry.tip_clearance / (geometry.height * math.cos(math.radians(exit_angle)))
    return 0.93 * clearance_height * geometry.tip_radius_exit / mean_radius


# ---------------------------------------------------------------------------
# Row loss (independent of the meanline)
# ---------------------------------------------------------------------------
def row_loss(geometry: RowGeometry, flow: RowFlow) -> RowLoss:
    """Kacker-Okapuu loss of one blade row at midspan."""
    warnings: list[str] = []
    inlet_angle, exit_angle = to_kacker_okapuu_angles(flow.inlet_angle, flow.exit_angle)
    inlet_metal_angle, _ = to_kacker_okapuu_angles(geometry.inlet_metal_angle, flow.exit_angle)

    mach_correction = compressibility_correction(flow.inlet_mach, flow.exit_mach)
    reynolds_factor = reynolds_correction(flow.reynolds)
    supersonic_factor = supersonic_profile_factor(flow.exit_mach)
    ainley_mathieson_loss = ainley_mathieson_profile_loss(geometry, inlet_metal_angle, exit_angle, warnings)
    profile_base = KO_DESIGN_LEVEL_FACTOR * KO_PROFILE_REDUCTION * ainley_mathieson_loss
    compressibility_increment = profile_base * (mach_correction - 1)
    reynolds_increment = profile_base * mach_correction * (reynolds_factor - 1)
    profile = profile_base + compressibility_increment + reynolds_increment
    hub_shock = KO_DESIGN_LEVEL_FACTOR * hub_shock_loss(geometry, flow, warnings) * reynolds_factor
    supersonic = (supersonic_factor - 1) * (profile + hub_shock)

    secondary = secondary_loss(geometry, inlet_angle, exit_angle, inlet_metal_angle, mach_correction)
    trailing_edge = trailing_edge_loss(geometry.trailing_edge_throat, inlet_metal_angle, exit_angle,
                                       flow.exit_mach, flow.gamma, warnings)

    total = math.fsum((
        profile_base,
        compressibility_increment,
        reynolds_increment,
        hub_shock,
        secondary,
        trailing_edge,
        supersonic,
    ))
    return RowLoss(
        kind=flow.kind,
        profile=profile,
        hub_shock=hub_shock,
        secondary=secondary,
        trailing_edge=trailing_edge,
        supersonic=supersonic,
        total=total,
        kinetic_loss=kinetic_loss_from_pressure_loss(total, flow.exit_mach, flow.gamma),
        tip_efficiency_debit=tip_efficiency_debit(geometry, flow, exit_angle),
        mach_correction=mach_correction,
        reynolds_correction=reynolds_factor,
        supersonic_correction=supersonic_factor,
        chart_warnings=tuple(warnings),
        profile_base=profile_base,
        compressibility_increment=compressibility_increment,
        reynolds_increment=reynolds_increment,
    )


def classify_mach(mach: float) -> MachRegime:
    if mach < 0.9:
        return "subsonic"
    if mach <= 1.0:
        return "transonic"
    return "supersonic"


def _span_row_proxy(
    meanline: MeanlineResult,
    span: SpanPosition,
    span_location: Literal["hub", "midspan", "tip"],
    row_kind: RowKind,
    airfoil: AirfoilParams,
) -> SpanLossProxy:
    if row_kind == "stator":
        geometry = row_geometry(
            airfoil,
            meanline,
            height=meanline.blade_height_2,
            inlet_angle=meanline.alpha_1,
            exit_angle=span.stator_exit.absolute_angle,
            hub_diameter_exit=meanline.D_hub_2,
            tip_diameter_exit=meanline.D_tip_2,
        )
        flow = RowFlow(
            kind="stator",
            inlet_angle=meanline.alpha_1,
            exit_angle=span.stator_exit.absolute_angle,
            inlet_mach=meanline.M_1,
            exit_mach=span.stator_exit.mach,
            inlet_static_pressure=meanline.P_01 / stagnation_pressure_ratio(
                meanline.M_1,
                meanline.inlet_air_state.gamma if meanline.inlet_air_state is not None else GAS.gamma,
            ),
            exit_static_pressure=span.stator_exit.static_pressure,
            reynolds=(
                span.stator_exit.density
                * span.stator_exit.absolute_speed
                * geometry.chord
                / span.stator_exit.viscosity_pa_s
            ),
            gamma=span.stator_exit.gamma,
        )
    else:
        geometry = row_geometry(
            airfoil,
            meanline,
            height=meanline.blade_height_3,
            inlet_angle=span.stator_exit.relative_angle,
            exit_angle=span.rotor_exit.relative_angle,
            hub_diameter_exit=meanline.D_hub_3,
            tip_diameter_exit=meanline.D_tip_3,
            tip_clearance=boundary_conditions.rotor_tip_clearance_ratio * meanline.blade_height_3,
        )
        flow = RowFlow(
            kind="rotor",
            inlet_angle=span.stator_exit.relative_angle,
            exit_angle=span.rotor_exit.relative_angle,
            inlet_mach=span.stator_exit.relative_mach,
            exit_mach=span.rotor_exit.relative_mach,
            inlet_static_pressure=span.stator_exit.static_pressure,
            exit_static_pressure=span.rotor_exit.static_pressure,
            reynolds=(
                span.rotor_exit.density
                * span.rotor_exit.relative_speed
                * geometry.chord
                / span.rotor_exit.viscosity_pa_s
            ),
            gamma=span.rotor_exit.gamma,
        )
    return SpanLossProxy(
        row_kind=row_kind,
        span_location=span_location,
        loss=row_loss(geometry, flow),
    )


def assess_transonic(
    meanline: MeanlineResult,
    spanwise: SpanwiseFlow,
    stator_airfoil: AirfoilParams,
    rotor_airfoil: AirfoilParams,
    stage_index: int = 1,
) -> TransonicAssessment:
    """Build hub/midspan/tip Mach evidence and preliminary empirical shock-loss proxies."""
    spans: list[SpanTransonicAssessment] = []
    for span_location, span_fraction, span in (
        ("hub", 0.0, spanwise.hub),
        ("midspan", 0.5, spanwise.midspan),
        ("tip", 1.0, spanwise.tip),
    ):
        mach_values = (
            MachAssessment(
                station_id=f"({stage_index}, 2)",
                frame="absolute",
                mach=span.stator_exit.mach,
                regime=classify_mach(span.stator_exit.mach),
                source="radial_evidence",
            ),
            MachAssessment(
                station_id=f"({stage_index}, 2)",
                frame="rotor_relative",
                mach=span.stator_exit.relative_mach,
                regime=classify_mach(span.stator_exit.relative_mach),
                source="radial_evidence",
            ),
            MachAssessment(
                station_id=f"({stage_index}, 3)",
                frame="absolute",
                mach=span.rotor_exit.mach,
                regime=classify_mach(span.rotor_exit.mach),
                source="radial_evidence",
            ),
            MachAssessment(
                station_id=f"({stage_index}, 3)",
                frame="rotor_relative",
                mach=span.rotor_exit.relative_mach,
                regime=classify_mach(span.rotor_exit.relative_mach),
                source="radial_evidence",
            ),
        )
        spans.append(SpanTransonicAssessment(
            span_location=span_location,
            span_fraction=span_fraction,
            mach_values=mach_values,
            stator_proxy=_span_row_proxy(
                meanline,
                span,
                span_location,
                "stator",
                stator_airfoil,
            ),
            rotor_proxy=_span_row_proxy(
                meanline,
                span,
                span_location,
                "rotor",
                rotor_airfoil,
            ),
        ))
    return TransonicAssessment(stage_index=stage_index, spans=tuple(spans))


def print_transonic_assessment(assessment: TransonicAssessment) -> None:
    print(f"Transonic assessment Stage {assessment.stage_index}")
    print(f"{'Span':<10}{'Station':>10}{'Frame':>18}{'Mach':>10}{'Regime':>12}")
    print("-" * 60)
    for span in assessment.spans:
        for mach_value in span.mach_values:
            print(
                f"{span.span_location:<10}"
                f"{mach_value.station_id:>10}"
                f"{mach_value.frame:>18}"
                f"{mach_value.mach:10.3f}"
                f"{mach_value.regime:>12}"
            )
    print("Preliminary span loss proxies use midspan geometry and are not fed back into the meanline.")


# ---------------------------------------------------------------------------
# Meanline solution -> stator and rotor losses
# ---------------------------------------------------------------------------
def row_geometry(airfoil: AirfoilParams, meanline: MeanlineResult, height: float, inlet_angle: float,
                 exit_angle: float, hub_diameter_exit: float, tip_diameter_exit: float,
                 tip_clearance: float = 0.0) -> RowGeometry:
    """Row geometry from the airfoil parameters; both rows take their inlet hub/tip ratio from station 2."""
    chord = height / airfoil.aspect_ratio
    inlet_metal_angle = inlet_angle if airfoil.inlet_metal_angle is None else airfoil.inlet_metal_angle
    return RowGeometry(
        pitch_chord=zweifel_pitch_chord(inlet_angle, exit_angle, airfoil.axial_chord_ratio, airfoil.zweifel),
        chord=chord,
        axial_chord=chord * airfoil.axial_chord_ratio,
        height=height,
        max_thickness_chord=airfoil.max_thickness_chord,
        trailing_edge_throat=airfoil.trailing_edge_throat,
        inlet_metal_angle=inlet_metal_angle,
        hub_tip_ratio_inlet=meanline.D_hub_2 / meanline.D_tip_2,
        hub_radius_exit=hub_diameter_exit / 2,
        tip_radius_exit=tip_diameter_exit / 2,
        tip_clearance=tip_clearance,
    )


def stator_row_inputs(meanline: MeanlineResult, airfoil: AirfoilParams = STATOR_AIRFOIL,
                      stage_index: int = 1) -> RowLossInputs:
    """Build the validated midspan stator loss package in the absolute frame."""
    exit_density = (
        meanline.stator_exit_air_state.density_kg_m3
        if meanline.stator_exit_air_state is not None
        else static_density(meanline.P_s2, meanline.T_2)
    )
    geometry = row_geometry(
        airfoil,
        meanline,
        height=meanline.blade_height_2,
        inlet_angle=meanline.alpha_1,
        exit_angle=meanline.alpha_2,
        hub_diameter_exit=meanline.D_hub_2,
        tip_diameter_exit=meanline.D_tip_2,
    )
    flow = RowFlow(
        kind="stator",
        inlet_angle=meanline.alpha_1,
        exit_angle=meanline.alpha_2,
        inlet_mach=meanline.M_1,
        exit_mach=meanline.M_2,
        inlet_static_pressure=meanline.P_01 / stagnation_pressure_ratio(
            meanline.M_1,
            meanline.inlet_air_state.gamma if meanline.inlet_air_state is not None else GAS.gamma,
        ),
        exit_static_pressure=meanline.P_s2,
        reynolds=exit_density * meanline.v_2 * geometry.chord / (
            meanline.stator_exit_air_state.viscosity_pa_s
            if meanline.stator_exit_air_state is not None
            else GAS.mu
        ),
        gamma=(
            meanline.stator_exit_air_state.gamma
            if meanline.stator_exit_air_state is not None
            else GAS.gamma
        ),
    )
    return RowLossInputs(
        stage_index=stage_index,
        row_kind="stator",
        reference_frame="absolute",
        span_location="midspan",
        geometry=geometry,
        flow=flow,
    )


def rotor_row_inputs(meanline: MeanlineResult, airfoil: AirfoilParams = ROTOR_AIRFOIL,
                     stage_index: int = 1) -> RowLossInputs:
    """Build the validated midspan rotor loss package in the rotor-relative frame."""
    exit_density = (
        meanline.rotor_exit_air_state.density_kg_m3
        if meanline.rotor_exit_air_state is not None
        else static_density(meanline.P_s3, meanline.T_3)
    )
    geometry = row_geometry(
        airfoil,
        meanline,
        height=meanline.blade_height_3,
        inlet_angle=meanline.beta_2,
        exit_angle=meanline.beta_3,
        hub_diameter_exit=meanline.D_hub_3,
        tip_diameter_exit=meanline.D_tip_3,
        tip_clearance=boundary_conditions.rotor_tip_clearance_ratio * meanline.blade_height_3,
    )
    flow = RowFlow(
        kind="rotor",
        inlet_angle=meanline.beta_2,
        exit_angle=meanline.beta_3,
        inlet_mach=meanline.M_2rel,
        exit_mach=meanline.M_3rel,
        inlet_static_pressure=meanline.P_s2,
        exit_static_pressure=meanline.P_s3,
        reynolds=exit_density * meanline.w_3 * geometry.chord / (
            meanline.rotor_exit_air_state.viscosity_pa_s
            if meanline.rotor_exit_air_state is not None
            else GAS.mu
        ),
        gamma=(
            meanline.rotor_exit_air_state.gamma
            if meanline.rotor_exit_air_state is not None
            else GAS.gamma
        ),
    )
    return RowLossInputs(
        stage_index=stage_index,
        row_kind="rotor",
        reference_frame="rotor_relative",
        span_location="midspan",
        geometry=geometry,
        flow=flow,
    )


def stator_losses(meanline: MeanlineResult, airfoil: AirfoilParams = STATOR_AIRFOIL,
                  stage_index: int = 1) -> RowLoss:
    """Kacker-Okapuu loss of the stator, from the stage inlet state to the stator exit."""
    row_inputs = stator_row_inputs(meanline, airfoil, stage_index)
    return row_loss(row_inputs.geometry, row_inputs.flow)


def rotor_losses(meanline: MeanlineResult, airfoil: AirfoilParams = ROTOR_AIRFOIL,
                 stage_index: int = 1) -> RowLoss:
    """Kacker-Okapuu loss of the rotor, in the relative frame."""
    row_inputs = rotor_row_inputs(meanline, airfoil, stage_index)
    return row_loss(row_inputs.geometry, row_inputs.flow)


def stage_efficiency(meanline: MeanlineResult, rotor: RowLoss) -> StageEfficiency:
    """Meanline efficiencies and work reduced by the rotor tip-clearance debit."""
    tip_clearance_factor = 1 - rotor.tip_efficiency_debit
    return StageEfficiency(
        specific_work=meanline.work_thermo * tip_clearance_factor,
        total_to_total=meanline.eta_tt * tip_clearance_factor,
        total_to_static=meanline.eta_ts * tip_clearance_factor,
        tip_clearance_debit=rotor.tip_efficiency_debit,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
BREAKDOWN_FIELDS = (
    "profile_base",
    "compressibility_increment",
    "reynolds_increment",
    "profile",
    "hub_shock",
    "secondary",
    "trailing_edge",
    "supersonic",
    "total",
    "kinetic_loss",
)
BREAKDOWN_METRICS = (
    "tip_efficiency_debit",
    "mach_correction",
    "reynolds_correction",
    "supersonic_correction",
)


def print_breakdown(row_losses: Sequence[RowLoss], names: Sequence[str] | None = None) -> None:
    names = names or [loss.kind for loss in row_losses]
    print(f"{'Loss component':<22}" + "".join(f"{name:>12}" for name in names))
    print("-" * (22 + 12 * len(names)))
    for field_name in BREAKDOWN_FIELDS:
        if field_name == "profile":
            values = [loss.profile for loss in row_losses]
        elif field_name == "profile_base":
            values = [loss.profile_base if loss.profile_base else loss.profile for loss in row_losses]
        else:
            values = [getattr(loss, field_name) for loss in row_losses]
        print(f"{field_name:<22}" + "".join(f"{value:12.4f}" for value in values))
    print(f"{'accounting_residual':<22}" + "".join(
        f"{loss.accounting_residual:12.4f}" for loss in row_losses
    ))
    for field_name in BREAKDOWN_METRICS:
        print(f"{field_name:<22}" + "".join(
            f"{getattr(loss, field_name):12.4f}" for loss in row_losses
        ))
    for name, loss in zip(names, row_losses):
        for warning in loss.chart_warnings:
            print(f"WARNING ({name}): {warning}")
