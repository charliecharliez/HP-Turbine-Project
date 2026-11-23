"""Kacker-Okapuu (1982) stator and rotor losses."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import root_scalar

from boundary_conditions import BC, boundary_conditions
from Meanline.gas_props import GAS, GasProps
from Meanline.solve_meanline import Inputs, MeanlineResult

RowKind = Literal["stator", "rotor"]

BAR = 1e5  # Pa
STATOR_INLET_FLOW_ANGLE = 0.0  # deg, axial inlet per the boundary conditions


# ---------------------------------------------------------------------------
# Digitized Kacker-Okapuu charts (NASA turbo-design digitization of KO 1982)
# ---------------------------------------------------------------------------
CHART_DIRECTORY = Path(__file__).with_name("data")


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
HUB_MACH_FACTOR: dict[RowKind, float] = {"stator": 1.8, "rotor": 5.2}  # K in M_hub / M = 1 + K |r_hub/r_tip - 1|^2.2


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


@dataclass(frozen=True)
class StageEfficiency:
    specific_work: float  # J/kg
    exit_total_pressure: float  # bar
    total_to_total: float
    total_to_static: float
    total_to_total_without_tip: float
    total_to_static_without_tip: float


# ---------------------------------------------------------------------------
# Gas and velocity-triangle relations
# ---------------------------------------------------------------------------
def stagnation_pressure_ratio(mach: float, gamma: float) -> float:
    """Isentropic P0 / P at the given Mach number."""
    return (1 + (gamma - 1) / 2 * mach**2) ** (gamma / (gamma - 1))


def mach_number(speed: float, static_temperature: float, gas: GasProps) -> float:
    return speed / math.sqrt(gas.gamma * gas.R * static_temperature)


def static_density(pressure: float, temperature: float, gas: GasProps) -> float:  # kg/m^3, pressure in bar
    return pressure * BAR / (gas.R * temperature)


def annulus_area(hub_diameter: float, tip_diameter: float) -> float:  # m^2
    return math.pi / 4 * (tip_diameter**2 - hub_diameter**2)


def turbine_exit_static_pressure(bcs: BC) -> float:  # bar
    return bcs.P_01 / bcs.pressure_ratio_TS


def subsonic_inlet_mach(mass_flow: float, area: float, total_temperature: float, total_pressure: float,
                        gas: GasProps) -> float:
    """Subsonic Mach number passing the mass flow [kg/s] through the area [m^2]; total pressure in Pa."""
    gamma, R = gas.gamma, gas.R
    choking_exponent = -(gamma + 1) / (2 * (gamma - 1))

    def mass_flow_at(mach: float) -> float:
        flow_function = mach * (1 + (gamma - 1) / 2 * mach**2) ** choking_exponent
        return total_pressure * area * math.sqrt(gamma / (R * total_temperature)) * flow_function

    solution = root_scalar(lambda mach: mass_flow_at(mach) - mass_flow, bracket=(1e-6, 1.0), method="brentq")
    return solution.root


def relative_flow_angle(absolute_speed: float, absolute_angle: float, blade_speed: float) -> float:
    """Relative flow angle [deg] from the absolute speed and angle [deg], signed positive along rotation."""
    absolute_angle_rad = math.radians(absolute_angle)
    relative_tangential_speed = absolute_speed * math.sin(absolute_angle_rad) - blade_speed
    axial_speed = absolute_speed * math.cos(absolute_angle_rad)
    return math.degrees(math.atan2(relative_tangential_speed, axial_speed))


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


def ainley_mathieson_profile_loss(geometry: RowGeometry, inlet_metal_angle: float, exit_angle: float,
                                  warnings: list[str]) -> float:
    """Ainley-Mathieson profile loss Y_p,AMDC from the axial-inlet and impulse charts."""
    metal_exit_ratio = inlet_metal_angle / exit_angle
    axial_inlet_loss = AXIAL_INLET_PROFILE_LOSS.profile_loss(geometry.pitch_chord, exit_angle, warnings)
    impulse_loss = axial_inlet_loss
    if metal_exit_ratio != 0:
        impulse_loss = IMPULSE_PROFILE_LOSS.profile_loss(geometry.pitch_chord, exit_angle, warnings)

    thickness_chord = float(np.clip(geometry.max_thickness_chord, 0.15, 0.25))
    blended_loss = blend_axial_inlet_and_impulse(axial_inlet_loss, impulse_loss, metal_exit_ratio)
    return blended_loss * (thickness_chord / 0.2) ** metal_exit_ratio


def reynolds_correction(reynolds: float) -> float:
    if reynolds <= 2e5:
        return (reynolds / 2e5) ** -0.4
    if reynolds < 1e6:
        return 1.0
    return (reynolds / 1e6) ** -0.2


def compressibility_correction(inlet_mach: float, exit_mach: float) -> float:
    """KO channel-acceleration factor K_p."""
    exit_mach_factor = 1.0 if exit_mach <= 0.2 else max(0.0, 1 - 1.25 * (exit_mach - 0.2))
    inlet_exit_mach_ratio = inlet_mach / exit_mach
    return 1 - inlet_exit_mach_ratio**2 * (1 - exit_mach_factor)


def supersonic_profile_factor(exit_mach: float) -> float:
    """KO supersonic-exit factor f_Ma = 1 + 60 (M2 - 1)^2 on the profile loss (as implemented in TurboFlow)."""
    return 1.0 if exit_mach <= 1.0 else 1 + 60 * (exit_mach - 1) ** 2


def hub_shock_loss(geometry: RowGeometry, flow: RowFlow) -> float:
    """KO inlet shock loss driven by the hub inlet Mach number."""
    hub_mach_rise = HUB_MACH_FACTOR[flow.kind] * abs(geometry.hub_tip_ratio_inlet - 1) ** 2.2
    inlet_hub_mach = flow.inlet_mach * (1 + hub_mach_rise)
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
    if not 0 <= trailing_edge_throat <= 0.4:
        warnings.append(f"Fig. 14: t_TE/o = {trailing_edge_throat:.3f} outside 0-0.4")

    axial_inlet_energy_loss = max(0.0, np.interp(trailing_edge_throat, *AXIAL_INLET_TRAILING_EDGE.T))
    impulse_energy_loss = max(0.0, np.interp(trailing_edge_throat, *IMPULSE_TRAILING_EDGE.T))
    energy_loss = blend_axial_inlet_and_impulse(axial_inlet_energy_loss, impulse_energy_loss,
                                                inlet_metal_angle / exit_angle)

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
    profile = KO_DESIGN_LEVEL_FACTOR * KO_PROFILE_REDUCTION * ainley_mathieson_loss * mach_correction * reynolds_factor
    hub_shock = KO_DESIGN_LEVEL_FACTOR * hub_shock_loss(geometry, flow) * reynolds_factor
    supersonic = (supersonic_factor - 1) * (profile + hub_shock)

    secondary = secondary_loss(geometry, inlet_angle, exit_angle, inlet_metal_angle, mach_correction)
    trailing_edge = trailing_edge_loss(geometry.trailing_edge_throat, inlet_metal_angle, exit_angle,
                                       flow.exit_mach, flow.gamma, warnings)

    total = profile + hub_shock + secondary + trailing_edge + supersonic
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
    )


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


def stator_losses(meanline: MeanlineResult, inputs: Inputs, gas: GasProps = GAS,
                  airfoil: AirfoilParams = STATOR_AIRFOIL, bcs: BC = boundary_conditions) -> RowLoss:
    """Kacker-Okapuu loss of the stator; its inlet annulus is taken equal to its exit annulus."""
    inlet_area = annulus_area(meanline.D_hub_2, meanline.D_tip_2)
    inlet_mach = subsonic_inlet_mach(meanline.m_dot, inlet_area, bcs.T_01, bcs.P_01 * BAR, gas)
    exit_density = static_density(meanline.P_s2, meanline.T_2, gas)

    geometry = row_geometry(
        airfoil,
        meanline,
        height=meanline.blade_height_2,
        inlet_angle=STATOR_INLET_FLOW_ANGLE,
        exit_angle=inputs.alpha_2,
        hub_diameter_exit=meanline.D_hub_2,
        tip_diameter_exit=meanline.D_tip_2,
    )
    flow = RowFlow(
        kind="stator",
        inlet_angle=STATOR_INLET_FLOW_ANGLE,
        exit_angle=inputs.alpha_2,
        inlet_mach=inlet_mach,
        exit_mach=meanline.M_2,
        inlet_static_pressure=bcs.P_01 / stagnation_pressure_ratio(inlet_mach, gas.gamma),
        exit_static_pressure=meanline.P_s2,
        reynolds=exit_density * meanline.v_2 * geometry.chord / gas.mu,
        gamma=gas.gamma,
    )
    return row_loss(geometry, flow)


def rotor_losses(meanline: MeanlineResult, inputs: Inputs, gas: GasProps = GAS,
                 airfoil: AirfoilParams = ROTOR_AIRFOIL, bcs: BC = boundary_conditions) -> RowLoss:
    """Kacker-Okapuu loss of the rotor, in the relative frame."""
    relative_inlet_angle = relative_flow_angle(meanline.v_2, inputs.alpha_2, meanline.u_2)
    relative_exit_angle = relative_flow_angle(meanline.v_3, meanline.alpha_3, meanline.u_3)
    exit_static_pressure = turbine_exit_static_pressure(bcs)
    exit_density = static_density(exit_static_pressure, meanline.T_3, gas)

    geometry = row_geometry(
        airfoil,
        meanline,
        height=meanline.blade_height_3,
        inlet_angle=relative_inlet_angle,
        exit_angle=relative_exit_angle,
        hub_diameter_exit=meanline.D_hub_3,
        tip_diameter_exit=meanline.D_tip_3,
        tip_clearance=bcs.rotor_tip_clearance_ratio * meanline.blade_height_3,
    )
    flow = RowFlow(
        kind="rotor",
        inlet_angle=relative_inlet_angle,
        exit_angle=relative_exit_angle,
        inlet_mach=meanline.M_2rel,
        exit_mach=mach_number(meanline.w_3, meanline.T_3, gas),
        inlet_static_pressure=meanline.P_s2,
        exit_static_pressure=exit_static_pressure,
        reynolds=exit_density * meanline.w_3 * geometry.chord / gas.mu,
        gamma=gas.gamma,
    )
    return row_loss(geometry, flow)


def stage_efficiency(meanline: MeanlineResult, rotor: RowLoss, gas: GasProps = GAS,
                     bcs: BC = boundary_conditions) -> StageEfficiency:
    """Uncooled stage efficiencies, with the unshrouded rotor tip-clearance debit applied."""
    exponent = (gas.gamma - 1) / gas.gamma
    exit_static_pressure = turbine_exit_static_pressure(bcs)
    exit_total_pressure = exit_static_pressure * (meanline.T_03 / meanline.T_3) ** (1 / exponent)
    total_temperature_drop = bcs.T_01 - meanline.T_03

    ideal_drop_total = bcs.T_01 * (1 - (exit_total_pressure / bcs.P_01) ** exponent)
    ideal_drop_static = bcs.T_01 * (1 - (exit_static_pressure / bcs.P_01) ** exponent)
    total_to_total = total_temperature_drop / ideal_drop_total
    total_to_static = total_temperature_drop / ideal_drop_static
    tip_clearance_factor = 1 - rotor.tip_efficiency_debit

    return StageEfficiency(
        specific_work=meanline.work_thermo,
        exit_total_pressure=exit_total_pressure,
        total_to_total=total_to_total * tip_clearance_factor,
        total_to_static=total_to_static * tip_clearance_factor,
        total_to_total_without_tip=total_to_total,
        total_to_static_without_tip=total_to_static,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
BREAKDOWN_FIELDS = ("profile", "hub_shock", "secondary", "trailing_edge", "supersonic", "total", "kinetic_loss",
                    "tip_efficiency_debit", "mach_correction", "reynolds_correction", "supersonic_correction")


def print_breakdown(row_losses: Sequence[RowLoss], names: Sequence[str] | None = None) -> None:
    names = names or [loss.kind for loss in row_losses]
    print(f"{'Loss component':<22}" + "".join(f"{name:>12}" for name in names))
    print("-" * (22 + 12 * len(names)))
    for field_name in BREAKDOWN_FIELDS:
        print(f"{field_name:<22}" + "".join(f"{getattr(loss, field_name):12.4f}" for loss in row_losses))
    for name, loss in zip(names, row_losses):
        for warning in loss.chart_warnings:
            print(f"WARNING ({name}): {warning}")
