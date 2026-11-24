"""Radial equilibrium at the stator and rotor exits for the vortex-law family V_theta = a r^n +/- b / r."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

from boundary_conditions import boundary_conditions, BC as bcs
from Meanline.gas_props import GAS
from Meanline.solve_meanline import Inputs, MeanlineResult

Station = Literal[2, 3]  # 2: stator exit, 3: rotor exit
Radius = float | np.ndarray  # m

BAR = 1e5  # Pa
SWIRL_SIGN: dict[int, float] = {2: 1.0, 3: -1.0}  # sign of the b / r term at each station
CONTINUITY_GRID_POINTS = 401

FREE_VORTEX = -1.0  # r V_theta constant, uniform axial velocity
EXPONENTIAL = 0.0  # V_theta = a +/- b / r
FIRST_POWER = 1.0  # V_theta = a r +/- b / r


# ---------------------------------------------------------------------------
# Annulus and vortex law
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Annulus:
    hub_radius: float  # m
    tip_radius: float  # m

    @classmethod
    def from_diameters(cls, hub_diameter: float, tip_diameter: float) -> Annulus:
        return cls(hub_diameter / 2, tip_diameter / 2)

    @property
    def mean_radius(self) -> float:
        return (self.hub_radius + self.tip_radius) / 2

    def radius_at(self, span_fraction: Radius) -> Radius:
        return self.hub_radius + span_fraction * (self.tip_radius - self.hub_radius)

    def span_fraction(self, radius: Radius) -> Radius:
        return (radius - self.hub_radius) / (self.tip_radius - self.hub_radius)

    def radial_grid(self) -> np.ndarray:
        return np.linspace(self.hub_radius, self.tip_radius, CONTINUITY_GRID_POINTS)


def antiderivative_of_power(radius: Radius, power: float) -> Radius:
    """Antiderivative of r^power."""
    if power == -1:
        return np.log(radius)
    return radius ** (power + 1) / (power + 1)


@dataclass(frozen=True)
class VortexLaw:
    """Swirl V_theta = a r^n + b / r at the stator exit and a r^n - b / r at the rotor exit [m/s]."""

    exponent: float  # n
    a: float
    b: float

    @classmethod
    def through_mean_swirl(cls, exponent: float, mean_radius_2: float, mean_swirl_2: float,
                           mean_radius_3: float, mean_swirl_3: float) -> VortexLaw:
        """Law whose swirl matches the meanline at both mean radii."""
        coefficients = np.array([[mean_radius_2**exponent, 1 / mean_radius_2],
                                 [mean_radius_3**exponent, -1 / mean_radius_3]])
        a, b = np.linalg.solve(coefficients, [mean_swirl_2, mean_swirl_3])
        return cls(exponent, float(a), float(b))

    def tangential_velocity(self, radius: Radius, station: Station) -> Radius:
        return self.a * radius**self.exponent + SWIRL_SIGN[station] * self.b / radius

    def axial_velocity_squared_shape(self, radius: Radius, station: Station) -> Radius:
        """Cx^2 up to an additive constant, from radial equilibrium with uniform total enthalpy and entropy."""
        n = self.exponent
        power_term = self.a * antiderivative_of_power(radius, 2 * n - 1)
        inverse_radius_term = SWIRL_SIGN[station] * self.b * antiderivative_of_power(radius, n - 2)
        return -2 * self.a * (n + 1) * (power_term + inverse_radius_term)


def axial_velocity_constant(law: VortexLaw, station: Station, annulus: Annulus, mass_flow: float,
                            density: float) -> float:
    """Constant C in Cx^2 = shape(r) + C that passes the mass flow [kg/s], with density held at its meanline value."""
    radii = annulus.radial_grid()
    shape = law.axial_velocity_squared_shape(radii, station)

    def mass_flow_error(constant: float) -> float:
        axial_velocity = np.sqrt(shape + constant)
        return 2 * math.pi * density * np.trapezoid(axial_velocity * radii, radii) - mass_flow

    smallest_constant = 1e-6 - shape.min()
    return root_scalar(mass_flow_error, bracket=(smallest_constant, 1e8), method="brentq").root


# ---------------------------------------------------------------------------
# Spanwise states
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StationState:
    """Flow at one radius of one station; velocities in m/s with tangential components positive along rotation."""

    radius: float  # m
    axial_velocity: float
    tangential_velocity: float
    blade_speed: float
    static_temperature: float  # K
    static_pressure: float  # bar
    density: float  # kg/m^3
    mach: float
    relative_mach: float

    @property
    def relative_tangential_velocity(self) -> float:
        return self.tangential_velocity - self.blade_speed

    @property
    def absolute_speed(self) -> float:
        return math.hypot(self.axial_velocity, self.tangential_velocity)

    @property
    def relative_speed(self) -> float:
        return math.hypot(self.axial_velocity, self.relative_tangential_velocity)

    @property
    def absolute_angle(self) -> float:  # deg
        return math.degrees(math.atan2(self.tangential_velocity, self.axial_velocity))

    @property
    def relative_angle(self) -> float:  # deg
        return math.degrees(math.atan2(self.relative_tangential_velocity, self.axial_velocity))


@dataclass(frozen=True)
class SpanPosition:
    span_fraction: float  # 0 at hub, 1 at tip
    stator_exit: StationState
    rotor_exit: StationState
    specific_work: float  # J/kg
    degree_of_reaction: float  # meanline definition: rotor share of the isentropic static enthalpy drop


@dataclass(frozen=True)
class SpanwiseFlow:
    """Radial-equilibrium flow at the stator exit (2) and rotor exit (3)."""

    law: VortexLaw
    annulus: dict[int, Annulus]
    axial_velocity_constant: dict[int, float]  # m^2/s^2
    mean_static_pressure: dict[int, float]  # bar
    omega: float  # rad/s
    inlet_total_temperature: float  # K
    inlet_total_pressure: float  # bar

    def degree_of_reaction(self, stator_exit_pressure: float, rotor_exit_pressure: float) -> float:
        """Meanline definition, inverting its P_s2 = P_01 (1 - (1 - R)(1 - (P_s3 / P_01)^((k-1)/k)))^(k/(k-1))."""
        exponent = (GAS.gamma - 1) / GAS.gamma
        stator_exit_ratio = (stator_exit_pressure / self.inlet_total_pressure) ** exponent
        rotor_exit_ratio = (rotor_exit_pressure / self.inlet_total_pressure) ** exponent
        return (stator_exit_ratio - rotor_exit_ratio) / (1 - rotor_exit_ratio)

    def axial_velocity(self, radius: Radius, station: Station) -> Radius:
        shape = self.law.axial_velocity_squared_shape(radius, station)
        return np.sqrt(shape + self.axial_velocity_constant[station])

    def stator_exit_radius(self, rotor_exit_radius: Radius) -> Radius:
        """Stator-exit radius on the same streamline, mapping hub to hub and tip to tip."""
        return self.annulus[2].radius_at(self.annulus[3].span_fraction(rotor_exit_radius))

    def specific_work(self, rotor_exit_radius: Radius) -> Radius:  # J/kg
        stator_exit_radius = self.stator_exit_radius(rotor_exit_radius)
        angular_momentum_2 = stator_exit_radius * self.law.tangential_velocity(stator_exit_radius, 2)
        angular_momentum_3 = rotor_exit_radius * self.law.tangential_velocity(rotor_exit_radius, 3)
        return self.omega * (angular_momentum_2 - angular_momentum_3)

    def total_temperature(self, radius: Radius, station: Station) -> Radius:  # K
        if station == 2:
            return self.inlet_total_temperature
        return self.inlet_total_temperature - self.specific_work(radius) / GAS.C_p

    def static_temperature(self, radius: Radius, station: Station) -> Radius:  # K
        axial_velocity = self.axial_velocity(radius, station)
        tangential_velocity = self.law.tangential_velocity(radius, station)
        kinetic_temperature = (axial_velocity**2 + tangential_velocity**2) / (2 * GAS.C_p)
        return self.total_temperature(radius, station) - kinetic_temperature

    def static_pressure(self, radius: float, station: Station) -> float:  # bar
        """Integrates dP/dr = rho V_theta^2 / r from the meanline pressure at the mean radius."""
        def log_pressure_gradient(r: float) -> float:
            tangential_velocity = self.law.tangential_velocity(r, station)
            return tangential_velocity**2 / (GAS.R * self.static_temperature(r, station) * r)

        integral, _ = quad(log_pressure_gradient, self.annulus[station].mean_radius, radius)
        return self.mean_static_pressure[station] * math.exp(integral)

    def station_state(self, radius: float, station: Station) -> StationState:
        gas = GAS
        axial_velocity = float(self.axial_velocity(radius, station))
        tangential_velocity = float(self.law.tangential_velocity(radius, station))
        blade_speed = self.omega * radius
        static_temperature = float(self.static_temperature(radius, station))
        static_pressure = self.static_pressure(radius, station)
        speed_of_sound = math.sqrt(gas.gamma * gas.R * static_temperature)
        return StationState(
            radius=radius,
            axial_velocity=axial_velocity,
            tangential_velocity=tangential_velocity,
            blade_speed=blade_speed,
            static_temperature=static_temperature,
            static_pressure=static_pressure,
            density=static_pressure * BAR / (gas.R * static_temperature),
            mach=math.hypot(axial_velocity, tangential_velocity) / speed_of_sound,
            relative_mach=math.hypot(axial_velocity, tangential_velocity - blade_speed) / speed_of_sound,
        )

    def at_span(self, span_fraction: float) -> SpanPosition:
        stator_exit = self.station_state(self.annulus[2].radius_at(span_fraction), 2)
        rotor_exit = self.station_state(self.annulus[3].radius_at(span_fraction), 3)
        return SpanPosition(
            span_fraction=span_fraction,
            stator_exit=stator_exit,
            rotor_exit=rotor_exit,
            specific_work=float(self.specific_work(rotor_exit.radius)),
            degree_of_reaction=self.degree_of_reaction(stator_exit.static_pressure, rotor_exit.static_pressure),
        )

    @property
    def hub(self) -> SpanPosition:
        return self.at_span(0.0)

    @property
    def midspan(self) -> SpanPosition:
        return self.at_span(0.5)

    @property
    def tip(self) -> SpanPosition:
        return self.at_span(1.0)


# ---------------------------------------------------------------------------
# Meanline solution -> spanwise flow
# ---------------------------------------------------------------------------
def solve_radial_equilibrium(meanline: MeanlineResult, inputs: Inputs, vortex_exponent: float = FREE_VORTEX) -> SpanwiseFlow:
    """Spanwise flow whose swirl, mass flow and static pressure match the meanline at the mean radii."""
    annulus_2 = Annulus.from_diameters(meanline.D_hub_2, meanline.D_tip_2)
    annulus_3 = Annulus.from_diameters(meanline.D_hub_3, meanline.D_tip_3)
    mean_swirl_2 = meanline.v_2 * math.sin(math.radians(inputs.alpha_2))
    mean_swirl_3 = meanline.v_3 * math.sin(math.radians(meanline.alpha_3))
    law = VortexLaw.through_mean_swirl(vortex_exponent, annulus_2.mean_radius, mean_swirl_2,
                                       annulus_3.mean_radius, mean_swirl_3)

    exit_static_pressure = bcs.P_01 / bcs.pressure_ratio_TS  # bar
    density_2 = meanline.P_s2 * BAR / (GAS.R * meanline.T_2)
    density_3 = exit_static_pressure * BAR / (GAS.R * meanline.T_3)

    return SpanwiseFlow(
        law=law,
        annulus={2: annulus_2, 3: annulus_3},
        axial_velocity_constant={
            2: axial_velocity_constant(law, 2, annulus_2, meanline.m_dot, density_2),
            3: axial_velocity_constant(law, 3, annulus_3, meanline.m_dot, density_3),
        },
        mean_static_pressure={2: meanline.P_s2, 3: exit_static_pressure},
        omega=inputs.N_shaft * 2 * math.pi / 60,
        inlet_total_temperature=bcs.T_01,
        inlet_total_pressure=bcs.P_01,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
SPANWISE_TABLE_ROWS: tuple[tuple[str, Callable[[SpanPosition], float], str], ...] = (
    ("alpha_2 [deg]", lambda span: span.stator_exit.absolute_angle, ".2f"),
    ("beta_2 [deg]", lambda span: span.stator_exit.relative_angle, ".2f"),
    ("alpha_3 [deg]", lambda span: span.rotor_exit.absolute_angle, ".2f"),
    ("beta_3 [deg]", lambda span: span.rotor_exit.relative_angle, ".2f"),
    ("M_2", lambda span: span.stator_exit.mach, ".3f"),
    ("M_2rel", lambda span: span.stator_exit.relative_mach, ".3f"),
    ("M_3", lambda span: span.rotor_exit.mach, ".3f"),
    ("M_3rel", lambda span: span.rotor_exit.relative_mach, ".3f"),
    ("P_2 [bar]", lambda span: span.stator_exit.static_pressure, ".3f"),
    ("P_3 [bar]", lambda span: span.rotor_exit.static_pressure, ".3f"),
    ("T_2 [K]", lambda span: span.stator_exit.static_temperature, ".1f"),
    ("T_3 [K]", lambda span: span.rotor_exit.static_temperature, ".1f"),
    ("rho_2 [kg/m^3]", lambda span: span.stator_exit.density, ".3f"),
    ("rho_3 [kg/m^3]", lambda span: span.rotor_exit.density, ".3f"),
    ("C_x2 [m/s]", lambda span: span.stator_exit.axial_velocity, ".1f"),
    ("V_theta2 [m/s]", lambda span: span.stator_exit.tangential_velocity, ".1f"),
    ("U_2 [m/s]", lambda span: span.stator_exit.blade_speed, ".1f"),
    ("C_x3 [m/s]", lambda span: span.rotor_exit.axial_velocity, ".1f"),
    ("V_theta3 [m/s]", lambda span: span.rotor_exit.tangential_velocity, ".1f"),
    ("U_3 [m/s]", lambda span: span.rotor_exit.blade_speed, ".1f"),
    ("Work [kJ/kg]", lambda span: span.specific_work / 1e3, ".1f"),
    ("Degree of reaction", lambda span: span.degree_of_reaction, ".3f"),
)


def print_spanwise_table(flow: SpanwiseFlow) -> None:
    positions = {"hub": flow.hub, "midspan": flow.midspan, "tip": flow.tip}
    print(f"Vortex exponent n = {flow.law.exponent:g}")
    print(f"{'Quantity':<20}" + "".join(f"{name:>12}" for name in positions))
    print("-" * (20 + 12 * len(positions)))
    for label, value_of, number_format in SPANWISE_TABLE_ROWS:
        print(f"{label:<20}" + "".join(f"{value_of(span):12{number_format}}" for span in positions.values()))
