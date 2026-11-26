"""Meanline design of one stage at the stator exit (station 2) and rotor exit (station 3)."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
from scipy.optimize import brentq, root_scalar

from boundary_conditions import boundary_conditions
from velocity_triangle import VelocityTriangle

from .gas_props import GAS

BAR = 1e5  # Pa


# Angles in deg from axial, signed positive along rotation. Pressures in bar.
@dataclass(frozen=True)
class Inputs:
    degree_of_reaction: float = 0.3
    N_shaft: float = 13000  # rpm
    radius_2: float = 0.5  # m, stator exit mean radius
    radius_3: float = 0.6  # m, rotor exit mean radius
    alpha_2: float = 71  # deg, stator exit absolute flow angle
    beta_3: float = -61  # deg, rotor exit relative flow angle
    loss_kinetic_stator: float = 0.027  # xi = 1 - V^2 / V_is^2
    loss_kinetic_rotor: float = 0.054  # xi = 1 - W^2 / W_is^2
    pressure_ratio: float = boundary_conditions.pressure_ratio_TS  # P_01 / P_s3 of this stage


@dataclass(frozen=True)
class StageInlet:
    T_0: float = boundary_conditions.T_01  # K
    P_0: float = boundary_conditions.P_01  # bar
    m_dot: float = boundary_conditions.m_dot  # kg/s
    alpha: float = 0.0  # deg, absolute flow angle
    mach: float | None = None  # absolute; None estimates it from continuity through the stator exit annulus


@dataclass(frozen=True)
class MeanlineResult:
    residual: float  # kg/s, m_dot minus the rotor exit mass flow
    m_dot: float  # kg/s
    omega: float  # rad/s

    # Stage inlet
    alpha_1: float  # deg
    M_1: float

    # Thermodynamic states (K, bar)
    T_01: float
    P_01: float
    P_s2: float
    T_2: float
    P_s3: float
    T_3: float
    T_03: float
    P_03: float

    # Velocity triangles (m/s, deg)
    v_2: float
    w_2: float
    u_2: float
    alpha_2: float
    beta_2: float
    v_3: float
    w_3: float
    u_3: float
    alpha_3: float
    beta_3: float
    M_2: float
    M_2rel: float
    M_3: float
    M_3rel: float

    # Annulus (m, m^2)
    blade_height_2: float
    blade_height_3: float
    D_hub_2: float
    D_tip_2: float
    D_hub_3: float
    D_tip_3: float
    area_3: float

    # Performance
    work_euler: float  # J/kg, u_2 v_theta_2 - u_3 v_theta_3
    work_thermo: float  # J/kg, C_p (T_01 - T_03)
    eta_ts: float  # before the rotor tip-clearance debit
    eta_tt: float  # before the rotor tip-clearance debit
    stage_loading: float  # work_euler / u_mean^2
    flow_coefficient: float  # mean axial velocity across the rotor / u_mean

    def exit_state(self) -> StageInlet:
        """Rotor exit flow as the inlet of a following stage."""
        return StageInlet(T_0=self.T_03, P_0=self.P_03, m_dot=self.m_dot, alpha=self.alpha_3, mach=self.M_3)


def subsonic_mach_number(mass_flow: float, area: float, total_temperature: float, total_pressure: float) -> float:
    """Subsonic axial Mach number passing the mass flow [kg/s] through the area [m^2]; total pressure in bar."""
    gamma, R = GAS.gamma, GAS.R
    choking_exponent = -(gamma + 1) / (2 * (gamma - 1))

    def mass_flow_at(mach: float) -> float:
        flow_function = mach * (1 + (gamma - 1) / 2 * mach**2) ** choking_exponent
        return total_pressure * BAR * area * math.sqrt(gamma / (R * total_temperature)) * flow_function

    solution = root_scalar(lambda mach: mass_flow_at(mach) - mass_flow, bracket=(1e-6, 1.0), method="brentq")
    return solution.root


def _solve_states(blade_height_3: float, inputs: Inputs, inlet: StageInlet) -> MeanlineResult:
    C_p, k, R = GAS.C_p, GAS.gamma, GAS.R
    exponent = (k - 1) / k
    T_01, P_01, m_dot = inlet.T_0, inlet.P_0, inlet.m_dot
    P_s3 = P_01 / inputs.pressure_ratio
    P_s2 = P_01 * (1 - (1 - inputs.degree_of_reaction) * (1 - (P_s3 / P_01) ** exponent)) ** (1 / exponent)

    omega = inputs.N_shaft * 2 * math.pi / 60
    u_2 = omega * inputs.radius_2
    u_3 = omega * inputs.radius_3

    # Stator exit
    T_2is = T_01 * (P_s2 / P_01) ** exponent
    v_2 = math.sqrt((1 - inputs.loss_kinetic_stator) * 2 * C_p * (T_01 - T_2is))
    T_2 = T_01 - v_2**2 / (2 * C_p)
    alpha_2 = math.radians(inputs.alpha_2)
    v_x2 = v_2 * math.cos(alpha_2)
    v_theta_2 = v_2 * math.sin(alpha_2)
    w_2 = math.hypot(v_x2, v_theta_2 - u_2)
    density_2 = P_s2 * BAR / (R * T_2)
    blade_height_2 = m_dot / (density_2 * v_x2) / (2 * math.pi * inputs.radius_2)
    speed_of_sound_2 = math.sqrt(k * R * T_2)

    # Rotor exit, with rothalpy conserved
    T_3is = T_2 * (P_s3 / P_s2) ** exponent
    T_03_relative = T_2 + (w_2**2 + u_3**2 - u_2**2) / (2 * C_p)
    w_3 = math.sqrt((1 - inputs.loss_kinetic_rotor) * 2 * C_p * (T_03_relative - T_3is))
    T_3 = T_03_relative - w_3**2 / (2 * C_p)
    beta_3 = math.radians(inputs.beta_3)
    v_x3 = w_3 * math.cos(beta_3)
    v_theta_3 = u_3 + w_3 * math.sin(beta_3)
    v_3 = math.hypot(v_x3, v_theta_3)
    T_03 = T_3 + v_3**2 / (2 * C_p)
    P_03 = P_s3 * (T_03 / T_3) ** (1 / exponent)
    density_3 = P_s3 * BAR / (R * T_3)
    area_3 = 2 * math.pi * inputs.radius_3 * blade_height_3
    speed_of_sound_3 = math.sqrt(k * R * T_3)

    # Performance
    work_euler = u_2 * v_theta_2 - u_3 * v_theta_3
    work_thermo = C_p * (T_01 - T_03)
    u_mean = (u_2 + u_3) / 2
    ideal_drop_static = C_p * T_01 * (1 - (P_s3 / P_01) ** exponent)
    ideal_drop_total = C_p * T_01 * (1 - (P_03 / P_01) ** exponent)

    return MeanlineResult(
        residual=m_dot - density_3 * v_x3 * area_3,
        m_dot=m_dot,
        omega=omega,
        alpha_1=inlet.alpha,
        M_1=math.nan if inlet.mach is None else inlet.mach,
        T_01=T_01, P_01=P_01, P_s2=P_s2, T_2=T_2, P_s3=P_s3, T_3=T_3, T_03=T_03, P_03=P_03,
        v_2=v_2, w_2=w_2, u_2=u_2,
        alpha_2=inputs.alpha_2,
        beta_2=math.degrees(math.atan2(v_theta_2 - u_2, v_x2)),
        v_3=v_3, w_3=w_3, u_3=u_3,
        alpha_3=math.degrees(math.atan2(v_theta_3, v_x3)),
        beta_3=inputs.beta_3,
        M_2=v_2 / speed_of_sound_2,
        M_2rel=w_2 / speed_of_sound_2,
        M_3=v_3 / speed_of_sound_3,
        M_3rel=w_3 / speed_of_sound_3,
        blade_height_2=blade_height_2,
        blade_height_3=blade_height_3,
        D_hub_2=2 * inputs.radius_2 - blade_height_2,
        D_tip_2=2 * inputs.radius_2 + blade_height_2,
        D_hub_3=2 * inputs.radius_3 - blade_height_3,
        D_tip_3=2 * inputs.radius_3 + blade_height_3,
        area_3=area_3,
        work_euler=work_euler,
        work_thermo=work_thermo,
        eta_ts=work_thermo / ideal_drop_static,
        eta_tt=work_thermo / ideal_drop_total,
        stage_loading=work_euler / u_mean**2,
        flow_coefficient=(v_x2 + v_x3) / 2 / u_mean,
    )


def solve_meanline(inputs: Inputs = Inputs(), inlet: StageInlet = StageInlet()) -> MeanlineResult:
    """Size the rotor exit blade height so the rotor exit passes the stage mass flow."""
    def residual(blade_height_3: float) -> float:
        return _solve_states(blade_height_3, inputs, inlet).residual

    largest_blade_height = 2 * inputs.radius_3
    if residual(largest_blade_height) > 0:
        raise ValueError("Cannot pass m_dot through the rotor exit even with hub radius 0; "
                         "check radius_3 / angles / reaction.")
    blade_height_3 = brentq(residual, 1e-6, largest_blade_height, xtol=1e-12)
    states = _solve_states(blade_height_3, inputs, inlet)
    if inlet.mach is not None:
        return states
    stator_exit_area = 2 * math.pi * inputs.radius_2 * states.blade_height_2
    return replace(states, M_1=subsonic_mach_number(inlet.m_dot, stator_exit_area, inlet.T_0, inlet.P_0))


def plot_velocity_triangles(result: MeanlineResult) -> None:
    """Velocity triangles at stations 2 (solid) and 3 (dashed) from a common apex."""
    station_2 = VelocityTriangle.from_U_C_W(result.u_2, result.v_2, result.w_2, station=2)
    station_3 = VelocityTriangle.from_U_C_W(result.u_3, result.v_3, result.w_3, station=3)
    VelocityTriangle.plot_combined(
        station_2, station_3,
        title="Meanline velocity triangles: station 2 (rotor inlet) and station 3 (rotor exit)")
    plt.show()
