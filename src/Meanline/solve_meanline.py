import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from boundary_conditions import boundary_conditions
from velocity_triangle import VelocityTriangle

from .gas_props import GAS, GasProps
from dataclasses import dataclass, field

@dataclass
class Inputs:
    degree_of_reaction: float = 0.3
    N_shaft: float = 13000  # RPM

    radius_2: float = 0.5 # m
    radius_3: float = 0.6 # m

    #alpha_1: float = 0  # deg
    alpha_2: float = 71  # deg
    #Beta2_metal: float = 45  # deg
    Beta3_metal: float = 61  # deg

    loss_kinetic_stator: float = 0.027 # assumed for now
    loss_kinetic_rotor: float = 0.054 # assumed for now
    pressure_ratio: float = boundary_conditions.pressure_ratio_TS

@dataclass
class StageInlet:
    T_0: float = boundary_conditions.T_01  # K, total temperature
    P_0: float = boundary_conditions.P_01  # bar, total pressure
    m_dot: float = boundary_conditions.m_dot  # kg/s

@dataclass
class MeanlineResult:
    residual: float  # kg/s, m_dot - m_dot_3
    m_dot: float
    T_01: float  # K
    P_01: float  # bar
    P_s2: float  # bar
    P_s3: float  # bar
    T_2: float
    T_3: float
    T_03: float
    P_03: float  # bar
    v_2: float
    w_2: float
    w_3: float
    v_3: float
    u_2: float
    u_3: float
    Beta2: float  # deg (unsigned)
    alpha_3: float  # deg, signed (+ along rotation)
    M_2: float
    M_2rel: float
    blade_height_2: float
    blade_height_3: float
    D_hub_2: float
    D_tip_2: float
    D_hub_3: float
    D_tip_3: float
    area_3: float
    work_euler: float  # J/kg, u2*vtheta2 - u3*vtheta3
    work_thermo: float  # J/kg, cp*(T_01 - T_03); must equal work_euler
    eta_ts: float
    eta_tt: float
    stage_loading: float  # work_euler / u_mean^2


def _solve_states(blade_height_3: float, inputs: Inputs, inlet: StageInlet) -> MeanlineResult:

    # angles (in velocity triangles) are measured starting fom the axial direction

    p_ratio = inputs.pressure_ratio
    reaction = inputs.degree_of_reaction
    loss_stator = inputs.loss_kinetic_stator
    loss_rotor = inputs.loss_kinetic_rotor
    r_2 = inputs.radius_2
    r_3 = inputs.radius_3
    Beta_3b = inputs.Beta3_metal
    alpha_2 = inputs.alpha_2
    rpm = inputs.N_shaft

    C_p = GAS.C_p  # J/(kg*K) specific heat with constant pressure
    k = GAS.gamma  # ratio of specific heats
    R = GAS.R  # gas constant

    m_dot = inlet.m_dot  # mass flowrate in kg/s

    T_01 = inlet.T_0  # total temperature at inlet, Kelvin
    P_01 = inlet.P_0  # total pressure at inlet, bar
    P_s3 = P_01 / p_ratio  # static pressure at exit, bar

    # static pressure at stator exit, depends on degree of reaction
    P_s2 = (1 - (1 - reaction) * (1 - (P_s3 / P_01) ** ((k - 1) / k))) ** (k / (k - 1)) * P_01

    # Assuming an isentropic stator:
    # T_2is can be found using the isentropic relation (T_2is/T_01) = (P_s2/P_02)^((k-1)/k)
    # = (P_s2/P_01)^((k-1)/k)
    T_2is = T_01 * (P_s2 / P_01) ** ((k - 1) / k)

    T_02 = T_01  # stator does not change total temperature; no work done

    # use total temperature T0 = Ts + v^2 / (2*C_p) to solve for V_2is
    # (J/kg/K * K)^1/2 = (J/kg)^(1/2) = (m/s^2 * m)^(1/2) = m/s
    v_2is = math.sqrt(2 * C_p * (T_02 - T_2is))  # m/s

    # use kinematic loss of stator to find true v_2
    # loss = 1 - v^2 / v_is^2
    v_2 = math.sqrt(1 - loss_stator) * v_2is

    # use total temp T0 = Ts + v^2 / (2*C_p) again to solve for non-isentropic static T_2
    # total temp T_02 = T_02is since no work is done
    T_2 = T_02 - v_2**2 / (2 * C_p)  # Kelvin

    # P = density * R * T
    density_2 = P_s2 * 1e5 / (R * T_2) # kg/m^3 (pressures are in bar)
    alpha_2 = math.radians(alpha_2)
    
    v_2a = v_2 * math.cos(alpha_2)
    area_2 = m_dot / density_2 / v_2a # m^2

    # A = pi * ((r_mean + h/2)^2 - (r_mean - h/2)^2)
    blade_height_2 = area_2 / (2 * math.pi * r_2)
    D_hub_2, D_tip_2 = 2 * (r_2 - blade_height_2/2), 2 * (r_2 + blade_height_2/2)

    # isentropic relations
    P_02 = (T_02 / T_2) ** (k / (k - 1)) * P_s2

    # velocity triangle
    omega = rpm * (2 * math.pi) / 60  # angular speed rad/s
    # Even though the stator is not rotating, the meanline position is defined by stator geometry
    u_2 = omega * r_2  # m/s
    
    w_2 = math.sqrt((v_2 * math.cos(alpha_2)) ** 2 + (v_2 * math.sin(alpha_2) - u_2) ** 2)

    M_2 = v_2 / math.sqrt(k * R * T_2)  # Mach number at station 2
    M_2rel = w_2 / math.sqrt(k * R * T_2)
    Beta2 = math.degrees(math.acos(v_2 * math.cos(alpha_2) / w_2))

    # 02 in relative frame of reference of the rotor

    # isentropic relations to find P_02R
    P_02R = P_s2 * (T_02 / T_2) ** (k / (k - 1))
    # similar to the process for stator,
    # first assume rotor process (2 -> 3) to be isentropic (2 -> 3s) to find T_3is
    T_3is = T_2 * (P_s3 / P_s2) ** ((k - 1) / k)

    # Conservation of rothalpy to find T_03
    # T_02R - U_2^2 / (2*C_p) = T_03R - U_3^2 / (2 * C_p)

    T_02R = T_2 + w_2**2 / (2 * C_p)  # total temperature in relative frame of reference of the rotor

    u_3 = omega * r_3  # m/s
    T_03R = T_02R + (u_3**2 - u_2**2) / (2 * C_p)

    # T_03Ris = T_03is + w_3is^2 / (2 * C_p)
    # where T_03R = T_03Ris
    w_3is = math.sqrt(2 * C_p * (T_03R - T_3is))
    # use kinematic loss to find w_3
    w_3 = math.sqrt(1 - loss_rotor) * w_3is
    T_3 = T_03R - w_3**2 / (2 * C_p)  # definition of total temperature

    Beta3 = math.radians(Beta_3b)  # geometric parameter, reasonably accurate

    v_3 = math.sqrt((w_3 * math.cos(Beta3)) ** 2 + (w_3 * math.sin(Beta3) - u_3) ** 2)

    alpha_3 = math.acos(w_3 * math.cos(Beta3) / v_3)
    v_3a = v_3 * math.cos(alpha_3)
    density_3 = P_s3 * 1e5 / (R * T_3) # kg / m^3 (pressures are in bar)
    area_3 = math.pi * ((r_3 + blade_height_3/2)**2 - (r_3 - blade_height_3/2)**2)
    residual = m_dot - density_3 * v_3a * area_3

    # Signed tangential velocities (+ along rotation) for the Euler cross-check.
    # The unsigned acos() angles above lose the swirl direction.
    v_theta_2 = v_2 * math.sin(alpha_2)
    v_theta_3 = u_3 - w_3 * math.sin(Beta3)
    T_03 = T_3 + v_3**2 / (2 * C_p)
    P_03 = P_s3 * (T_03 / T_3) ** (k / (k - 1))
    work_euler = u_2 * v_theta_2 - u_3 * v_theta_3  # J/kg
    work_thermo = C_p * (T_01 - T_03)  # J/kg

    stage_loading = work_euler / (omega * (r_2 + r_3) / 2) ** 2
    exponent = (k - 1) / k
    eta_ts = work_thermo / (C_p * T_01 * (1 - (P_s3 / P_01) ** exponent))
    eta_tt = work_thermo / (C_p * T_01 * (1 - (P_03 / P_01) ** exponent))

    return MeanlineResult(
        residual=residual,
        m_dot=m_dot,
        T_01=T_01, P_01=P_01,
        P_s2=P_s2, P_s3=P_s3, T_2=T_2, T_3=T_3, T_03=T_03,
        v_2=v_2, w_2=w_2, w_3=w_3, v_3=v_3, u_2=u_2, u_3=u_3,
        Beta2=Beta2, alpha_3=math.degrees(math.atan2(v_theta_3, v_3a)),
        M_2=M_2, M_2rel=M_2rel,
        blade_height_2=blade_height_2, blade_height_3=blade_height_3,
        D_hub_2=D_hub_2, D_tip_2=D_tip_2,
        D_hub_3=2 * (r_3 - blade_height_3 / 2),
        D_tip_3=2 * (r_3 + blade_height_3 / 2),
        area_3=area_3,
        work_euler=work_euler, work_thermo=work_thermo,
        eta_ts=eta_ts,
        eta_tt=eta_tt,
        stage_loading=stage_loading,
        P_03=P_03,
    )


def solve_meanline(inputs: Inputs = Inputs(), inlet: StageInlet = StageInlet()) -> MeanlineResult:
    """Size the rotor exit blade height so mass flow at station 3 equals m_dot.

    Radii and degree of reaction are inputs; the unknown is blade_height_3.
    """
    def residual(h: float) -> float:
        return _solve_states(h, inputs, inlet).residual

    # residual(0) = +m_dot; it goes negative once the annulus can pass more than m_dot.
    h_max = 2 * inputs.radius_3  # hub radius -> 0
    if residual(h_max) > 0:
        raise ValueError("Cannot pass m_dot through the rotor exit even with hub radius 0; "
                         "check radius_3 / angles / reaction.")
    h_3 = brentq(residual, 1e-6, h_max, xtol=1e-12)
    return _solve_states(h_3, inputs, inlet)

def plot_velocity_triangles(result: MeanlineResult) -> None:
    """Show the velocity triangles at station 2 (stator exit / rotor inlet) and
    station 3 (rotor exit) in one plot, with C, W and C_m of both starting from
    a common apex (station 2 solid, station 3 dashed).

    Each triangle is built from the magnitudes u, v and w, so the signs of the
    flow angles follow from the geometry rather than from the unsigned Beta2.
    """
    station_2 = VelocityTriangle.from_U_C_W(result.u_2, result.v_2, result.w_2, station=2)
    station_3 = VelocityTriangle.from_U_C_W(result.u_3, result.v_3, result.w_3, station=3)
    VelocityTriangle.plot_combined(
        station_2, station_3,
        title="Meanline velocity triangles: station 2 (rotor inlet) and station 3 (rotor exit)")
    plt.show()
