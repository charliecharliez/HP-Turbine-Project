import json
import math
from pathlib import Path
from boundary_conditions import boundary_conditions

from .gas_props import GAS, GasProps
from dataclasses import dataclass, field

@dataclass
class Inputs:
    degree_of_reaction: float = 0.3
    N_shaft: float = 13000  # RPM

    D_hub_stator: float = 0.69  # m
    D_tip_stator: float = 0.79  # m

    D_hub_rotor: float = 0.68  # m
    D_tip_rotor: float = 0.79  # m

    #alpha_1: float = 0  # deg
    alpha_2: float = 71  # deg
    Beta2_metal: float = 45  # deg
    Beta3_metal: float = 61  # deg

    loss_kinetic_stator: float = 0.027 #assumed for now
    loss_kinetic_rotor: float = 0.054 # assumed for now

def _solve_states(gas: GasProps = GAS, inputs: Inputs=Inputs()):

    C_p = gas.C_p  # J/(kg*K) specific heat with constant pressure
    k = gas.gamma  # ratio of specific heats
    R = gas.R  # gas constant

    T_01 = boundary_conditions.T_01 # total temperature at inlet, Kelvin
    P_01 = boundary_conditions.P_01  # total pressure at inlet, bar
    P_s3 = P_01 / boundary_conditions.pressure_ratio_TS  # static pressure at exit, bar

    # static pressure at stator exit, depends on degree of reaction
    P_s2 = (1 - (1 - inputs.degree_of_reaction) * (1 - (P_s3 / P_01) ** ((k - 1) / k))) ** (k / (k - 1)) * P_01

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
    v_2 = math.sqrt(1 - inputs.loss_kinetic_stator) * v_2is

    # use total temp T0 = Ts + v^2 / (2*C_p) again to solve for non-isentropic static T_2
    # total temp T_02 = T_02is since no work is done
    T_2 = T_02 - v_2**2 / (2 * C_p)  # Kelvin

    # isentropic relations
    P_02 = (T_02 / T_2) ** (k / (k - 1)) * P_s2

    # velocity triangle
    omega = inputs.N_shaft * (2 * math.pi) / 60  # angular speed rad/s
    avg_stator_tip_radius = (inputs.D_hub_stator + inputs.D_tip_stator) / 4  # m
    # Even though the stator is not rotating, the meanline position is defined by stator geometry
    u_2 = omega * avg_stator_tip_radius  # m/s
    alpha_2 = math.radians(inputs.alpha_2)
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

    avg_rotor_tip_radius = (inputs.D_hub_rotor + inputs.D_tip_rotor) / 4  # m
    u_3 = omega * avg_rotor_tip_radius  # m/s
    T_03R = T_02R + (u_3**2 - u_2**2) / (2 * C_p)

    # T_03Ris = T_03is + w_3is^2 / (2 * C_p)
    # where T_03R = T_03Ris
    w_3is = math.sqrt(2 * C_p * (T_03R - T_3is))
    # use kinematic loss to find w_3
    w_3 = math.sqrt(1 - inputs.loss_kinetic_rotor) * w_3is
    T_3 = T_03R - w_3**2 / (2 * C_p)  # definition of total temperature

    Beta3 = math.radians(inputs.Beta3_metal)  # geometric parameter, reasonably accurate

    v_3 = math.sqrt((w_3 * math.cos(Beta3)) ** 2 + (w_3 * math.sin(Beta3) - u_3) ** 2)

    alpha_3 = math.acos(w_3 * math.cos(Beta3) / v_3)

    # ------------------------Continuity equations to solve for mass flow---------------------------
    stator_area = math.pi / 4 * (inputs.D_tip_stator**2 - inputs.D_hub_stator**2)
    rotor_area = math.pi / 4 * (inputs.D_tip_rotor**2 - inputs.D_hub_rotor**2)

    # Solve for density using ideal gas law
    # bar * 10^5 / (J / kg / K * K) = kg / m^3
    density_2 = P_s2 * 1e5 / (R * T_2)
    density_3 = P_s3 * 1e5 / (R * T_3)

    # Meridional velocities (normal to areas)
    v_2m = math.cos(alpha_2) * v_2
    v_3m = math.cos(alpha_3) * v_3

    # Mass flow
    m_dot_2 = v_2m * stator_area * density_2
    m_dot_3 = v_3m * rotor_area * density_3

    residual = m_dot_2 - m_dot_3

    return residual