import json
import math
from pathlib import Path

from .gas_props import GAS, GasProps
from .setpoints import SETPOINTS, Setpoints
from dataclasses import dataclass, field

def _solve_states(r: float, gas: GasProps = GAS, sp: Setpoints = SETPOINTS) -> tuple[float, dict]:
    """Meanline states for degree of reaction r.

    Returns (residual, solution), where residual = m_dot_2 - m_dot_3 [kg/s] and
    solution maps each state name to (value, unit).
    """
    solution: dict[str, tuple[float, str]] = {}

    # r is the degree of reaction
    solution['degree of reaction'] = (r, 'unitless')

    C_p = gas.C_p  # J/(kg*K) specific heat with constant pressure
    k = gas.gamma  # ratio of specific heats
    R = gas.R  # gas constant

    T_01 = sp.T_01  # total temperature at inlet, Kelvin
    P_01 = sp.P_01  # total pressure at inlet, bar
    P_s3 = P_01 / sp.pressure_ratio  # static pressure at exit, bar

    # static pressure at stator exit, depends on degree of reaction
    P_s2 = (1 - (1 - r) * (1 - (P_s3 / P_01) ** ((k - 1) / k))) ** (k / (k - 1)) * P_01
    solution['P_s2'] = (P_s2, 'bar')

    # Assuming an isentropic stator:
    # T_2is can be found using the isentropic relation (T_2is/T_01) = (P_s2/P_02)^((k-1)/k)
    # = (P_s2/P_01)^((k-1)/k)
    T_2is = T_01 * (P_s2 / P_01) ** ((k - 1) / k)
    solution['T_2is'] = (T_2is, 'K')

    T_02 = T_01  # stator does not change total temperature; no work done

    # use total temperature T0 = Ts + v^2 / (2*C_p) to solve for V_2is
    # (J/kg/K * K)^1/2 = (J/kg)^(1/2) = (m/s^2 * m)^(1/2) = m/s
    v_2is = math.sqrt(2 * C_p * (T_02 - T_2is))  # m/s
    solution['v_2is'] = (v_2is, 'm/s')

    # use kinematic loss of stator to find true v_2
    # loss = 1 - v^2 / v_is^2
    v_2 = math.sqrt(1 - sp.loss_kinetic_stator) * v_2is
    solution['v_2'] = (v_2, 'm/s')

    # use total temp T0 = Ts + v^2 / (2*C_p) again to solve for non-isentropic static T_2
    # total temp T_02 = T_02is since no work is done
    T_2 = T_02 - v_2**2 / (2 * C_p)  # Kelvin
    solution['T_2'] = (T_2, 'K')

    # isentropic relations
    P_02 = (T_02 / T_2) ** (k / (k - 1)) * P_s2
    solution['P_02'] = (P_02, 'bar')

    # velocity triangle
    omega = sp.N_shaft * (2 * math.pi) / 60  # angular speed rad/s
    avg_stator_tip_radius = (sp.D_hub_stator + sp.D_tip_stator) / 4  # m
    # Even though the stator is not rotating, the meanline position is defined by stator geometry
    u_2 = omega * avg_stator_tip_radius  # m/s
    solution['u_2'] = (u_2, 'm/s')
    alpha_2 = math.radians(sp.alpha_2)
    w_2 = math.sqrt((v_2 * math.cos(alpha_2)) ** 2 + (v_2 * math.sin(alpha_2) - u_2) ** 2)
    solution['w_2'] = (w_2, 'm/s')

    M_2 = v_2 / math.sqrt(k * R * T_2)  # Mach number at station 2
    solution['M_2'] = (M_2, 'unitless')
    M_2rel = w_2 / math.sqrt(k * R * T_2)
    solution['M_2rel'] = (M_2rel, 'unitless')
    Beta2 = math.degrees(math.acos(v_2 * math.cos(alpha_2) / w_2))
    solution['Beta2'] = (Beta2, 'deg')

    # 02 in relative frame of reference of the rotor

    # isentropic relations to find P_02R
    P_02R = P_s2 * (T_02 / T_2) ** (k / (k - 1))
    solution['P_02R'] = (P_02R, 'bar')
    # similar to the process for stator,
    # first assume rotor process (2 -> 3) to be isentropic (2 -> 3s) to find T_3is
    T_3is = T_2 * (P_s3 / P_s2) ** ((k - 1) / k)
    solution['T_3is'] = (T_3is, 'K')

    # Conservation of rothalpy to find T_03
    # T_02R - U_2^2 / (2*C_p) = T_03R - U_3^2 / (2 * C_p)

    T_02R = T_2 + w_2**2 / (2 * C_p)  # total temperature in relative frame of reference of the rotor
    solution['T_02R'] = (T_02R, 'K')

    avg_rotor_tip_radius = (sp.D_hub_rotor + sp.D_tip_rotor) / 4  # m
    u_3 = omega * avg_rotor_tip_radius  # m/s
    solution['u_3'] = (u_3, 'm/s')
    T_03R = T_02R + (u_3**2 - u_2**2) / (2 * C_p)
    solution['T_03R'] = (T_03R, 'K')

    # T_03Ris = T_03is + w_3is^2 / (2 * C_p)
    # where T_03R = T_03Ris
    w_3is = math.sqrt(2 * C_p * (T_03R - T_3is))
    solution['w_3is'] = (w_3is, 'm/s')
    # use kinematic loss to find w_3
    w_3 = math.sqrt(1 - sp.loss_kinetic_rotor) * w_3is
    solution['w_3'] = (w_3, 'm/s')
    T_3 = T_03R - w_3**2 / (2 * C_p)  # definition of total temperature
    solution['T_3'] = (T_3, 'K')

    Beta3 = math.radians(sp.Beta3_metal)  # geometric parameter, reasonably accurate

    v_3 = math.sqrt((w_3 * math.cos(Beta3)) ** 2 + (w_3 * math.sin(Beta3) - u_3) ** 2)
    solution['v_3'] = (v_3, 'm/s')

    alpha_3 = math.acos(w_3 * math.cos(Beta3) / v_3)
    solution['alpha_3'] = (math.degrees(alpha_3), 'deg')

    # ------------------------Continuity equations to solve for mass flow---------------------------
    stator_area = math.pi / 4 * (sp.D_tip_stator**2 - sp.D_hub_stator**2)
    rotor_area = math.pi / 4 * (sp.D_tip_rotor**2 - sp.D_hub_rotor**2)

    solution['stator_area'] = (stator_area, 'm^2')
    solution['rotor_area'] = (rotor_area, 'm^2')

    # Solve for density using ideal gas law
    # bar * 10^5 / (J / kg / K * K) = kg / m^3
    density_2 = P_s2 * 1e5 / (R * T_2)
    density_3 = P_s3 * 1e5 / (R * T_3)

    solution['density_2'] = (density_2, 'kg/m^3')
    solution['density_3'] = (density_3, 'kg/m^3')

    # Meridional velocities (normal to areas)
    v_2m = math.cos(alpha_2) * v_2
    v_3m = math.cos(alpha_3) * v_3

    solution['v_2m'] = (v_2m, 'm/s')
    solution['v_3m'] = (v_3m, 'm/s')

    # Mass flow
    m_dot_2 = v_2m * stator_area * density_2
    m_dot_3 = v_3m * rotor_area * density_3

    solution['m_dot_2'] = (m_dot_2, 'kg/s')
    solution['m_dot_3'] = (m_dot_3, 'kg/s')

    residual = m_dot_2 - m_dot_3
    solution['residual'] = (residual, 'kg/s')

    return residual, solution

CONVERGENCE_CRITERIA_ORDER = 7

SOLUTION_CACHE = Path(__file__).with_name('meanline_solution.json')


def save_solution(solution: dict, path: Path = SOLUTION_CACHE) -> None:
    """Cache the solution (state name -> (value, unit)) as JSON; tuples come back as lists."""
    with open(path, 'w') as f:
        json.dump(solution, f, indent=4)


@dataclass
class SolveResult:
    degree_of_reaction: float
    residual: float  # kg/s
    iterations: int
    solution: dict  # state name -> (value, unit), see solve_states
    residuals: list[float] = field(default_factory=list)  # residual history per iteration


def Solve(
    gas: GasProps = GAS,
    sp: Setpoints = SETPOINTS,
    guess_low: float = 0,
    guess_high: float = 1,
    tol: float = 10**-CONVERGENCE_CRITERIA_ORDER,
) -> SolveResult:
    """Bisect on degree of reaction until stator and rotor mass flows match."""
    residual_high, _ = _solve_states(guess_high, gas, sp)
    residual_low, _ = _solve_states(guess_low, gas, sp)

    if math.copysign(1, residual_high) == math.copysign(1, residual_low):
        raise ValueError('No root bracketed')

    residuals = []
    iter_count = 0

    while True:
        iter_count += 1
        guess = (guess_high + guess_low) / 2
        residual, solution = _solve_states(guess, gas, sp)
        residuals.append(residual)

        if abs(residual) < tol:
            break

        # update bounds based on sign
        if residual * residual_high > 0:
            residual_high = residual
            guess_high = guess
        else:
            residual_low = residual
            guess_low = guess

    solution['degree_of_reaction'] = guess
    
    return SolveResult(guess, residual, iter_count, solution, residuals)

def main():
    result = Solve()
    print(result.residual)
    print(result.degree_of_reaction)
    print(result.solution['P_s2'])

    # cache the solution so other scripts can skip the solve
    save_solution(result.solution)
    print(f'Saved solution to {SOLUTION_CACHE}')

if __name__ == '__main__':
    main()