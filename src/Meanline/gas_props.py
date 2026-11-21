from dataclasses import dataclass
from CoolProp import CoolProp

from boundary_conditions import boundary_conditions

# Inlet state (total) and isentropic exit state (static), from the setpoints
T_IN = boundary_conditions.T_01  # K
P_IN = boundary_conditions.p_01 * 1e-5  # convert Pa to bar
P_OUT = P_IN / boundary_conditions.pressure_ratio_TS  # Pa, exit static pressure
S_IN = CoolProp.PropsSI("SMASS", "T", T_IN, "P", P_IN, "Air")
T_OUT_S = CoolProp.PropsSI("T", "P", P_OUT, "SMASS", S_IN, "Air")  # K, isentropic exit


def _cp_gamma(T, p):
    cp = CoolProp.PropsSI("CPMASS", "T", T, "P", p, "Air")
    cv = CoolProp.PropsSI("CVMASS", "T", T, "P", p, "Air")
    return cp, cp / cv


def _mu(T, p):
    return CoolProp.PropsSI("VISCOSITY", "T", T, "P", p, "Air")


_CP_IN, _GAMMA_IN = _cp_gamma(T_IN, P_IN)
_CP_OUT, _GAMMA_OUT = _cp_gamma(T_OUT_S, P_OUT)
_MU_IN = _mu(T_IN, P_IN)
_MU_OUT = _mu(T_OUT_S, P_OUT)


@dataclass(frozen=True)
class GasProps:
    # Averaged properties between the inlet (total) and the isentropic exit
    gamma: float = 0.5 * (_GAMMA_IN + _GAMMA_OUT)  # ratio of specific heats
    C_p: float = 0.5 * (_CP_IN + _CP_OUT)  # J/(kg*K)
    R: float = CoolProp.PropsSI("GAS_CONSTANT", "Air") / CoolProp.PropsSI("MOLAR_MASS", "Air")  # J/(kg*K)
    mu: float = 0.5 * (_MU_IN + _MU_OUT)  # Pa*s, dynamic viscosity

GAS = GasProps()

if __name__ == '__main__':
    # Compare properties at turbine inlet vs. outlet to justify constant C_p and gamma
    ETA = 0.90  # assumed total-to-static efficiency, for the actual exit temperature
    T_out = T_IN - ETA * (T_IN - T_OUT_S)

    print(f"{'State':28s}{'T [K]':>9s}{'p [bar]':>9s}{'C_p [J/kgK]':>13s}{'gamma':>9s}"
          f"{'mu [uPa*s]':>12s}{'dC_p [%]':>10s}{'dgamma [%]':>11s}{'dmu [%]':>9s}")
    for name, T, p in [("Inlet (total)", T_IN, P_IN),
                       ("Outlet, isentropic (static)", T_OUT_S, P_OUT),
                       (f"Outlet, eta_ts={ETA} (static)", T_out, P_OUT)]:
        cp, g = _cp_gamma(T, p)
        mu = _mu(T, p)
        print(f"{name:28s}{T:9.1f}{p / 1e5:9.2f}{cp:13.1f}{g:9.4f}{mu * 1e6:12.2f}"
              f"{100 * (cp / GAS.C_p - 1):10.2f}{100 * (g / GAS.gamma - 1):11.2f}"
              f"{100 * (mu / GAS.mu - 1):9.2f}")
    print(f"{'Average (used)':28s}{'':9s}{'':9s}{GAS.C_p:13.1f}{GAS.gamma:9.4f}{GAS.mu * 1e6:12.2f}")
    print("dC_p, dgamma and dmu are relative to the averaged values.")
