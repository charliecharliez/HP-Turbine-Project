from dataclasses import dataclass
import math
from CoolProp import CoolProp

from Constraints.boundary_conditions import boundary_conditions

# Inlet state (total) and isentropic exit state (static), from the setpoints
T_IN = boundary_conditions.T_01  # K
P_IN = boundary_conditions.P_01 * 1e5  # convert bar to Pa
P_OUT = P_IN / boundary_conditions.pressure_ratio_TS  # Pa, exit static pressure
S_IN = CoolProp.PropsSI("SMASS", "T", T_IN, "P", P_IN, "Air")
T_OUT_S = CoolProp.PropsSI("T", "P", P_OUT, "SMASS", S_IN, "Air")  # K, isentropic exit


def _cp_gamma(T: float, p: float) -> tuple[float, float]:  # T in K, p in Pa
    cp = CoolProp.PropsSI("CPMASS", "T", T, "P", p, "Air")
    cv = CoolProp.PropsSI("CVMASS", "T", T, "P", p, "Air")
    return cp, cp / cv


def _mu(T: float, p: float) -> float:  # T in K, p in Pa
    return CoolProp.PropsSI("VISCOSITY", "T", T, "P", p, "Air")


_CP_IN, _ = _cp_gamma(T_IN, P_IN)
_CP_OUT, _ = _cp_gamma(T_OUT_S, P_OUT)
_MU_IN = _mu(T_IN, P_IN)
_MU_OUT = _mu(T_OUT_S, P_OUT)
_AVERAGE_CP = 0.5 * (_CP_IN + _CP_OUT)
_AIR_GAS_CONSTANT = CoolProp.PropsSI("GAS_CONSTANT", "Air") / CoolProp.PropsSI("MOLAR_MASS", "Air")
_CONSISTENT_GAMMA = _AVERAGE_CP / (_AVERAGE_CP - _AIR_GAS_CONSTANT)


@dataclass(frozen=True)
class GasProps:
    gamma: float = _CONSISTENT_GAMMA  # ratio of specific heats
    C_p: float = _AVERAGE_CP  # J/(kg*K)
    R: float = _AIR_GAS_CONSTANT  # J/(kg*K)
    mu: float = 0.5 * (_MU_IN + _MU_OUT)  # Pa*s, dynamic viscosity


GAS = GasProps()


@dataclass(frozen=True)
class AirState:
    temperature_k: float
    pressure_pa: float
    enthalpy_j_kg: float
    entropy_j_kg_k: float
    cp_j_kg_k: float
    cv_j_kg_k: float
    gamma: float
    viscosity_pa_s: float
    density_kg_m3: float
    gas_constant_j_kg_k: float

    @classmethod
    def from_temperature_pressure(cls, temperature_k: float, pressure_pa: float) -> "AirState":
        if not math.isfinite(temperature_k) or not math.isfinite(pressure_pa):
            raise ValueError("Air state temperature and pressure must be finite")
        if temperature_k <= 0 or pressure_pa <= 0:
            raise ValueError("Air state temperature and pressure must be positive")

        enthalpy = float(CoolProp.PropsSI("HMASS", "T", temperature_k, "P", pressure_pa, "Air"))
        entropy = float(CoolProp.PropsSI("SMASS", "T", temperature_k, "P", pressure_pa, "Air"))
        cp = float(CoolProp.PropsSI("CPMASS", "T", temperature_k, "P", pressure_pa, "Air"))
        cv = float(CoolProp.PropsSI("CVMASS", "T", temperature_k, "P", pressure_pa, "Air"))
        viscosity = float(CoolProp.PropsSI("VISCOSITY", "T", temperature_k, "P", pressure_pa, "Air"))
        density = float(CoolProp.PropsSI("DMASS", "T", temperature_k, "P", pressure_pa, "Air"))
        gas_constant = _AIR_GAS_CONSTANT
        return cls(
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            enthalpy_j_kg=enthalpy,
            entropy_j_kg_k=entropy,
            cp_j_kg_k=cp,
            cv_j_kg_k=cv,
            gamma=cp / (cp - gas_constant),
            viscosity_pa_s=viscosity,
            density_kg_m3=density,
            gas_constant_j_kg_k=gas_constant,
        )

    @classmethod
    def isentropic_from_temperature_pressure(
        cls,
        temperature_k: float,
        pressure_pa: float,
        exit_pressure_pa: float,
    ) -> "AirState":
        inlet = cls.from_temperature_pressure(temperature_k, pressure_pa)
        return cls.isentropic_from_state(inlet, exit_pressure_pa)

    @classmethod
    def isentropic_from_state(
        cls,
        inlet: "AirState",
        exit_pressure_pa: float,
    ) -> "AirState":
        exit_temperature = float(
            CoolProp.PropsSI(
                "T",
                "P",
                exit_pressure_pa,
                "SMASS",
                inlet.entropy_j_kg_k,
                "Air",
            )
        )
        return cls.from_temperature_pressure(exit_temperature, exit_pressure_pa)

    @classmethod
    def from_enthalpy_pressure(
        cls,
        enthalpy_j_kg: float,
        pressure_pa: float,
    ) -> "AirState":
        temperature = float(
            CoolProp.PropsSI("T", "P", pressure_pa, "HMASS", enthalpy_j_kg, "Air")
        )
        return cls.from_temperature_pressure(temperature, pressure_pa)

    @classmethod
    def stagnation_state(cls, entropy_state: "AirState", enthalpy_j_kg: float) -> "AirState":
        pressure = float(
            CoolProp.PropsSI(
                "P",
                "H",
                enthalpy_j_kg,
                "SMASS",
                entropy_state.entropy_j_kg_k,
                "Air",
            )
        )
        return cls.from_enthalpy_pressure(enthalpy_j_kg, pressure)

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
