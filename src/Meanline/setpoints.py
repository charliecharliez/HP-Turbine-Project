from dataclasses import dataclass
from boundary_conditions import boundary_conditions

@dataclass(frozen=True)
class Setpoints:
    P_01: float = boundary_conditions.p_01  # bar
    T_01: float = boundary_conditions.T_01  # K
    pressure_ratio: float = boundary_conditions.pressure_ratio_TS  # total to static: P_01 / P_s3
    N_shaft: float = 13000  # RPM

    D_hub_stator: float = 0.69  # m
    D_tip_stator: float = 0.79  # m

    D_hub_rotor: float = 0.68  # m
    D_tip_rotor: float = 0.79  # m

    alpha_1: float = 0  # deg
    alpha_2: float = 71  # deg
    Beta2_metal: float = 45  # deg
    Beta3_metal: float = 61  # deg

    loss_kinetic_stator: float = 0.027
    loss_kinetic_rotor: float = 0.054


SETPOINTS = Setpoints()
