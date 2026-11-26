from dataclasses import dataclass


@dataclass(frozen=True)
class BC:
    P_01: float = 20  # bar
    T_01: float = 1850  # K
    m_dot: float = 30  # kg/s
    pressure_ratio_TS: float = 4.1  # turbine inlet total to turbine exit static pressure

    P_0_coolant: float = 21  # bar
    T_0_coolant: float = 750  # K

    rotor_tip_clearance_ratio: float = 1e-2  # clearance / blade height
    max_mean_blade_speed: float = 500  # m/s

    max_AN2: float = 3.2e7  # rotor exit annulus area times rotational speed^2 [m^2 rpm^2]

    rig_min_hub_tip_ratio: float = 0.70  # AMBRA, for the rotor it tests
    rig_max_hub_tip_ratio: float = 0.94


boundary_conditions = BC()
