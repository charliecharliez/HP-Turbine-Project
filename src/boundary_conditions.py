from dataclasses import dataclass

@dataclass(frozen=True)
class BC:
	p_01 = 20 # bar
	T_01 = 1850 # K
	m_dot = 30 # kg/s
	pressure_ratio_TS = 4.1 # turbine total to static pressure ratio from inlet to exit

	p_0_coolant = 21 # bar
	T_0_coolant = 750 # K

	rotor_tip_clearance_ratio = 1e-2 # clearance is 1% of blade height
	max_mean_blade_speed = 500 # m/s

	max_AN2 = 3.2e-7 # rotor exit annulus area times rotational speed^2 [m^2 * rpm^2] 

boundary_conditions = BC()