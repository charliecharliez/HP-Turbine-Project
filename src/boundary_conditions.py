from dataclasses import dataclass

from design_models import BoundaryConditions


@dataclass(frozen=True)
class BC:
    P_01: float = 20  # bar
    T_01: float = 1850  # K
    m_dot: float = 30  # kg/s
    pressure_ratio_TS: float = 4.1  # turbine inlet total to turbine exit static pressure

    P_0_coolant: float = 21  # bar
    T_0_coolant: float = 750  # K

    rotor_tip_clearance_ratio: float = 1e-2  # clearance / blade height
	max_metal_temperature: float = 1250 # K
    max_mean_blade_speed: float = 500  # m/s

    max_AN2: float = 3.2e7  # rotor exit annulus area times rotational speed^2 [m^2 rpm^2]


	def as_si(self) -> BoundaryConditions:
		return BoundaryConditions.from_bar(
			inlet_total_pressure_bar=self.P_01,
			inlet_total_temperature_k=self.T_01,
			main_mass_flow_kg_s=self.m_dot,
			total_to_static_pressure_ratio=self.pressure_ratio_TS,
			coolant_total_pressure_bar=self.P_0_coolant,
			coolant_total_temperature_k=self.T_0_coolant,
			rotor_tip_clearance_ratio=self.rotor_tip_clearance_ratio,
			max_metal_temperature_k=self.max_metal_temperature,
			max_mean_blade_speed_m_s=self.max_mean_blade_speed,
			max_rotor_exit_an2_m2_rpm2=self.max_AN2,
		)

	def as_si(self) -> BoundaryConditions:
		return BoundaryConditions.from_bar(
			inlet_total_pressure_bar=self.P_01,
			inlet_total_temperature_k=self.T_01,
			main_mass_flow_kg_s=self.m_dot,
			total_to_static_pressure_ratio=self.pressure_ratio_TS,
			coolant_total_pressure_bar=self.P_0_coolant,
			coolant_total_temperature_k=self.T_0_coolant,
			rotor_tip_clearance_ratio=self.rotor_tip_clearance_ratio,
			max_metal_temperature_k=self.max_metal_temperature,
			max_mean_blade_speed_m_s=self.max_mean_blade_speed,
			max_rotor_exit_an2_m2_rpm2=self.max_AN2,
		)

boundary_conditions = BC()
