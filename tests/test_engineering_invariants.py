import math
import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from Meanline.gas_props import AirState, GAS
from Meanline.solve_meanline import solve_meanline
from RadialEquilibrium.radial_equilibrium_solver import (
    Annulus,
    EXPONENTIAL,
    FIRST_POWER,
    FREE_VORTEX,
    VortexLaw,
    axial_velocity_constant,
    solve_radial_equilibrium,
)
from velocity_triangle import VelocityTriangle


class EngineeringInvariantTests(unittest.TestCase):
    def test_velocity_triangle_closes_euler_velocity_relation(self) -> None:
        triangle = VelocityTriangle.from_U_Cm_alpha(U=100, C_m=200, alpha=30)

        self.assertAlmostEqual(
            triangle.C**2 - triangle.W**2,
            2 * triangle.U * triangle.C_u - triangle.U**2,
            places=8,
        )

    def test_radial_continuity_case_recovers_uniform_axial_flow(self) -> None:
        annulus = Annulus(hub_radius=0.2, tip_radius=0.4)
        law = VortexLaw(exponent=-1, a=0, b=0)
        mass_flow = 2.0
        density = 1.2

        constant = axial_velocity_constant(law, 2, annulus, mass_flow, density)
        radii = annulus.radial_grid()
        axial_velocity = np.sqrt(
            law.axial_velocity_squared_shape(radii, 2) + constant
        )
        recovered_mass_flow = (
            2 * math.pi * density * np.trapezoid(axial_velocity * radii, radii)
        )

        self.assertAlmostEqual(recovered_mass_flow, mass_flow, places=7)

    def test_annulus_span_mapping_has_hub_and_tip_limits(self) -> None:
        annulus = Annulus(hub_radius=0.2, tip_radius=0.4)

        self.assertAlmostEqual(annulus.radius_at(0.0), 0.2)
        self.assertAlmostEqual(annulus.radius_at(1.0), 0.4)

    def test_vortex_shape_derivative_covers_supported_exponents(self) -> None:
        radius = 0.3
        step = 1e-6

        for exponent in (FREE_VORTEX, EXPONENTIAL, FIRST_POWER):
            law = VortexLaw(exponent=exponent, a=20.0, b=5.0)
            for station in (2, 3):
                with self.subTest(exponent=exponent, station=station):
                    shape_derivative = (
                        law.axial_velocity_squared_shape(radius + step, station)
                        - law.axial_velocity_squared_shape(radius - step, station)
                    ) / (2 * step)
                    swirl = law.tangential_velocity(radius, station)
                    swirl_derivative = (
                        law.tangential_velocity(radius + step, station)
                        - law.tangential_velocity(radius - step, station)
                    ) / (2 * step)
                    expected_derivative = -2 * swirl * (swirl_derivative + swirl / radius)

                    self.assertAlmostEqual(shape_derivative, expected_derivative, delta=1e-4)

    def test_radial_pressure_preserves_entropy_across_vortex_law_family(self) -> None:
        meanline = solve_meanline()

        for exponent in (FREE_VORTEX, EXPONENTIAL, FIRST_POWER):
            flow = solve_radial_equilibrium(meanline, exponent)
            for station in (2, 3):
                annulus = flow.annulus[station]
                reference = flow.station_state(annulus.mean_radius, station)
                radii = np.linspace(annulus.hub_radius, annulus.tip_radius, 9)

                for radius in radii:
                    state = flow.station_state(float(radius), station)
                    reference_air = AirState.from_temperature_pressure(
                        reference.static_temperature,
                        reference.static_pressure * 1e5,
                    )
                    local_air = AirState.from_temperature_pressure(
                        state.static_temperature,
                        state.static_pressure * 1e5,
                    )

                    with self.subTest(exponent=exponent, station=station, radius=radius):
                        self.assertAlmostEqual(
                            local_air.entropy_j_kg_k,
                            reference_air.entropy_j_kg_k,
                            delta=5e-3,
                        )

    def test_spanwise_mass_flow_matches_with_local_density_within_point_one_percent(self) -> None:
        meanline = solve_meanline()

        for exponent in (FREE_VORTEX, EXPONENTIAL, FIRST_POWER):
            flow = solve_radial_equilibrium(meanline, exponent)
            for station in (2, 3):
                radii = np.linspace(
                    flow.annulus[station].hub_radius,
                    flow.annulus[station].tip_radius,
                    51,
                )
                states = [flow.station_state(float(radius), station) for radius in radii]
                mass_flow = np.trapezoid(
                    [
                        2 * math.pi * state.radius * state.density * state.axial_velocity
                        for state in states
                    ],
                    radii,
                )

                with self.subTest(exponent=exponent, station=station):
                    relative_error = mass_flow / meanline.m_dot - 1
                    self.assertLess(abs(relative_error), 1e-3)

    def test_free_vortex_reproduces_meanline_state_at_each_mean_radius(self) -> None:
        meanline = solve_meanline()
        flow = solve_radial_equilibrium(meanline, FREE_VORTEX)

        for station, mean_speed, mean_angle, mean_pressure, mean_temperature in (
            (2, meanline.v_2, meanline.alpha_2, meanline.P_s2, meanline.T_2),
            (3, meanline.v_3, meanline.alpha_3, meanline.P_s3, meanline.T_3),
        ):
            state = flow.station_state(flow.annulus[station].mean_radius, station)
            expected_axial_velocity = mean_speed * math.cos(math.radians(mean_angle))
            expected_swirl = mean_speed * math.sin(math.radians(mean_angle))

            with self.subTest(station=station):
                self.assertAlmostEqual(state.axial_velocity, expected_axial_velocity, places=7)
                self.assertAlmostEqual(state.tangential_velocity, expected_swirl, places=7)
                self.assertAlmostEqual(state.static_pressure, mean_pressure, places=10)
                self.assertAlmostEqual(state.static_temperature, mean_temperature, places=7)

    def test_vortex_family_preserves_meanline_swirl_and_pressure_anchors(self) -> None:
        meanline = solve_meanline()

        for exponent in (FREE_VORTEX, EXPONENTIAL, FIRST_POWER):
            flow = solve_radial_equilibrium(meanline, exponent)
            for station, mean_speed, mean_angle, mean_pressure in (
                (2, meanline.v_2, meanline.alpha_2, meanline.P_s2),
                (3, meanline.v_3, meanline.alpha_3, meanline.P_s3),
            ):
                state = flow.station_state(flow.annulus[station].mean_radius, station)
                expected_swirl = mean_speed * math.sin(math.radians(mean_angle))

                with self.subTest(exponent=exponent, station=station):
                    self.assertAlmostEqual(state.tangential_velocity, expected_swirl, places=7)
                    self.assertAlmostEqual(state.static_pressure, mean_pressure, places=10)


if __name__ == "__main__":
    unittest.main()
