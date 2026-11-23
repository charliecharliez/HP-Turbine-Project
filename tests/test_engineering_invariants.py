import math
import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from RadialEquilibrium.radial_equilibrium_solver import (
    Annulus,
    VortexLaw,
    axial_velocity_constant,
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


if __name__ == "__main__":
    unittest.main()
