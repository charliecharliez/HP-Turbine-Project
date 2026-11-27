from pathlib import Path

import matplotlib.pyplot as plt

from Meanline.solve_meanline import Inputs
from RadialEquilibrium.radial_equilibrium_solver import plot_spanwise_velocity_triangles
from stage_solver import StageDesign
from turbine_solver import TurbineDesign, print_turbine_report, solve_turbine

FIGURE_DIRECTORY = Path(__file__).resolve().parents[1] / "figures"
FIGURE_RESOLUTION = 200  # dpi

SELECTED_DESIGN = TurbineDesign(stages=(
    StageDesign(inputs=Inputs(
        degree_of_reaction=0.27,
        N_shaft=12790,  # rpm
        radius_2=0.3732,  # m
        radius_3=0.3733,  # m
        alpha_2=70.0,  # deg
        beta_3=-70.0,  # deg
        loss_kinetic_stator=0.08,
        loss_kinetic_rotor=0.10,
    )),
))

if __name__ == "__main__":
    solution = solve_turbine(SELECTED_DESIGN)
    print_turbine_report(solution)
    if not solution.feasible:
        raise SystemExit("The selected design violates the project design constraints")

    FIGURE_DIRECTORY.mkdir(exist_ok=True)
    stage_count = len(solution.stages)
    for number, stage in enumerate(solution.stages, start=1):
        title = f"Stage {number} of {stage_count}: velocity triangles at hub, midspan and tip"
        figure = plot_spanwise_velocity_triangles(stage.final.spanwise_flow, title)
        figure.savefig(FIGURE_DIRECTORY / f"stage_{number}_of_{stage_count}_velocity_triangles.png",
                       dpi=FIGURE_RESOLUTION)
    plt.show()
