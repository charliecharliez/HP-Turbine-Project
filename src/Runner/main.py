from __future__ import annotations

import sys
from pathlib import Path

SRC_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from Pipeline.Meanline.solve_meanline import Inputs, plot_velocity_triangles
from Helpers.stage_solver import StageDesign
from Helpers.turbine_solver import TurbineDesign, print_turbine_report, solve_turbine
from Runner.turbine_reports import write_turbine_reports

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
    write_turbine_reports(solution)
    print_turbine_report(solution)
    if not solution.feasible:
        raise SystemExit("The selected design violates the project design constraints")
    stage_count = len(solution.stages)
    for stage_index, stage in enumerate(solution.stages, start=1):
        plot_velocity_triangles(
            stage.final.meanline,
            stage_label=f"Stage {stage_index} of {stage_count}",
        )
