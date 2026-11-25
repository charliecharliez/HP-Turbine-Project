from Meanline.solve_meanline import Inputs, plot_velocity_triangles
from stage_solver import StageDesign, print_stage_report, solve_stage

SELECTED_DESIGN = StageDesign(inputs=Inputs(
    degree_of_reaction=0.25,
    N_shaft=11000,  # rpm
    radius_2=0.4339,  # m
    radius_3=0.4339,  # m
    alpha_2=70,  # deg
    beta_3=-70,  # deg
    loss_kinetic_stator=0.089,
    loss_kinetic_rotor=0.112,
))

if __name__ == "__main__":
    solution = solve_stage(SELECTED_DESIGN)
    print_stage_report(solution)
    if not solution.feasible:
        raise SystemExit("The selected design violates the project design constraints")
    plot_velocity_triangles(solution.final.meanline)
