from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

SRC_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from Pipeline.Meanline.solve_meanline import Inputs, plot_velocity_triangles
from Helpers.stage_solver import StageDesign
from Helpers.turbine_solver import TurbineDesign, print_turbine_report, solve_turbine
from Runner.turbine_reports import RUNS_DIRECTORY, write_turbine_reports

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

CONFIG_KEYS = frozenset(("stage_count", "stages", "expansion_shares"))
STAGE_INPUT_KEYS = frozenset((
    "degree_of_reaction",
    "shaft_speed_rpm",
    "radius_2_m",
    "radius_3_m",
    "stator_exit_angle_deg",
    "rotor_exit_metal_angle_deg",
    "stator_kinetic_loss_fraction",
    "rotor_kinetic_loss_fraction",
))


def _read_number(configuration: dict[str, object], key: str, default: float) -> float:
    value = configuration.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    try:
        numeric_value = float(value)
    except OverflowError as error:
        raise ValueError(f"{key} must be a finite number, got {value!r}") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return numeric_value


def _stage_design(configuration: dict[str, object], stage_number: int) -> StageDesign:
    unknown_keys = set(configuration) - STAGE_INPUT_KEYS
    if unknown_keys:
        raise ValueError(f"unknown stage {stage_number} configuration keys: {sorted(unknown_keys)}")

    defaults = SELECTED_DESIGN.stages[0].inputs
    inputs = Inputs(
        degree_of_reaction=_read_number(configuration, "degree_of_reaction", defaults.degree_of_reaction),
        N_shaft=_read_number(configuration, "shaft_speed_rpm", defaults.N_shaft),
        radius_2=_read_number(configuration, "radius_2_m", defaults.radius_2),
        radius_3=_read_number(configuration, "radius_3_m", defaults.radius_3),
        alpha_2=_read_number(configuration, "stator_exit_angle_deg", defaults.alpha_2),
        beta_3=_read_number(configuration, "rotor_exit_metal_angle_deg", defaults.beta_3),
        loss_kinetic_stator=_read_number(
            configuration, "stator_kinetic_loss_fraction", defaults.loss_kinetic_stator
        ),
        loss_kinetic_rotor=_read_number(
            configuration, "rotor_kinetic_loss_fraction", defaults.loss_kinetic_rotor
        ),
    )
    return StageDesign(inputs=inputs)


def load_turbine_design(config_path: Path | None) -> TurbineDesign:
    if config_path is None:
        return SELECTED_DESIGN

    configuration_value: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(configuration_value, dict):
        raise ValueError("turbine configuration must be a JSON object")
    configuration: dict[str, object] = configuration_value
    unknown_keys = set(configuration) - CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"unknown turbine configuration keys: {sorted(unknown_keys)}")

    stage_count = configuration.get("stage_count")
    if isinstance(stage_count, bool) or not isinstance(stage_count, int) or stage_count not in (1, 2):
        raise ValueError(f"stage_count must be 1 or 2, got {stage_count!r}")

    stage_values = configuration.get("stages")
    if not isinstance(stage_values, list) or len(stage_values) != stage_count:
        raise ValueError(f"stage_count {stage_count} requires exactly {stage_count} stage configurations")

    stages = []
    for stage_number, stage_value in enumerate(stage_values, start=1):
        if not isinstance(stage_value, dict):
            raise ValueError(f"stage {stage_number} configuration must be a JSON object")
        stages.append(_stage_design(stage_value, stage_number))

    share_values = configuration.get("expansion_shares", [1.0 / stage_count] * stage_count)
    if not isinstance(share_values, list) or len(share_values) != stage_count:
        raise ValueError(f"stage_count {stage_count} requires exactly {stage_count} expansion shares")
    shares = tuple(
        _read_number({"share": value}, "share", 0.0)
        for value in share_values
    )
    return TurbineDesign(stages=tuple(stages), expansion_shares=shares)


def default_output_directory(stage_count: int) -> Path:
    if isinstance(stage_count, bool) or not isinstance(stage_count, int) or stage_count not in (1, 2):
        raise ValueError(f"stage_count must be 1 or 2, got {stage_count!r}")
    folder_name = "one-stage" if stage_count == 1 else "two-stage"
    return RUNS_DIRECTORY / folder_name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a full one- or two-stage turbine design.")
    parser.add_argument("--config", type=Path, help="JSON file selecting stage count and stage inputs.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Directory for this run's reports; defaults to a separate folder for each stage count.",
    )
    arguments = parser.parse_args(argv)

    design = load_turbine_design(arguments.config)
    output_directory = arguments.output_directory or default_output_directory(len(design.stages))
    solution = solve_turbine(design)
    csv_path, json_path = write_turbine_reports(solution, output_directory)
    print_turbine_report(solution)
    print(f"Run outputs: {csv_path} and {json_path}")
    if not solution.feasible:
        print("The selected design violates one or more checked design constraints")
        return 2

    stage_count = len(solution.stages)
    for stage_index, stage in enumerate(solution.stages, start=1):
        plot_velocity_triangles(
            stage.final.meanline,
            stage_label=f"Stage {stage_index} of {stage_count}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
