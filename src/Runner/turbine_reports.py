from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from Helpers.design_models import JsonObject, JsonValue
from Helpers.turbine_reporting import (
    AnnulusReport,
    StageReport,
    StationReport,
    VelocityReport,
    build_stage_reports,
)
from Helpers.stage_solver import Constraint, StageSolution
from Helpers.turbine_solver import TurbineSolution
from Pipeline.Airfoil.airfoil import BladeRow

RUNS_DIRECTORY = Path(__file__).resolve().parents[2] / "runs"
CSV_FIELDS = (
    "stage_index",
    "station_index",
    "station_id",
    "location",
    "static_pressure_pa",
    "static_temperature_k",
    "hub_radius_m",
    "tip_radius_m",
    "hub_diameter_m",
    "tip_diameter_m",
    "blade_height_m",
    "annulus_area_m2",
    "blade_speed_m_s",
    "absolute_axial_velocity_m_s",
    "absolute_tangential_velocity_m_s",
    "absolute_speed_m_s",
    "absolute_flow_angle_deg",
    "absolute_mach",
    "relative_axial_velocity_m_s",
    "relative_tangential_velocity_m_s",
    "relative_speed_m_s",
    "relative_flow_angle_deg",
    "relative_mach",
)


def _annulus_snapshot(annulus: AnnulusReport) -> JsonObject:
    return {
        "hub_radius_m": annulus.hub_radius_m,
        "tip_radius_m": annulus.tip_radius_m,
        "hub_diameter_m": annulus.hub_diameter_m,
        "tip_diameter_m": annulus.tip_diameter_m,
        "blade_height_m": annulus.blade_height_m,
        "area_m2": annulus.area_m2,
    }


def _velocity_snapshot(velocity: VelocityReport) -> JsonObject:
    return {
        "axial_velocity_m_s": velocity.axial_velocity_m_s,
        "tangential_velocity_m_s": velocity.tangential_velocity_m_s,
        "speed_m_s": velocity.speed_m_s,
        "flow_angle_deg": velocity.flow_angle_deg,
        "mach": velocity.mach,
    }


def _station_snapshot(station: StationReport) -> JsonObject:
    return {
        "station_index": station.station_index,
        "station_id": station.station_id,
        "location": station.location,
        "static_pressure_pa": station.static_pressure_pa,
        "static_temperature_k": station.static_temperature_k,
        "annulus": _annulus_snapshot(station.annulus),
        "blade_speed_m_s": station.blade_speed_m_s,
        "absolute_velocity": _velocity_snapshot(station.absolute_velocity),
        "relative_velocity": _velocity_snapshot(station.relative_velocity),
    }


def _constraint_snapshot(constraint: Constraint, scope: str) -> JsonObject:
    return {
        "scope": scope,
        "name": constraint.name,
        "value": constraint.value,
        "limit": constraint.limit,
        "is_upper_limit": constraint.is_upper_limit,
        "satisfied": constraint.satisfied,
    }


def _row_snapshot(row: BladeRow) -> JsonObject:
    return {
        "blade_count": row.blade_count,
        "height_m": row.height,
        "midspan_chord_m": row.midspan.chord,
        "midspan_axial_chord_m": row.midspan.axial_chord,
    }


def _stage_comparison_snapshot(
    stage: StageSolution,
    stage_index: int,
    solution: TurbineSolution,
) -> JsonObject:
    final = stage.final
    meanline = final.meanline
    constraints: list[JsonValue] = [
        _constraint_snapshot(constraint, f"stage_{stage_index}")
        for constraint in stage.constraints
    ]
    return {
        "stage_index": stage_index,
        "shaft_speed_rpm": final.inputs.N_shaft,
        "pressure_ratio": final.inputs.pressure_ratio,
        "degree_of_reaction": final.inputs.degree_of_reaction,
        "radius_2_m": final.inputs.radius_2,
        "radius_3_m": final.inputs.radius_3,
        "stator_exit_angle_deg": final.inputs.alpha_2,
        "rotor_exit_metal_angle_deg": final.inputs.beta_3,
        "stator_kinetic_loss_fraction": final.inputs.loss_kinetic_stator,
        "rotor_kinetic_loss_fraction": final.inputs.loss_kinetic_rotor,
        "flow_coefficient": meanline.flow_coefficient,
        "loading_coefficient": meanline.stage_loading,
        "specific_work_j_kg": stage.efficiency.specific_work,
        "work_share": solution.performance.work_shares[stage_index - 1],
        "expansion_share": solution.design.expansion_shares[stage_index - 1],
        "total_to_total_efficiency": stage.efficiency.total_to_total,
        "total_to_static_efficiency": stage.efficiency.total_to_static,
        "iteration_count": len(stage.evaluations),
        "converged": stage.converged,
        "feasible": stage.feasible,
        "stator_row": _row_snapshot(final.stator),
        "rotor_row": _row_snapshot(final.rotor),
        "constraints": constraints,
    }


def _comparison_snapshot(solution: TurbineSolution) -> JsonObject:
    stages: list[JsonValue] = [
        _stage_comparison_snapshot(stage, index, solution)
        for index, stage in enumerate(solution.stages, start=1)
    ]
    constraints: list[JsonValue] = [
        _constraint_snapshot(constraint, f"stage_{index}")
        for index, stage in enumerate(solution.stages, start=1)
        for constraint in stage.constraints
    ]
    constraints.extend(
        _constraint_snapshot(constraint, "rig")
        for constraint in solution.rig_constraints
    )
    inlet = solution.design.inlet
    return {
        "stage_count": len(solution.stages),
        "converged": solution.converged,
        "feasible": solution.feasible,
        "operating_conditions": {
            "inlet_total_pressure_bar": inlet.P_0,
            "inlet_total_temperature_k": inlet.T_0,
            "mass_flow_kg_s": inlet.m_dot,
            "total_to_static_pressure_ratio": solution.design.pressure_ratio,
        },
        "performance": {
            "specific_work_j_kg": solution.performance.specific_work,
            "power_w": solution.performance.power_w,
            "total_to_total_efficiency": solution.performance.total_to_total,
            "total_to_static_efficiency": solution.performance.total_to_static,
            "exit_kinetic_energy_j_kg": solution.performance.exit_kinetic_energy_j_kg,
            "exit_swirl_kinetic_energy_j_kg": solution.performance.exit_swirl_kinetic_energy_j_kg,
        },
        "expansion_shares": list(solution.design.expansion_shares),
        "work_shares": list(solution.performance.work_shares),
        "stages": stages,
        "constraints": constraints,
        "comparison_notes": {
            "weight_and_length": (
                "Additional stages add stator and rotor rows and may affect mass and flow-path length. "
                "No validated total mass or axial-length model is applied."
            ),
            "cooling_and_metal_temperature": "Not assessed.",
            "automatic_stage_count_selection": "Not performed; compare separate run outputs.",
        },
    }


def _snapshot_for_solution(
    solution: TurbineSolution,
    stage_reports: Sequence[StageReport],
) -> JsonObject:
    comparison = _comparison_snapshot(solution)
    comparison_stages = comparison.pop("stages")
    if not isinstance(comparison_stages, list):
        raise ValueError("stage comparison records must be a JSON array")
    if len(comparison_stages) != len(stage_reports):
        raise ValueError("stage comparison and station report counts do not match")
    stages: list[JsonValue] = []
    for comparison_stage, report in zip(comparison_stages, stage_reports):
        if not isinstance(comparison_stage, dict):
            raise ValueError("stage comparison record must be a JSON object")
        stages.append({
            **comparison_stage,
            "stations": [_station_snapshot(station) for station in report.stations],
        })
    return {
        "schema_version": 2,
        "velocity_sign_convention": {
            "axial_positive": "downstream",
            "tangential_positive": "shaft_rotation",
            "absolute_frame": "C",
            "relative_frame": "W",
        },
        **comparison,
        "stages": stages,
    }


def turbine_report_snapshot(solution: TurbineSolution) -> JsonObject:
    return _snapshot_for_solution(solution, build_stage_reports(solution.stages))


def _station_csv_row(station: StationReport) -> dict[str, str | int | float]:
    annulus = station.annulus
    absolute = station.absolute_velocity
    relative = station.relative_velocity
    return {
        "stage_index": station.stage_index,
        "station_index": station.station_index,
        "station_id": station.station_id,
        "location": station.location,
        "static_pressure_pa": station.static_pressure_pa,
        "static_temperature_k": station.static_temperature_k,
        "hub_radius_m": annulus.hub_radius_m,
        "tip_radius_m": annulus.tip_radius_m,
        "hub_diameter_m": annulus.hub_diameter_m,
        "tip_diameter_m": annulus.tip_diameter_m,
        "blade_height_m": annulus.blade_height_m,
        "annulus_area_m2": annulus.area_m2,
        "blade_speed_m_s": station.blade_speed_m_s,
        "absolute_axial_velocity_m_s": absolute.axial_velocity_m_s,
        "absolute_tangential_velocity_m_s": absolute.tangential_velocity_m_s,
        "absolute_speed_m_s": absolute.speed_m_s,
        "absolute_flow_angle_deg": absolute.flow_angle_deg,
        "absolute_mach": absolute.mach,
        "relative_axial_velocity_m_s": relative.axial_velocity_m_s,
        "relative_tangential_velocity_m_s": relative.tangential_velocity_m_s,
        "relative_speed_m_s": relative.speed_m_s,
        "relative_flow_angle_deg": relative.flow_angle_deg,
        "relative_mach": relative.mach,
    }


def write_turbine_reports(
    solution: TurbineSolution,
    output_directory: Path = RUNS_DIRECTORY,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "turbine_report.csv"
    json_path = output_directory / "turbine_report.json"
    stage_reports = build_stage_reports(solution.stages)
    snapshot = _snapshot_for_solution(solution, stage_reports)
    serialized_json = json.dumps(snapshot, allow_nan=False, indent=2, sort_keys=True)
    json_path.write_text(f"{serialized_json}\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(
            _station_csv_row(station)
            for stage in stage_reports
            for station in stage.stations
        )
    return csv_path, json_path
