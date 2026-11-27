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
from Helpers.turbine_solver import TurbineSolution

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


def _stage_snapshot(stage: StageReport) -> JsonObject:
    stations: list[JsonValue] = [_station_snapshot(station) for station in stage.stations]
    return {
        "stage_index": stage.stage_index,
        "stations": stations,
    }


def _snapshot_for_stages(stage_reports: Sequence[StageReport]) -> JsonObject:
    stages: list[JsonValue] = [
        _stage_snapshot(stage) for stage in stage_reports
    ]
    return {
        "schema_version": 1,
        "velocity_sign_convention": {
            "axial_positive": "downstream",
            "tangential_positive": "shaft_rotation",
            "absolute_frame": "C",
            "relative_frame": "W",
        },
        "stages": stages,
    }


def turbine_report_snapshot(solution: TurbineSolution) -> JsonObject:
    return _snapshot_for_stages(build_stage_reports(solution.stages))


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
    snapshot = _snapshot_for_stages(stage_reports)
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
