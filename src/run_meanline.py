from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from boundary_conditions import boundary_conditions
from design_models import BAR_TO_PA, JsonObject, RunRecord, RunStatus
from Meanline.solve_meanline import Inputs, MeanlineResult, solve_meanline
from run_records import write_run_record

MASS_FLOW_RESIDUAL_TOLERANCE_KG_S = 1e-6
CONFIG_DEFAULTS = Inputs()
CONFIG_KEYS = frozenset(
    (
        "degree_of_reaction",
        "shaft_speed_rpm",
        "radius_2_m",
        "radius_3_m",
        "stator_exit_angle_deg",
        "rotor_exit_metal_angle_deg",
        "stator_kinetic_loss_fraction",
        "rotor_kinetic_loss_fraction",
    )
)


def _read_number(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return float(value)


def _validate_inputs(inputs: Inputs) -> None:
    if not 0 < inputs.degree_of_reaction < 1:
        raise ValueError("degree_of_reaction must be between 0 and 1")
    for name in ("N_shaft", "radius_2", "radius_3"):
        value = getattr(inputs, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive value")
    for name in ("alpha_2", "Beta3_metal"):
        value = getattr(inputs, name)
        if not math.isfinite(value) or not -90 < value < 90:
            raise ValueError(f"{name} must be strictly between -90 and 90 degrees")
    for name in ("loss_kinetic_stator", "loss_kinetic_rotor"):
        value = getattr(inputs, name)
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")


def load_meanline_inputs(config_path: Path | None) -> Inputs:
    if config_path is None:
        inputs = Inputs()
        _validate_inputs(inputs)
        return inputs

    configuration_value: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(configuration_value, dict):
        raise ValueError("meanline configuration must be a JSON object")
    configuration: dict[str, object] = configuration_value
    unknown_keys = set(configuration) - CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"unknown meanline configuration keys: {sorted(unknown_keys)}")

    inputs = Inputs(
        degree_of_reaction=_read_number(
            configuration, "degree_of_reaction", CONFIG_DEFAULTS.degree_of_reaction
        ),
        N_shaft=_read_number(configuration, "shaft_speed_rpm", CONFIG_DEFAULTS.N_shaft),
        radius_2=_read_number(configuration, "radius_2_m", CONFIG_DEFAULTS.radius_2),
        radius_3=_read_number(configuration, "radius_3_m", CONFIG_DEFAULTS.radius_3),
        alpha_2=_read_number(
            configuration, "stator_exit_angle_deg", CONFIG_DEFAULTS.alpha_2
        ),
        Beta3_metal=_read_number(
            configuration, "rotor_exit_metal_angle_deg", CONFIG_DEFAULTS.Beta3_metal
        ),
        loss_kinetic_stator=_read_number(
            configuration,
            "stator_kinetic_loss_fraction",
            CONFIG_DEFAULTS.loss_kinetic_stator,
        ),
        loss_kinetic_rotor=_read_number(
            configuration,
            "rotor_kinetic_loss_fraction",
            CONFIG_DEFAULTS.loss_kinetic_rotor,
        ),
    )
    _validate_inputs(inputs)
    return inputs


def _input_snapshot(inputs: Inputs) -> JsonObject:
    boundary = boundary_conditions.as_si()
    boundary_snapshot: JsonObject = {
        "inlet_total_pressure_pa": boundary.inlet_total_pressure_pa,
        "inlet_total_temperature_k": boundary.inlet_total_temperature_k,
        "main_mass_flow_kg_s": boundary.main_mass_flow_kg_s,
        "total_to_static_pressure_ratio": boundary.total_to_static_pressure_ratio,
        "coolant_total_pressure_pa": boundary.coolant_total_pressure_pa,
        "coolant_total_temperature_k": boundary.coolant_total_temperature_k,
        "rotor_tip_clearance_ratio": boundary.rotor_tip_clearance_ratio,
        "max_metal_temperature_k": boundary.max_metal_temperature_k,
        "max_mean_blade_speed_m_s": boundary.max_mean_blade_speed_m_s,
        "max_rotor_exit_an2_m2_rpm2": boundary.max_rotor_exit_an2_m2_rpm2,
    }
    meanline_snapshot: JsonObject = {
        "degree_of_reaction": inputs.degree_of_reaction,
        "shaft_speed_rpm": inputs.N_shaft,
        "radius_2_m": inputs.radius_2,
        "radius_3_m": inputs.radius_3,
        "stator_exit_angle_deg": inputs.alpha_2,
        "rotor_exit_metal_angle_deg": inputs.Beta3_metal,
        "stator_kinetic_loss_fraction": inputs.loss_kinetic_stator,
        "rotor_kinetic_loss_fraction": inputs.loss_kinetic_rotor,
    }
    return {
        "boundary_conditions": boundary_snapshot,
        "meanline_inputs": meanline_snapshot,
    }


def _result_snapshot(result: MeanlineResult) -> JsonObject:
    return {
        "mass_flow_residual_kg_s": result.residual,
        "mass_flow_residual_tolerance_kg_s": MASS_FLOW_RESIDUAL_TOLERANCE_KG_S,
        "mass_flow_kg_s": result.m_dot,
        "stator_exit_static_pressure_pa": result.P_s2 * BAR_TO_PA,
        "stator_exit_static_temperature_k": result.T_2,
        "rotor_exit_static_temperature_k": result.T_3,
        "rotor_exit_total_temperature_k": result.T_03,
        "stator_exit_absolute_speed_m_s": result.v_2,
        "rotor_inlet_relative_speed_m_s": result.w_2,
        "rotor_exit_relative_speed_m_s": result.w_3,
        "rotor_exit_absolute_speed_m_s": result.v_3,
        "stator_exit_blade_speed_m_s": result.u_2,
        "rotor_exit_blade_speed_m_s": result.u_3,
        "rotor_inlet_relative_mach": result.M_2rel,
        "stator_exit_absolute_mach": result.M_2,
        "rotor_inlet_relative_flow_angle_deg_unsigned": result.Beta2,
        "rotor_exit_absolute_flow_angle_deg": result.alpha_3,
        "stator_exit_blade_height_m": result.blade_height_2,
        "rotor_exit_blade_height_m": result.blade_height_3,
        "stator_exit_hub_diameter_m": result.D_hub_2,
        "stator_exit_tip_diameter_m": result.D_tip_2,
        "rotor_exit_hub_diameter_m": result.D_hub_3,
        "rotor_exit_tip_diameter_m": result.D_tip_3,
        "rotor_exit_annulus_area_m2": result.area_3,
        "euler_specific_work_j_kg": result.work_euler,
        "thermodynamic_specific_work_j_kg": result.work_thermo,
        "total_to_static_efficiency": result.eta_ts,
    }


def _record(
    inputs: Inputs,
    status: RunStatus,
    converged: bool | None,
    result: MeanlineResult | None,
    error: str | None = None,
) -> RunRecord:
    warnings = (
        "This is the single-stage meanline prototype; integrated cooling and all hardware-limit "
        "checks are not yet part of this run.",
    )
    return RunRecord(
        run_id=str(uuid4()),
        schema_version=1,
        input_snapshot=_input_snapshot(inputs),
        assumptions=(
            "The meanline solver uses its current constant-property gas model and configured "
            "kinetic-loss inputs.",
            "The model execution status does not certify design feasibility.",
        ),
        warnings=warnings,
        status=status,
        converged=converged,
        result=_result_snapshot(result) if result is not None else None,
        error=error,
    )


def run_meanline(
    config_path: Path | None,
    output_path: Path,
) -> RunRecord:
    inputs = load_meanline_inputs(config_path)
    try:
        result = solve_meanline(inputs)
    except ValueError as error:
        record = _record(inputs, RunStatus.FAILED, None, None, str(error))
        write_run_record(output_path, record)
        raise

    converged = abs(result.residual) <= MASS_FLOW_RESIDUAL_TOLERANCE_KG_S
    status = RunStatus.SUCCEEDED if converged else RunStatus.NON_CONVERGED
    record = _record(inputs, status, converged, result)
    write_run_record(output_path, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the current single-stage meanline model.")
    parser.add_argument("--config", type=Path, help="Optional JSON file with unit-labelled meanline inputs.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the JSON run record.")
    arguments = parser.parse_args(argv)

    record = run_meanline(arguments.config, arguments.output)
    print(f"{record.status.value}: run record written to {arguments.output}")
    return 0 if record.status is RunStatus.SUCCEEDED else 2


if __name__ == "__main__":
    raise SystemExit(main())
