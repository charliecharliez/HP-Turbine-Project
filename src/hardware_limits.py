import math

from design_models import BoundaryConditions, JsonObject


def check_rotor_hardware_limits(
    stage_index: int,
    shaft_speed_rpm: float,
    inlet_blade_speed_m_s: float,
    exit_blade_speed_m_s: float,
    exit_annulus_area_m2: float,
    boundary: BoundaryConditions,
) -> JsonObject:
    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 1:
        raise ValueError("stage_index must be a positive integer")
    for name, value in (
        ("shaft_speed_rpm", shaft_speed_rpm),
        ("inlet_blade_speed_m_s", inlet_blade_speed_m_s),
        ("exit_blade_speed_m_s", exit_blade_speed_m_s),
        ("exit_annulus_area_m2", exit_annulus_area_m2),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive value")

    exit_an2 = exit_annulus_area_m2 * shaft_speed_rpm * shaft_speed_rpm
    if not math.isfinite(exit_an2):
        raise ValueError("rotor exit AN2 must be finite")

    checks: list[JsonObject] = [
        {
            "stage_index": stage_index,
            "constraint": "rotor_inlet_mean_blade_speed",
            "value": inlet_blade_speed_m_s,
            "limit": boundary.max_mean_blade_speed_m_s,
            "unit": "m/s",
            "passed": inlet_blade_speed_m_s <= boundary.max_mean_blade_speed_m_s,
        },
        {
            "stage_index": stage_index,
            "constraint": "rotor_exit_mean_blade_speed",
            "value": exit_blade_speed_m_s,
            "limit": boundary.max_mean_blade_speed_m_s,
            "unit": "m/s",
            "passed": exit_blade_speed_m_s <= boundary.max_mean_blade_speed_m_s,
        },
        {
            "stage_index": stage_index,
            "constraint": "rotor_exit_an2",
            "value": exit_an2,
            "limit": boundary.max_rotor_exit_an2_m2_rpm2,
            "unit": "m^2 rpm^2",
            "passed": exit_an2 <= boundary.max_rotor_exit_an2_m2_rpm2,
        },
    ]
    return {"passed": all(check["passed"] is True for check in checks), "checks": checks}
