from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from Helpers.design_models import BAR_TO_PA
from Helpers.stage_solver import StageSolution


@dataclass(frozen=True)
class AnnulusReport:
    hub_radius_m: float
    tip_radius_m: float
    hub_diameter_m: float
    tip_diameter_m: float
    blade_height_m: float
    area_m2: float


@dataclass(frozen=True)
class VelocityReport:
    axial_velocity_m_s: float
    tangential_velocity_m_s: float
    speed_m_s: float
    flow_angle_deg: float
    mach: float


@dataclass(frozen=True)
class StationReport:
    stage_index: int
    station_index: int
    location: str
    static_pressure_pa: float
    static_temperature_k: float
    annulus: AnnulusReport
    blade_speed_m_s: float
    absolute_velocity: VelocityReport
    relative_velocity: VelocityReport

    @property
    def station_id(self) -> str:
        return f"({self.stage_index}, {self.station_index})"


@dataclass(frozen=True)
class StageReport:
    stage_index: int
    stations: tuple[StationReport, ...]


def _annulus_report(
    hub_diameter_m: float,
    tip_diameter_m: float,
    blade_height_m: float,
    area_m2: float,
) -> AnnulusReport:
    return AnnulusReport(
        hub_radius_m=hub_diameter_m / 2,
        tip_radius_m=tip_diameter_m / 2,
        hub_diameter_m=hub_diameter_m,
        tip_diameter_m=tip_diameter_m,
        blade_height_m=blade_height_m,
        area_m2=area_m2,
    )


def _velocity_report(speed_m_s: float, angle_deg: float, mach: float) -> VelocityReport:
    angle_rad = math.radians(angle_deg)
    return VelocityReport(
        axial_velocity_m_s=speed_m_s * math.cos(angle_rad),
        tangential_velocity_m_s=speed_m_s * math.sin(angle_rad),
        speed_m_s=speed_m_s,
        flow_angle_deg=angle_deg,
        mach=mach,
    )


def build_stage_reports(stages: Sequence[StageSolution]) -> tuple[StageReport, ...]:
    reports = []
    for stage_index, stage in enumerate(stages, start=1):
        meanline = stage.final.meanline
        inputs = stage.final.inputs
        station_2_area = 2 * math.pi * inputs.radius_2 * meanline.blade_height_2
        station_2 = StationReport(
            stage_index=stage_index,
            station_index=2,
            location="stator_exit_rotor_inlet",
            static_pressure_pa=meanline.P_s2 * BAR_TO_PA,
            static_temperature_k=meanline.T_2,
            annulus=_annulus_report(
                meanline.D_hub_2,
                meanline.D_tip_2,
                meanline.blade_height_2,
                station_2_area,
            ),
            blade_speed_m_s=meanline.u_2,
            absolute_velocity=_velocity_report(
                meanline.v_2,
                meanline.alpha_2,
                meanline.M_2,
            ),
            relative_velocity=_velocity_report(
                meanline.w_2,
                meanline.beta_2,
                meanline.M_2rel,
            ),
        )
        station_3 = StationReport(
            stage_index=stage_index,
            station_index=3,
            location="rotor_exit",
            static_pressure_pa=meanline.P_s3 * BAR_TO_PA,
            static_temperature_k=meanline.T_3,
            annulus=_annulus_report(
                meanline.D_hub_3,
                meanline.D_tip_3,
                meanline.blade_height_3,
                meanline.area_3,
            ),
            blade_speed_m_s=meanline.u_3,
            absolute_velocity=_velocity_report(
                meanline.v_3,
                meanline.alpha_3,
                meanline.M_3,
            ),
            relative_velocity=_velocity_report(
                meanline.w_3,
                meanline.beta_3,
                meanline.M_3rel,
            ),
        )
        reports.append(StageReport(stage_index, (station_2, station_3)))
    return tuple(reports)


def print_stage_station_report(report: StageReport) -> None:
    print("\nMeanline annulus geometry")
    print(
        f"{'Station':<12}{'Location':<28}{'Hub radius [m]':>16}{'Tip radius [m]':>16}"
        f"{'Hub diameter [m]':>18}{'Tip diameter [m]':>18}{'Blade height [m]':>18}{'Area [m^2]':>14}"
    )
    for station in report.stations:
        annulus = station.annulus
        print(
            f"{station.station_id:<12}{station.location:<28}{annulus.hub_radius_m:>16.5f}"
            f"{annulus.tip_radius_m:>16.5f}{annulus.hub_diameter_m:>18.5f}"
            f"{annulus.tip_diameter_m:>18.5f}{annulus.blade_height_m:>18.5f}{annulus.area_m2:>14.6f}"
        )

    print("\nMeanline velocity triangles (C absolute; W rotor-relative)")
    print(
        f"{'Station':<12}{'Location':<28}{'U [m/s]':>10}{'C_x [m/s]':>12}"
        f"{'C_theta [m/s]':>15}{'C [m/s]':>11}{'alpha [deg]':>13}"
        f"{'W_x [m/s]':>12}{'W_theta [m/s]':>15}{'W [m/s]':>11}"
        f"{'beta [deg]':>12}{'M_C abs':>10}{'M_W rel':>10}"
    )
    for station in report.stations:
        absolute = station.absolute_velocity
        relative = station.relative_velocity
        print(
            f"{station.station_id:<12}{station.location:<28}{station.blade_speed_m_s:>10.2f}"
            f"{absolute.axial_velocity_m_s:>12.2f}{absolute.tangential_velocity_m_s:>15.2f}"
            f"{absolute.speed_m_s:>11.2f}{absolute.flow_angle_deg:>13.2f}"
            f"{relative.axial_velocity_m_s:>12.2f}{relative.tangential_velocity_m_s:>15.2f}"
            f"{relative.speed_m_s:>11.2f}{relative.flow_angle_deg:>12.2f}"
            f"{absolute.mach:>10.3f}{relative.mach:>10.3f}"
        )
