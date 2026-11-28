from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Literal, TypeAlias

RowKind: TypeAlias = Literal["stator", "rotor"]
LossKind: TypeAlias = Literal[
    "profile",
    "profile_reynolds_correction",
    "profile_compressibility_correction",
    "transonic_shock",
    "secondary",
    "trailing_edge",
    "tip_clearance",
    "coolant_mixing",
]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonScalar
    | list["JsonValue"]
    | tuple["JsonValue", ...]
    | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]
BAR_TO_PA = 100_000.0


class ReferenceFrame(str, Enum):
    ABSOLUTE = "absolute"
    ROTOR_RELATIVE = "rotor_relative"


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    NON_CONVERGED = "non_converged"
    FAILED = "failed"


def _validate_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _validate_positive(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_nonnegative(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")


def _validate_efficiency(value: float, name: str) -> None:
    _validate_finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _validate_index(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class BoundaryConditions:
    inlet_total_pressure_pa: float
    inlet_total_temperature_k: float
    main_mass_flow_kg_s: float
    total_to_static_pressure_ratio: float
    coolant_total_pressure_pa: float
    coolant_total_temperature_k: float
    rotor_tip_clearance_ratio: float
    max_metal_temperature_k: float
    max_mean_blade_speed_m_s: float
    max_rotor_exit_an2_m2_rpm2: float

    def __post_init__(self) -> None:
        for name in (
            "inlet_total_pressure_pa",
            "inlet_total_temperature_k",
            "main_mass_flow_kg_s",
            "total_to_static_pressure_ratio",
            "coolant_total_pressure_pa",
            "coolant_total_temperature_k",
            "max_metal_temperature_k",
            "max_mean_blade_speed_m_s",
            "max_rotor_exit_an2_m2_rpm2",
        ):
            _validate_positive(getattr(self, name), name)
        _validate_finite(self.rotor_tip_clearance_ratio, "rotor_tip_clearance_ratio")
        if not 0 <= self.rotor_tip_clearance_ratio < 1:
            raise ValueError(
                "rotor_tip_clearance_ratio must be in [0, 1), "
                f"got {self.rotor_tip_clearance_ratio}"
            )
        if self.total_to_static_pressure_ratio <= 1:
            raise ValueError(
                "total_to_static_pressure_ratio must exceed 1, "
                f"got {self.total_to_static_pressure_ratio}"
            )

    @classmethod
    def from_bar(
        cls,
        inlet_total_pressure_bar: float,
        inlet_total_temperature_k: float,
        main_mass_flow_kg_s: float,
        total_to_static_pressure_ratio: float,
        coolant_total_pressure_bar: float,
        coolant_total_temperature_k: float,
        rotor_tip_clearance_ratio: float,
        max_metal_temperature_k: float,
        max_mean_blade_speed_m_s: float,
        max_rotor_exit_an2_m2_rpm2: float,
    ) -> BoundaryConditions:
        _validate_positive(inlet_total_pressure_bar, "inlet_total_pressure_bar")
        _validate_positive(coolant_total_pressure_bar, "coolant_total_pressure_bar")
        return cls(
            inlet_total_pressure_pa=inlet_total_pressure_bar * BAR_TO_PA,
            inlet_total_temperature_k=inlet_total_temperature_k,
            main_mass_flow_kg_s=main_mass_flow_kg_s,
            total_to_static_pressure_ratio=total_to_static_pressure_ratio,
            coolant_total_pressure_pa=coolant_total_pressure_bar * BAR_TO_PA,
            coolant_total_temperature_k=coolant_total_temperature_k,
            rotor_tip_clearance_ratio=rotor_tip_clearance_ratio,
            max_metal_temperature_k=max_metal_temperature_k,
            max_mean_blade_speed_m_s=max_mean_blade_speed_m_s,
            max_rotor_exit_an2_m2_rpm2=max_rotor_exit_an2_m2_rpm2,
        )


@dataclass(frozen=True)
class RowGeometry:
    stage_index: int
    row_kind: RowKind
    chord_m: float
    axial_chord_m: float
    pitch_m: float
    hub_radius_m: float
    tip_radius_m: float
    blade_count: int
    trailing_edge_thickness_m: float = 0.0
    tip_clearance_m: float = 0.0
    max_thickness_chord_ratio: float | None = None

    def __post_init__(self) -> None:
        _validate_index(self.stage_index, "stage_index")
        if self.row_kind not in ("stator", "rotor"):
            raise ValueError(f"row_kind must be 'stator' or 'rotor', got {self.row_kind!r}")
        for name in ("chord_m", "axial_chord_m", "pitch_m", "hub_radius_m", "tip_radius_m"):
            _validate_positive(getattr(self, name), name)
        if self.tip_radius_m <= self.hub_radius_m:
            raise ValueError("tip_radius_m must exceed hub_radius_m")
        if (
            isinstance(self.blade_count, bool)
            or not isinstance(self.blade_count, int)
            or self.blade_count < 1
        ):
            raise ValueError(f"blade_count must be a positive integer, got {self.blade_count!r}")
        _validate_nonnegative(self.trailing_edge_thickness_m, "trailing_edge_thickness_m")
        _validate_nonnegative(self.tip_clearance_m, "tip_clearance_m")
        if self.max_thickness_chord_ratio is not None:
            _validate_finite(self.max_thickness_chord_ratio, "max_thickness_chord_ratio")
            if not 0 < self.max_thickness_chord_ratio <= 1:
                raise ValueError(
                    "max_thickness_chord_ratio must be in (0, 1], "
                    f"got {self.max_thickness_chord_ratio}"
                )


@dataclass(frozen=True)
class RowInputs:
    stage_index: int
    row_kind: RowKind
    geometry: RowGeometry
    coolant_mass_flow_kg_s: float = 0.0

    def __post_init__(self) -> None:
        _validate_index(self.stage_index, "stage_index")
        if self.row_kind not in ("stator", "rotor"):
            raise ValueError(f"row_kind must be 'stator' or 'rotor', got {self.row_kind!r}")
        if self.geometry.stage_index != self.stage_index or self.geometry.row_kind != self.row_kind:
            raise ValueError("row input and row geometry must identify the same stage and row")
        _validate_nonnegative(self.coolant_mass_flow_kg_s, "coolant_mass_flow_kg_s")


@dataclass(frozen=True)
class StageInputs:
    stage_index: int
    shaft_speed_rpm: float
    flow_coefficient: float
    loading_coefficient: float
    degree_of_reaction: float
    work_fraction: float
    rows: tuple[RowInputs, ...]

    def __post_init__(self) -> None:
        _validate_index(self.stage_index, "stage_index")
        for name in ("shaft_speed_rpm", "flow_coefficient", "loading_coefficient"):
            _validate_positive(getattr(self, name), name)
        _validate_efficiency(self.degree_of_reaction, "degree_of_reaction")
        _validate_finite(self.work_fraction, "work_fraction")
        if not 0 < self.work_fraction <= 1:
            raise ValueError(f"work_fraction must be in (0, 1], got {self.work_fraction}")
        if tuple(row.row_kind for row in self.rows) != ("stator", "rotor"):
            raise ValueError("rows must contain a stator followed by a rotor")
        if any(row.stage_index != self.stage_index for row in self.rows):
            raise ValueError("all rows must belong to the stage")


@dataclass(frozen=True)
class DesignInputs:
    boundary_conditions: BoundaryConditions
    stages: tuple[StageInputs, ...]

    def __post_init__(self) -> None:
        if len(self.stages) not in (1, 2):
            raise ValueError(f"design must contain one or two stages, got {len(self.stages)}")
        expected_indices = tuple(range(1, len(self.stages) + 1))
        actual_indices = tuple(stage.stage_index for stage in self.stages)
        if actual_indices != expected_indices:
            raise ValueError(f"stage indices must be consecutive from 1, got {actual_indices}")
        if not math.isclose(
            sum(stage.work_fraction for stage in self.stages),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("stage work fractions must sum to 1")


@dataclass(frozen=True)
class FlowState:
    stage_index: int
    local_station: int
    frame: ReferenceFrame
    total_pressure_pa: float
    static_pressure_pa: float
    total_temperature_k: float
    static_temperature_k: float
    mass_flow_kg_s: float
    axial_velocity_m_s: float
    tangential_velocity_m_s: float
    mach_number: float

    def __post_init__(self) -> None:
        _validate_index(self.stage_index, "stage_index")
        _validate_index(self.local_station, "local_station")
        if self.local_station not in (1, 2, 3):
            raise ValueError(f"local_station must be 1, 2, or 3, got {self.local_station}")
        if not isinstance(self.frame, ReferenceFrame):
            raise ValueError(f"frame must be a ReferenceFrame, got {self.frame!r}")
        for name in (
            "total_pressure_pa",
            "static_pressure_pa",
            "total_temperature_k",
            "static_temperature_k",
            "mass_flow_kg_s",
            "axial_velocity_m_s",
        ):
            _validate_positive(getattr(self, name), name)
        _validate_finite(self.tangential_velocity_m_s, "tangential_velocity_m_s")
        _validate_nonnegative(self.mach_number, "mach_number")

    @property
    def flow_angle_deg(self) -> float:
        return math.degrees(math.atan2(self.tangential_velocity_m_s, self.axial_velocity_m_s))


@dataclass(frozen=True)
class LossComponent:
    kind: LossKind
    coefficient: float

    def __post_init__(self) -> None:
        if self.kind not in (
            "profile",
            "profile_reynolds_correction",
            "profile_compressibility_correction",
            "transonic_shock",
            "secondary",
            "trailing_edge",
            "tip_clearance",
            "coolant_mixing",
        ):
            raise ValueError(f"unsupported loss kind: {self.kind!r}")
        if self.kind in ("profile_reynolds_correction", "profile_compressibility_correction"):
            _validate_finite(self.coefficient, "coefficient")
        else:
            _validate_nonnegative(self.coefficient, "coefficient")


@dataclass(frozen=True)
class RowLosses:
    stage_index: int
    row_kind: RowKind
    total_coefficient: float
    components: tuple[LossComponent, ...]

    def __post_init__(self) -> None:
        _validate_index(self.stage_index, "stage_index")
        if self.row_kind not in ("stator", "rotor"):
            raise ValueError(f"row_kind must be 'stator' or 'rotor', got {self.row_kind!r}")
        _validate_nonnegative(self.total_coefficient, "total_coefficient")
        kinds = tuple(component.kind for component in self.components)
        if len(kinds) != len(set(kinds)):
            raise ValueError("loss component kinds must be unique within a row")


@dataclass(frozen=True)
class CoolantState:
    stage_index: int
    row_kind: RowKind
    mass_flow_kg_s: float
    total_pressure_pa: float
    total_temperature_k: float
    metal_temperature_k: float | None = None
    effectiveness: float | None = None

    def __post_init__(self) -> None:
        _validate_index(self.stage_index, "stage_index")
        if self.row_kind not in ("stator", "rotor"):
            raise ValueError(f"row_kind must be 'stator' or 'rotor', got {self.row_kind!r}")
        _validate_nonnegative(self.mass_flow_kg_s, "mass_flow_kg_s")
        _validate_positive(self.total_pressure_pa, "total_pressure_pa")
        _validate_positive(self.total_temperature_k, "total_temperature_k")
        if self.metal_temperature_k is not None:
            _validate_positive(self.metal_temperature_k, "metal_temperature_k")
        if self.effectiveness is not None:
            _validate_efficiency(self.effectiveness, "effectiveness")


@dataclass(frozen=True)
class StageResult:
    stage_index: int
    flow_states: tuple[FlowState, ...]
    row_losses: tuple[RowLosses, ...]
    coolant_states: tuple[CoolantState, ...]
    specific_work_j_kg: float
    total_to_total_efficiency: float | None = None
    total_to_static_efficiency: float | None = None

    def __post_init__(self) -> None:
        _validate_index(self.stage_index, "stage_index")
        _validate_finite(self.specific_work_j_kg, "specific_work_j_kg")
        for name in ("total_to_total_efficiency", "total_to_static_efficiency"):
            value = getattr(self, name)
            if value is not None:
                _validate_efficiency(value, name)
        if any(state.stage_index != self.stage_index for state in self.flow_states):
            raise ValueError("all flow states must belong to the stage")
        if any(loss.stage_index != self.stage_index for loss in self.row_losses):
            raise ValueError("all row losses must belong to the stage")
        if any(state.stage_index != self.stage_index for state in self.coolant_states):
            raise ValueError("all coolant states must belong to the stage")


@dataclass(frozen=True)
class DesignResult:
    stages: tuple[StageResult, ...]
    specific_work_j_kg: float
    total_to_total_efficiency: float | None = None
    total_to_static_efficiency: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.stages) not in (1, 2):
            raise ValueError(f"result must contain one or two stages, got {len(self.stages)}")
        expected_indices = tuple(range(1, len(self.stages) + 1))
        actual_indices = tuple(stage.stage_index for stage in self.stages)
        if actual_indices != expected_indices:
            raise ValueError(f"stage result indices must be consecutive from 1, got {actual_indices}")
        _validate_finite(self.specific_work_j_kg, "specific_work_j_kg")
        for name in ("total_to_total_efficiency", "total_to_static_efficiency"):
            value = getattr(self, name)
            if value is not None:
                _validate_efficiency(value, name)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    schema_version: int
    input_snapshot: JsonObject
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    status: RunStatus
    converged: bool | None
    result: JsonObject | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        _validate_index(self.schema_version, "schema_version")
        if self.status is RunStatus.SUCCEEDED:
            if self.converged is not True or self.error is not None or self.result is None:
                raise ValueError(
                    "a succeeded run requires a result and converged=True, without an error"
                )
        elif self.status is RunStatus.NON_CONVERGED:
            if self.converged is not False or self.error is not None or self.result is None:
                raise ValueError(
                    "a non-converged run requires a result and converged=False, without an error"
                )
        elif self.status is RunStatus.FAILED:
            if not self.error or self.result is not None:
                raise ValueError("a failed run requires an error and must not report a result")
        else:
            raise ValueError(f"unsupported run status: {self.status!r}")
