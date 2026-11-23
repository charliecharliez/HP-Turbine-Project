import sys
import unittest
from pathlib import Path

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from boundary_conditions import boundary_conditions
from design_models import (
    BoundaryConditions,
    CoolantState,
    DesignResult,
    DesignInputs,
    FlowState,
    LossComponent,
    ReferenceFrame,
    RowLosses,
    RowKind,
    RunRecord,
    RunStatus,
    RowGeometry,
    RowInputs,
    StageInputs,
    StageResult,
)


def _row(stage_index: int, row_kind: RowKind) -> RowInputs:
    geometry = RowGeometry(
        stage_index=stage_index,
        row_kind=row_kind,
        chord_m=0.05,
        axial_chord_m=0.04,
        pitch_m=0.03,
        hub_radius_m=0.2,
        tip_radius_m=0.3,
        blade_count=40,
    )
    return RowInputs(stage_index=stage_index, row_kind=row_kind, geometry=geometry)


def _stage(stage_index: int, work_fraction: float) -> StageInputs:
    return StageInputs(
        stage_index=stage_index,
        shaft_speed_rpm=13_000,
        flow_coefficient=0.6,
        loading_coefficient=1.8,
        degree_of_reaction=0.4,
        work_fraction=work_fraction,
        rows=(_row(stage_index, "stator"), _row(stage_index, "rotor")),
    )


class DesignModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boundary = boundary_conditions.as_si()

    def test_legacy_boundary_adapter_converts_pressures_to_pascals(self) -> None:
        self.assertEqual(self.boundary.inlet_total_pressure_pa, 2_000_000)
        self.assertEqual(self.boundary.coolant_total_pressure_pa, 2_100_000)
        self.assertEqual(self.boundary.max_metal_temperature_k, 1250)

    def test_design_inputs_support_one_and_two_stages(self) -> None:
        one_stage = DesignInputs(boundary_conditions=self.boundary, stages=(_stage(1, 1.0),))
        two_stage = DesignInputs(
            boundary_conditions=self.boundary,
            stages=(_stage(1, 0.55), _stage(2, 0.45)),
        )

        self.assertEqual(len(one_stage.stages), 1)
        self.assertEqual(len(two_stage.stages), 2)

    def test_design_inputs_reject_incomplete_work_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "work fractions must sum to 1"):
            DesignInputs(
                boundary_conditions=self.boundary,
                stages=(_stage(1, 0.6), _stage(2, 0.5)),
            )

    def test_design_inputs_reject_nonconsecutive_stage_indexes(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage indices must be consecutive"):
            DesignInputs(
                boundary_conditions=self.boundary,
                stages=(_stage(2, 1.0),),
            )

    def test_flow_state_angle_preserves_tangential_sign(self) -> None:
        positive = FlowState(
            stage_index=1,
            local_station=2,
            frame=ReferenceFrame.ABSOLUTE,
            total_pressure_pa=2_000_000,
            static_pressure_pa=1_800_000,
            total_temperature_k=1850,
            static_temperature_k=1700,
            mass_flow_kg_s=30,
            axial_velocity_m_s=100,
            tangential_velocity_m_s=100,
            mach_number=0.5,
        )
        negative = FlowState(
            stage_index=1,
            local_station=2,
            frame=ReferenceFrame.ABSOLUTE,
            total_pressure_pa=2_000_000,
            static_pressure_pa=1_800_000,
            total_temperature_k=1850,
            static_temperature_k=1700,
            mass_flow_kg_s=30,
            axial_velocity_m_s=100,
            tangential_velocity_m_s=-100,
            mach_number=0.5,
        )

        self.assertAlmostEqual(positive.flow_angle_deg, 45)
        self.assertAlmostEqual(negative.flow_angle_deg, -45)

    def test_flow_state_rejects_nonpositive_axial_velocity(self) -> None:
        with self.assertRaisesRegex(ValueError, "axial_velocity_m_s must be positive"):
            FlowState(
                stage_index=1,
                local_station=2,
                frame=ReferenceFrame.ABSOLUTE,
                total_pressure_pa=2_000_000,
                static_pressure_pa=1_800_000,
                total_temperature_k=1850,
                static_temperature_k=1700,
                mass_flow_kg_s=30,
                axial_velocity_m_s=0,
                tangential_velocity_m_s=0,
                mach_number=0,
            )

    def test_boundary_model_rejects_invalid_clearance_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "rotor_tip_clearance_ratio"):
            BoundaryConditions(
                inlet_total_pressure_pa=2_000_000,
                inlet_total_temperature_k=1850,
                main_mass_flow_kg_s=30,
                total_to_static_pressure_ratio=4.1,
                coolant_total_pressure_pa=2_100_000,
                coolant_total_temperature_k=750,
                rotor_tip_clearance_ratio=1.0,
                max_metal_temperature_k=1250,
                max_mean_blade_speed_m_s=500,
                max_rotor_exit_an2_m2_rpm2=3.2e7,
            )

    def test_successful_run_record_requires_a_converged_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a result and converged=True"):
            RunRecord(
                run_id="run-1",
                schema_version=1,
                input_snapshot={"mass_flow_kg_s": 30},
                assumptions=(),
                warnings=(),
                status=RunStatus.SUCCEEDED,
                converged=True,
                result=None,
            )

    def test_design_result_groups_row_flow_loss_and_coolant(self) -> None:
        flow_state = FlowState(
            stage_index=1,
            local_station=2,
            frame=ReferenceFrame.ABSOLUTE,
            total_pressure_pa=2_000_000,
            static_pressure_pa=1_800_000,
            total_temperature_k=1850,
            static_temperature_k=1700,
            mass_flow_kg_s=30,
            axial_velocity_m_s=100,
            tangential_velocity_m_s=100,
            mach_number=0.5,
        )
        row_losses = RowLosses(
            stage_index=1,
            row_kind="stator",
            total_coefficient=0.08,
            components=(LossComponent(kind="profile", coefficient=0.05),),
        )
        coolant = CoolantState(
            stage_index=1,
            row_kind="stator",
            mass_flow_kg_s=0.3,
            total_pressure_pa=2_100_000,
            total_temperature_k=750,
            metal_temperature_k=1200,
            effectiveness=0.8,
        )
        stage_result = StageResult(
            stage_index=1,
            flow_states=(flow_state,),
            row_losses=(row_losses,),
            coolant_states=(coolant,),
            specific_work_j_kg=200_000,
            total_to_static_efficiency=0.9,
        )
        design_result = DesignResult(stages=(stage_result,), specific_work_j_kg=200_000)

        self.assertEqual(len(design_result.stages[0].row_losses), 1)
        self.assertAlmostEqual(design_result.stages[0].coolant_states[0].effectiveness, 0.8)


if __name__ == "__main__":
    unittest.main()
