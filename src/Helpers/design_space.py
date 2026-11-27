from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VariableBounds:
    shaft_speed: tuple[float, float] = (8000.0, 20000.0)  # rpm
    blade_speed: tuple[float, float] = (350.0, 500.0)  # m/s
    radius_ratio: tuple[float, float] = (1.0, 1.15)  # radius_3 / radius_2
    reaction: tuple[float, float] = (0.0, 0.6)
    stator_exit_angle: tuple[float, float] = (52.0, 78.0)  # deg
    rotor_exit_angle: tuple[float, float] = (-78.0, -52.0)  # deg
    expansion_share: tuple[float, float] = (0.3, 0.7)

    def __post_init__(self) -> None:
        for name in (
            "shaft_speed",
            "blade_speed",
            "radius_ratio",
            "reaction",
            "stator_exit_angle",
            "rotor_exit_angle",
            "expansion_share",
        ):
            limits = getattr(self, name)
            if not isinstance(limits, tuple) or len(limits) != 2:
                raise ValueError(f"{name} must contain a lower and upper bound")
            lower, upper = limits
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in limits
            ):
                raise ValueError(f"{name} bounds must be finite numbers")
            if lower >= upper:
                raise ValueError(f"{name} lower bound must be less than its upper bound")

        if self.shaft_speed[0] <= 0 or self.blade_speed[0] <= 0:
            raise ValueError("speed bounds must be positive")
        if self.radius_ratio[0] < 1:
            raise ValueError("radius_ratio bounds must be at least 1")
        if self.reaction[0] < 0 or self.reaction[1] > 1:
            raise ValueError("reaction bounds must be within [0, 1]")
        if self.stator_exit_angle[0] <= -90 or self.stator_exit_angle[1] >= 90:
            raise ValueError("stator_exit_angle bounds must be within (-90, 90) degrees")
        if self.rotor_exit_angle[0] <= -90 or self.rotor_exit_angle[1] >= 90:
            raise ValueError("rotor_exit_angle bounds must be within (-90, 90) degrees")
        if self.expansion_share[0] <= 0 or self.expansion_share[1] >= 1:
            raise ValueError("expansion_share bounds must be within (0, 1)")


@dataclass(frozen=True)
class ExitFlowLimits:
    max_absolute_exit_mach: float | None = None
    max_absolute_exit_swirl_m_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_absolute_exit_mach", "max_absolute_exit_swirl_m_s"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be None or a finite nonnegative number")
