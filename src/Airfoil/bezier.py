from __future__ import annotations

import math

import numpy as np

__all__ = ["bezier_curve"]


def bezier_curve(control_points: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Evaluate a Bezier curve of any degree at n_points evenly spaced parameter values."""
    control_points = np.asarray(control_points, dtype=float)
    degree = len(control_points) - 1
    t = np.linspace(0.0, 1.0, n_points)[:, np.newaxis]
    index = np.arange(degree + 1)
    binomials = np.array([math.comb(degree, i) for i in index])
    bernstein_basis = binomials * t**index * (1.0 - t) ** (degree - index)
    return bernstein_basis @ control_points
