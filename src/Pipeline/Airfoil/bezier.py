from __future__ import annotations

import math

import numpy as np

__all__ = ["bezier_curve", "bezier_derivative", "bezier_points"]


def bezier_curve(control_points: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Evaluate a Bezier curve of any degree at n_points evenly spaced parameter values."""
    return bezier_points(control_points, np.linspace(0.0, 1.0, n_points))


def bezier_points(control_points: np.ndarray, parameters: np.ndarray | float) -> np.ndarray:
    """Evaluate a Bezier curve of any degree at the given parameter values in [0, 1]."""
    control_points = np.asarray(control_points, dtype=float)
    degree = len(control_points) - 1
    t = np.atleast_1d(np.asarray(parameters, dtype=float))[:, np.newaxis]
    index = np.arange(degree + 1)
    binomials = np.array([math.comb(degree, i) for i in index])
    bernstein_basis = binomials * t**index * (1.0 - t) ** (degree - index)
    return bernstein_basis @ control_points


def bezier_derivative(control_points: np.ndarray, parameters: np.ndarray | float) -> np.ndarray:
    """First derivative of a Bezier curve with respect to its parameter, at the given values."""
    control_points = np.asarray(control_points, dtype=float)
    degree = len(control_points) - 1
    return degree * bezier_points(np.diff(control_points, axis=0), parameters)
