"""Fits a polynomial to each digitized Kacker & Okapuu (1982) Fig. 5 stagger curve and serializes the coefficients to JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["CSV_PATH", "JSON_PATH", "POLYNOMIAL_DEGREE", "read_digitized_curves", "serialize_polynomials"]

CSV_PATH = Path(__file__).with_name("K&O_stagger-blade-angle-relations.csv")
JSON_PATH = Path(__file__).with_name("K&O_stagger-polynomials.json")
POLYNOMIAL_DEGREE = 3
_HEADER_ROWS = 2  # beta_2 values, then the X,Y format row


def read_digitized_curves(csv_path: Path = CSV_PATH) -> dict[float, np.ndarray]:
    """Return {beta_2: (n, 2) array of (beta_1, stagger) points sorted by beta_1}, all in deg."""
    with csv_path.open() as csv_file:
        beta_2_row = csv_file.readline().split(",")
    beta_2_values = [float(value) for value in beta_2_row if value.strip()]
    points = np.genfromtxt(csv_path, delimiter=",", skip_header=_HEADER_ROWS)

    curves = {}
    for index, beta_2 in enumerate(beta_2_values):
        curve = points[:, 2 * index : 2 * index + 2]
        curve = curve[~np.isnan(curve).any(axis=1)]
        curves[beta_2] = curve[np.argsort(curve[:, 0])]
    return curves


def fit_polynomial(curve: np.ndarray, degree: int) -> list[float]:
    """Return least-squares polynomial coefficients of stagger(beta_1), highest power first."""
    beta_1 = curve[:, 0]
    stagger = curve[:, 1]
    return np.polyfit(beta_1, stagger, degree).tolist()


def serialize_polynomials(
    csv_path: Path = CSV_PATH,
    json_path: Path = JSON_PATH,
    degree: int = POLYNOMIAL_DEGREE,
) -> Path:
    """Fit every beta_2 curve in the CSV and write the coefficients to the JSON file."""
    serialized_curves = [
        {
            "beta_2": beta_2,
            "beta_1_endpoint": float(curve[-1, 0]),
            "coefficients": fit_polynomial(curve, degree),
        }
        for beta_2, curve in read_digitized_curves(csv_path).items()
    ]
    contents = {
        "source": csv_path.name,
        "units": "deg",
        "degree": degree,
        "coefficient_order": "highest power first",
        "curves": serialized_curves,
    }
    json_path.write_text(json.dumps(contents, indent=4))
    return json_path


if __name__ == "__main__":
    print(f"Wrote {serialize_polynomials()}")
