import numpy as np
from mpmath import mp

from src.estimators.sun_wu import (
    sun_wu_scalar_rbf_le_estimator,
)


def generate_affine_float(a, L, x0, N):
    x = np.empty(N + 1, dtype=float)
    x[0] = x0

    for t in range(N):
        x[t + 1] = L + a * (x[t] - L)

    return x


def test_sun_wu_affine_returns_valid_finite_estimate():
    x = generate_affine_float(
        a=0.95,
        L=2.0,
        x0=1.3,
        N=30,
    )

    result = sun_wu_scalar_rbf_le_estimator(x)

    assert result["valid"]
    assert np.isfinite(result["lambda_hat"])
    assert 2 <= result["selected_hidden_nodes"] <= 20
    assert result["n_pairs"] == len(x) - 1
    assert np.all(np.isfinite(result["local_derivatives"]))


def test_sun_wu_analytic_derivative_matches_finite_difference():
    x = generate_affine_float(
        a=0.8,
        L=2.0,
        x0=1.4,
        N=25,
    )

    result = sun_wu_scalar_rbf_le_estimator(x)

    assert result["valid"]

    centers = result["centers"]
    widths = result["widths"]
    weights = result["weights"]

    z = float(result["normalized_inputs"][3])
    h = 1e-6

    def fitted_model(value):
        phi = np.exp(
            -((value - centers) ** 2)
            / (2.0 * widths**2)
        )
        return weights[0] + np.dot(phi, weights[1:])

    numerical = (
        fitted_model(z + h)
        - fitted_model(z - h)
    ) / (2.0 * h)

    phi = np.exp(
        -((z - centers) ** 2)
        / (2.0 * widths**2)
    )

    analytical = np.sum(
        weights[1:]
        * phi
        * (centers - z)
        / (widths**2)
    )

    assert np.isclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )


def test_sun_wu_preserves_high_precision_late_window_resolution():
    mp.dps = 200

    a = mp.mpf("0.2")
    L = mp.mpf("2.0")
    x0 = mp.mpf("1.3578696273508724")

    trajectory = [x0]

    for _ in range(201):
        trajectory.append(
            L + a * (trajectory[-1] - L)
        )

    K = 10
    window = trajectory[200 - K : 202]

    result = sun_wu_scalar_rbf_le_estimator(window)

    assert result["valid"]
    assert result["n_pairs"] == K + 1
    assert result["scaling_mode"] == "mpmath_to_scaled_float64"
    assert np.unique(result["normalized_inputs"]).size == K + 1
