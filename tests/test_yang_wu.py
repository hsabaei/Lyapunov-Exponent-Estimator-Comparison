import numpy as np

from src.estimators.yang_wu import (
    yang_wu_scalar_le_estimator,
)


def generate_affine(a, L, x0, N):
    x = np.empty(N + 1, dtype=float)
    x[0] = x0

    for t in range(N):
        x[t + 1] = (
            L
            + a * (x[t] - L)
        )

    return x


def generate_quadratic(a, beta, L, x0, N):
    x = np.empty(N + 1, dtype=float)
    x[0] = x0

    for t in range(N):
        e = x[t] - L

        x[t + 1] = (
            L
            + a * e
            + beta * e**2
        )

    return x


def test_yang_wu_affine_positive():

    a = 0.8
    L = 2.0
    x0 = 3.0

    x = generate_affine(
        a=a,
        L=L,
        x0=x0,
        N=30,
    )

    result = (
        yang_wu_scalar_le_estimator(
            x=x,
            evolution=1,
            r_min=None,
        )
    )

    assert result["valid"]

    assert np.isclose(
        result["lambda_hat"],
        np.log(abs(a)),
        rtol=1e-9,
        atol=1e-9,
    )


def test_yang_wu_affine_negative():

    a = -0.8
    L = 2.0
    x0 = 3.0

    x = generate_affine(
        a=a,
        L=L,
        x0=x0,
        N=30,
    )

    result = (
        yang_wu_scalar_le_estimator(
            x=x,
            evolution=1,
            r_min=None,
        )
    )

    assert result["valid"]

    assert np.isclose(
        result["lambda_hat"],
        np.log(abs(a)),
        rtol=1e-9,
        atol=1e-9,
    )


def test_yang_wu_quadratic_recovers_local_derivative():

    a = 0.8
    beta = 0.1
    L = 2.0
    x0 = 2.5

    x = generate_quadratic(
        a=a,
        beta=beta,
        L=L,
        x0=x0,
        N=30,
    )

    result = (
        yang_wu_scalar_le_estimator(
            x=x,
            evolution=1,
            r_min=None,
        )
    )

    assert result["valid"]

    indices = result[
        "fiducial_indices"
    ]

    true_local_jacobians = np.array(
        [
            a
            + 2.0
            * beta
            * (x[i] - L)
            for i in indices
        ]
    )

    assert np.allclose(
        result["local_jacobians"],
        true_local_jacobians,
        rtol=1e-8,
        atol=1e-8,
    )

    expected_le = np.mean(
        np.log(
            np.abs(
                true_local_jacobians
            )
        )
    )

    assert np.isclose(
        result["lambda_hat"],
        expected_le,
        rtol=1e-8,
        atol=1e-8,
    )
