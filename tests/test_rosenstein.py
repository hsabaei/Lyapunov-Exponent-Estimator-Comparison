import numpy as np
from mpmath import mp

from src.estimators.rosenstein import (
    rosenstein_le_estimator,
)


def affine_float(a, n):
    x = np.empty(n, dtype=float)
    x[0] = 1.0

    for t in range(n - 1):
        x[t + 1] = a * x[t]

    return x


def affine_mp(a, L, x0, n):
    a = mp.mpf(a)
    L = mp.mpf(L)
    x0 = mp.mpf(x0)

    x = [x0]

    for _ in range(n - 1):
        x.append(
            L
            + a * (x[-1] - L)
        )

    return x


def test_rosenstein_affine_returns_negative_slope():
    x = affine_float(
        a=0.8,
        n=30,
    )

    result = rosenstein_le_estimator(
        x=x,
        embedding_dim=1,
        lag=1,
        temporal_exclusion=0,
        sample_time=1.0,
        max_horizon=None,
        numeric_backend="float64",
    )

    assert result["valid"]
    assert np.isfinite(
        result["lambda_hat"]
    )
    assert result["lambda_hat"] < 0.0


def test_rosenstein_positive_affine_curve_has_half_rate_after_initial_horizon():
    """
    For x_t=a^t with 0<a<1 and no temporal exclusion, the nearest
    neighbor of each nonterminal point is the next point.

    Because Rosenstein averages log distances over a horizon-dependent
    surviving pair population, for h>=1 the mean-log-distance curve has
    slope exactly (1/2)*log(a). This test validates the implemented
    Rosenstein averaging mechanism; it is not a claim that the method
    should equal the true LE in this nonstationary fixed-point setting.
    """

    a = 0.8

    x = affine_float(
        a=a,
        n=30,
    )

    result = rosenstein_le_estimator(
        x=x,
        embedding_dim=1,
        temporal_exclusion=0,
        numeric_backend="float64",
    )

    assert result["valid"]

    h = np.asarray(
        result["horizons"],
        dtype=float,
    )

    y = np.asarray(
        result["mean_log_distance"],
        dtype=float,
    )

    # h>=1 removes the special terminal-pair duplication at h=0.
    mask = h >= 1.0

    slope = np.polyfit(
        h[mask],
        y[mask],
        1,
    )[0]

    expected = (
        0.5
        * np.log(a)
    )

    assert np.isclose(
        slope,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_rosenstein_negative_affine_returns_negative_slope():
    x = affine_float(
        a=-0.8,
        n=30,
    )

    result = rosenstein_le_estimator(
        x=x,
        embedding_dim=1,
        temporal_exclusion=0,
        numeric_backend="float64",
    )

    assert result["valid"]
    assert np.isfinite(
        result["lambda_hat"]
    )
    assert result["lambda_hat"] < 0.0


def test_rosenstein_mpmath_preserves_tiny_fixed_point_offsets():
    mp.dps = 100

    x = affine_mp(
        a="0.2",
        L="2.0",
        x0="1.3",
        n=40,
    )

    result = rosenstein_le_estimator(
        x=x,
        embedding_dim=1,
        temporal_exclusion=0,
        numeric_backend="mpmath",
    )

    assert result["valid"]
    assert result[
        "n_reconstructed_states"
    ] == len(x)

    assert result[
        "n_neighbor_pairs"
    ] == len(x)

    assert mp.isfinite(
        result["lambda_hat"]
    )
