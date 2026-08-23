"""
Scalar specialization of the Yang-Wu second-order nonlinear local-mapping
Lyapunov-exponent estimator.

Reference:
C. Yang and C.Q. Wu,
"A robust method on estimation of Lyapunov exponents from a noisy time series,"
Nonlinear Dynamics 64, 279-292 (2011).

This version supports both:
    numeric_backend="float64"
    numeric_backend="mpmath"

For the present noiseless scalar benchmarks:
- embedding dimension d_E = 1
- Taylor order N_Tay = 2
- minimum parameter count N_p = 2
- no positive inner-radius threshold is imposed unless explicitly supplied

The paper's noise-specific averaging and trial-and-error inner-radius selection
are not activated in the noiseless benchmark.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from mpmath import mp


def _validate_float_series(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(
            "x must be a one-dimensional scalar time series."
        )

    if x.size < 4:
        raise ValueError(
            "x must contain at least four observations."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "x contains nonfinite observations."
        )

    return x


def _to_mpf(value):
    if isinstance(value, mp.mpf):
        return value

    return mp.mpf(str(value))


def _validate_mp_series(x):
    x = [_to_mpf(v) for v in x]

    if len(x) < 4:
        raise ValueError(
            "x must contain at least four observations."
        )

    if not all(mp.isfinite(v) for v in x):
        raise ValueError(
            "x contains nonfinite observations."
        )

    return x


def _float64_estimator(
    x,
    evolution,
    r_min,
):
    x = _validate_float_series(x)

    if r_min is not None:
        r_min = float(r_min)

        if not np.isfinite(r_min):
            raise ValueError(
                "r_min must be finite."
            )

        if r_min < 0.0:
            raise ValueError(
                "r_min must be nonnegative."
            )

    n = x.size

    last_base_index = (
        n - evolution - 1
    )

    local_jacobians = []
    fiducial_indices = []

    attempted = 0

    for i in range(
        last_base_index + 1
    ):
        attempted += 1

        candidates = []

        for j in range(
            last_base_index + 1
        ):
            if j == i:
                continue

            displacement = (
                x[j] - x[i]
            )

            distance = abs(
                displacement
            )

            if distance == 0.0:
                continue

            if (
                r_min is not None
                and distance < r_min
            ):
                continue

            candidates.append(
                (distance, j)
            )

        # d_E=1 and N_Tay=2:
        # N_p = 2.
        if len(candidates) < 2:
            continue

        candidates.sort(
            key=lambda item: item[0]
        )

        j1 = candidates[0][1]
        j2 = candidates[1][1]

        d1 = x[j1] - x[i]
        d2 = x[j2] - x[i]

        z1 = (
            x[j1 + evolution]
            - x[i + evolution]
        )

        z2 = (
            x[j2 + evolution]
            - x[i + evolution]
        )

        design = np.array(
            [
                [
                    d1,
                    0.5 * d1 * d1,
                ],
                [
                    d2,
                    0.5 * d2 * d2,
                ],
            ],
            dtype=float,
        )

        response = np.array(
            [z1, z2],
            dtype=float,
        )

        coefficients, _, rank, _ = (
            np.linalg.lstsq(
                design,
                response,
                rcond=None,
            )
        )

        if rank < 2:
            continue

        local_jacobian = float(
            coefficients[0]
        )

        if not np.isfinite(
            local_jacobian
        ):
            continue

        if local_jacobian == 0.0:
            continue

        local_jacobians.append(
            local_jacobian
        )

        fiducial_indices.append(
            i
        )

    if len(local_jacobians) == 0:
        return {
            "lambda_hat":
                np.nan,

            "local_jacobians":
                np.asarray(
                    [],
                    dtype=float,
                ),

            "fiducial_indices":
                np.asarray(
                    [],
                    dtype=int,
                ),

            "n_valid_local_maps":
                0,

            "n_attempted_local_maps":
                attempted,

            "evolution":
                evolution,

            "r_min":
                r_min,

            "numeric_backend":
                "float64",

            "valid":
                False,

            "failure_reason":
                "No valid second-order local maps could be reconstructed.",
        }

    local_jacobians = np.asarray(
        local_jacobians,
        dtype=float,
    )

    lambda_hat = float(
        np.mean(
            np.log(
                np.abs(
                    local_jacobians
                )
            )
        )
        / evolution
    )

    return {
        "lambda_hat":
            lambda_hat,

        "local_jacobians":
            local_jacobians,

        "fiducial_indices":
            np.asarray(
                fiducial_indices,
                dtype=int,
            ),

        "n_valid_local_maps":
            local_jacobians.size,

        "n_attempted_local_maps":
            attempted,

        "evolution":
            evolution,

        "r_min":
            r_min,

        "numeric_backend":
            "float64",

        "valid":
            True,

        "failure_reason":
            None,
    }


def _mpmath_estimator(
    x,
    evolution,
    r_min,
):
    x = _validate_mp_series(x)

    if r_min is not None:
        r_min = _to_mpf(
            r_min
        )

        if not mp.isfinite(
            r_min
        ):
            raise ValueError(
                "r_min must be finite."
            )

        if r_min < 0:
            raise ValueError(
                "r_min must be nonnegative."
            )

    n = len(x)

    last_base_index = (
        n - evolution - 1
    )

    local_jacobians = []
    fiducial_indices = []

    attempted = 0

    half = mp.mpf("0.5")

    for i in range(
        last_base_index + 1
    ):
        attempted += 1

        candidates = []

        for j in range(
            last_base_index + 1
        ):
            if j == i:
                continue

            displacement = (
                x[j] - x[i]
            )

            distance = mp.fabs(
                displacement
            )

            if distance == 0:
                continue

            if (
                r_min is not None
                and distance < r_min
            ):
                continue

            candidates.append(
                (distance, j)
            )

        if len(candidates) < 2:
            continue

        candidates.sort(
            key=lambda item: item[0]
        )

        j1 = candidates[0][1]
        j2 = candidates[1][1]

        d1 = x[j1] - x[i]
        d2 = x[j2] - x[i]

        z1 = (
            x[j1 + evolution]
            - x[i + evolution]
        )

        z2 = (
            x[j2 + evolution]
            - x[i + evolution]
        )

        q1 = half * d1 * d1
        q2 = half * d2 * d2

        determinant = (
            d1 * q2
            - d2 * q1
        )

        if determinant == 0:
            continue

        # Exact 2 x 2 solve:
        #
        # [d1 q1] [c1] = [z1]
        # [d2 q2] [c2]   [z2]
        #
        # c1 is the reconstructed
        # local first derivative.
        c1 = (
            z1 * q2
            - z2 * q1
        ) / determinant

        if not mp.isfinite(c1):
            continue

        if c1 == 0:
            continue

        local_jacobians.append(
            c1
        )

        fiducial_indices.append(
            i
        )

    if len(local_jacobians) == 0:
        return {
            "lambda_hat":
                mp.nan,

            "local_jacobians":
                [],

            "fiducial_indices":
                [],

            "n_valid_local_maps":
                0,

            "n_attempted_local_maps":
                attempted,

            "evolution":
                evolution,

            "r_min":
                r_min,

            "numeric_backend":
                "mpmath",

            "valid":
                False,

            "failure_reason":
                "No valid second-order local maps could be reconstructed.",
        }

    lambda_hat = (
        mp.fsum(
            [
                mp.log(
                    mp.fabs(jac)
                )
                for jac
                in local_jacobians
            ]
        )
        / len(
            local_jacobians
        )
        / evolution
    )

    return {
        "lambda_hat":
            lambda_hat,

        "local_jacobians":
            local_jacobians,

        "fiducial_indices":
            fiducial_indices,

        "n_valid_local_maps":
            len(
                local_jacobians
            ),

        "n_attempted_local_maps":
            attempted,

        "evolution":
            evolution,

        "r_min":
            r_min,

        "numeric_backend":
            "mpmath",

        "valid":
            True,

        "failure_reason":
            None,
    }


def yang_wu_scalar_le_estimator(
    x,
    evolution: int = 1,
    r_min=None,
    numeric_backend: str = "float64",
):
    """
    Estimate a scalar LE using the Yang-Wu
    second-order local-mapping principle.

    For the current discrete-time benchmarks,
    use evolution=1.

    numeric_backend:
        "float64"
        "mpmath"
    """

    start_time = (
        time.perf_counter()
    )

    try:
        if (
            not isinstance(
                evolution,
                (int, np.integer),
            )
            or evolution < 1
        ):
            raise ValueError(
                "evolution must be a positive integer."
            )

        if (
            numeric_backend
            == "float64"
        ):
            result = (
                _float64_estimator(
                    x=x,
                    evolution=evolution,
                    r_min=r_min,
                )
            )

        elif (
            numeric_backend
            == "mpmath"
        ):
            result = (
                _mpmath_estimator(
                    x=x,
                    evolution=evolution,
                    r_min=r_min,
                )
            )

        else:
            raise ValueError(
                "numeric_backend must be 'float64' or 'mpmath'."
            )

        result["runtime_sec"] = (
            time.perf_counter()
            - start_time
        )

        return result

    except Exception as exc:
        invalid_value = (
            mp.nan
            if numeric_backend
            == "mpmath"
            else np.nan
        )

        return {
            "lambda_hat":
                invalid_value,

            "local_jacobians":
                [],

            "fiducial_indices":
                [],

            "n_valid_local_maps":
                0,

            "n_attempted_local_maps":
                0,

            "evolution":
                evolution,

            "r_min":
                r_min,

            "numeric_backend":
                numeric_backend,

            "runtime_sec":
                time.perf_counter()
                - start_time,

            "valid":
                False,

            "failure_reason":
                str(exc),
        }
