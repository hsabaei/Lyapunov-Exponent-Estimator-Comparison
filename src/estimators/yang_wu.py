"""
Scalar specialization of the Yang-Wu nonlinear local-mapping LE estimator.

Reference:
C. Yang and C.Q. Wu,
"A robust method on estimation of Lyapunov exponents from a noisy time series,"
Nonlinear Dynamics 64, 279-292 (2011).

This module implements the core SECOND-ORDER NONLINEAR LOCAL MAPPING
for the present project's scalar, discrete-time, noiseless benchmarks.

Important:
- The 2011 paper's additional noise-robust steps (inner-radius selection and
  neighbor-vector averaging) are intentionally NOT enabled here. The paper
  selects the inner radius by trial and error, so using it in the noiseless
  benchmark would introduce a truth-tuned parameter.
- For d_E = 1 and Taylor order N_Tay = 2, the paper's parameter-count formula
  gives N_p = 2. Therefore two neighboring displacement pairs are sufficient
  for the quadratic local fit.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _validate_1d_series(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional scalar time series.")

    if x.size < 4:
        raise ValueError("x must contain at least four observations.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x contains nonfinite observations.")

    return x


def yang_wu_scalar_le_estimator(
    x,
    evolution: int = 1,
    r_min: float | None = None,
):
    """
    Estimate a scalar Lyapunov exponent using a second-order local map.

    Parameters
    ----------
    x : array-like, shape (N,)
        Scalar observed trajectory.

    evolution : int, default=1
        Number of discrete map steps between the initial and evolved
        neighboring displacement. For the present discrete-time benchmark
        we use evolution=1.

    r_min : float or None, default=None
        Optional Yang-Wu inner-radius exclusion.

        If None, no positive inner radius is imposed; only exact duplicate
        observations are excluded. This is the intended setting for the
        present noiseless benchmark.

        The 2011 paper selects r_min by trial and error for noisy data.
        Therefore this implementation does not invent a default positive
        threshold.

    Returns
    -------
    dict
        Keys include:
            lambda_hat
            local_jacobians
            fiducial_indices
            n_valid_local_maps
            n_attempted_local_maps
            evolution
            r_min
            runtime_sec
            valid
            failure_reason

    Notes
    -----
    For a fiducial point x_i and a neighboring observation x_j, define

        d_j = x_j - x_i

    and after `evolution` steps

        z_j = x_{j+m} - x_{i+m}.

    In the scalar second-order specialization we fit

        z_j ≈ c1 * d_j + (1/2) * c2 * d_j^2.

    The coefficient c1 is the reconstructed local Jacobian at the fiducial
    point because it is the first derivative of the Taylor map at d=0.

    For d_E=1 and N_Tay=2, Yang-Wu's formula gives N_p=2, so the two nearest
    eligible neighboring displacement pairs are used.

    With unit-time discrete dynamics and evolution=1,

        lambda_hat = mean_i log(|c1_i|).

    More generally, an m-step local map is normalized per discrete map step:

        lambda_hat = (1/m) * mean_i log(|c1_i|).
    """

    start_time = time.perf_counter()

    try:
        x = _validate_1d_series(x)

        if not isinstance(evolution, (int, np.integer)) or evolution < 1:
            raise ValueError("evolution must be a positive integer.")

        if r_min is not None:
            r_min = float(r_min)

            if not np.isfinite(r_min):
                raise ValueError("r_min must be finite.")

            if r_min < 0.0:
                raise ValueError("r_min must be nonnegative.")

        n = x.size

        # A neighbor j is usable only if j + evolution exists.
        last_base_index = n - evolution - 1

        local_jacobians = []
        fiducial_indices = []

        attempted = 0

        for i in range(last_base_index + 1):

            attempted += 1

            candidates = []

            for j in range(last_base_index + 1):

                if j == i:
                    continue

                displacement = x[j] - x[i]
                distance = abs(displacement)

                # Exact duplicates cannot identify a local map.
                if distance == 0.0:
                    continue

                # Optional Yang-Wu inner-radius exclusion.
                if r_min is not None and distance < r_min:
                    continue

                candidates.append(
                    (
                        distance,
                        j,
                    )
                )

            # d_E=1 and N_Tay=2 => N_p=2.
            if len(candidates) < 2:
                continue

            candidates.sort(
                key=lambda item: item[0]
            )

            neighbor_indices = [
                candidates[0][1],
                candidates[1][1],
            ]

            d = np.array(
                [
                    x[j] - x[i]
                    for j in neighbor_indices
                ],
                dtype=float,
            )

            z = np.array(
                [
                    x[j + evolution]
                    - x[i + evolution]
                    for j in neighbor_indices
                ],
                dtype=float,
            )

            # Second-order Taylor design:
            #
            # z = c1*d + (1/2)*c2*d^2
            design = np.column_stack(
                (
                    d,
                    0.5 * d * d,
                )
            )

            coefficients, _, rank, _ = np.linalg.lstsq(
                design,
                z,
                rcond=None,
            )

            if rank < 2:
                continue

            local_jacobian = float(
                coefficients[0]
            )

            if not np.isfinite(local_jacobian):
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

                "runtime_sec":
                    time.perf_counter()
                    - start_time,

                "valid":
                    False,

                "failure_reason":
                    "No valid second-order local maps could be reconstructed.",
            }

        local_jacobians = np.asarray(
            local_jacobians,
            dtype=float,
        )

        log_stretches = np.log(
            np.abs(local_jacobians)
        )

        lambda_hat = float(
            np.mean(log_stretches)
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

            "runtime_sec":
                time.perf_counter()
                - start_time,

            "valid":
                True,

            "failure_reason":
                None,
        }

    except Exception as exc:

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
                0,

            "evolution":
                evolution,

            "r_min":
                r_min,

            "runtime_sec":
                time.perf_counter()
                - start_time,

            "valid":
                False,

            "failure_reason":
                str(exc),
        }
