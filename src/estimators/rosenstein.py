"""
Reproducible fixed-point adaptation of the Rosenstein et al. largest-Lyapunov
estimator.

Reference
---------
M. T. Rosenstein, J. J. Collins, C. J. De Luca,
"A practical method for calculating largest Lyapunov exponents from small
data sets," Physica D 65 (1993), 117-134.

Original Rosenstein ingredients preserved here:
1. reconstruct states from delay coordinates;
2. choose one nearest neighbor for each reconstructed state;
3. follow the paired trajectories forward;
4. compute the mean log-separation curve
       y(h) = < log d_j(h) >_j;
5. estimate the exponent from a least-squares slope of y versus time.

Adaptations for the present stable fixed-point benchmark
--------------------------------------------------------
- The benchmark is already a scalar full-state map, so the default is
  embedding_dim=1 and lag=1.
- A convergent fixed-point trajectory has no meaningful oscillation period.
  Therefore the original "greater than the mean period" temporal exclusion
  is not imposed by default. temporal_exclusion=0 means only the point itself
  is excluded.
- Rosenstein et al. selected an approximately linear region, partly by visual
  inspection. To avoid a truth-guided or subjective fit window, this
  reproducible adaptation fits ALL available horizons unless max_horizon is
  explicitly supplied.
- No R^2 threshold, minimum-divergence threshold, or truth-based window
  selection is introduced.

This is therefore an ADAPTATION of Rosenstein's estimator to the negative
fixed-point setting, not a claim of an exact reproduction of the paper's
chaotic-attractor protocol.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from mpmath import mp


def _to_mpf(value):
    if isinstance(value, mp.mpf):
        return value
    return mp.mpf(str(value))


def _validate_common(
    embedding_dim,
    lag,
    temporal_exclusion,
    sample_time,
    max_horizon,
):
    if (
        not isinstance(embedding_dim, (int, np.integer))
        or embedding_dim < 1
    ):
        raise ValueError("embedding_dim must be a positive integer.")

    if (
        not isinstance(lag, (int, np.integer))
        or lag < 1
    ):
        raise ValueError("lag must be a positive integer.")

    if (
        not isinstance(temporal_exclusion, (int, np.integer))
        or temporal_exclusion < 0
    ):
        raise ValueError(
            "temporal_exclusion must be a nonnegative integer."
        )

    if sample_time <= 0:
        raise ValueError("sample_time must be positive.")

    if max_horizon is not None:
        if (
            not isinstance(max_horizon, (int, np.integer))
            or max_horizon < 1
        ):
            raise ValueError(
                "max_horizon must be None or a positive integer."
            )


def _build_embedding_float(
    x,
    embedding_dim,
    lag,
):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(
            "x must be a one-dimensional scalar time series."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("x contains nonfinite values.")

    m = (
        x.size
        - (embedding_dim - 1) * lag
    )

    if m < 3:
        raise ValueError(
            "Not enough observations for the requested embedding."
        )

    states = np.empty(
        (m, embedding_dim),
        dtype=float,
    )

    for i in range(m):
        for q in range(embedding_dim):
            states[i, q] = x[i + q * lag]

    return states


def _build_embedding_mp(
    x,
    embedding_dim,
    lag,
):
    x = [_to_mpf(v) for v in x]

    if not all(mp.isfinite(v) for v in x):
        raise ValueError("x contains nonfinite values.")

    m = (
        len(x)
        - (embedding_dim - 1) * lag
    )

    if m < 3:
        raise ValueError(
            "Not enough observations for the requested embedding."
        )

    states = []

    for i in range(m):
        states.append(
            tuple(
                x[i + q * lag]
                for q in range(embedding_dim)
            )
        )

    return states


def _dist_float(a, b):
    return float(
        np.linalg.norm(a - b)
    )


def _dist_mp(a, b):
    return mp.sqrt(
        mp.fsum(
            [
                (u - v) ** 2
                for u, v in zip(a, b)
            ]
        )
    )


def _nearest_pairs_float(
    states,
    temporal_exclusion,
):
    m = states.shape[0]
    pairs = []

    for j in range(m):
        best_k = None
        best_distance = np.inf

        for k in range(m):
            if k == j:
                continue

            if abs(k - j) <= temporal_exclusion:
                continue

            distance = _dist_float(
                states[j],
                states[k],
            )

            if (
                not np.isfinite(distance)
                or distance <= 0.0
            ):
                continue

            if distance < best_distance:
                best_distance = distance
                best_k = k

        if best_k is not None:
            pairs.append(
                (j, best_k, best_distance)
            )

    return pairs


def _nearest_pairs_mp(
    states,
    temporal_exclusion,
):
    m = len(states)
    pairs = []

    for j in range(m):
        best_k = None
        best_distance = None

        for k in range(m):
            if k == j:
                continue

            if abs(k - j) <= temporal_exclusion:
                continue

            distance = _dist_mp(
                states[j],
                states[k],
            )

            if (
                not mp.isfinite(distance)
                or distance <= 0
            ):
                continue

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_distance = distance
                best_k = k

        if best_k is not None:
            pairs.append(
                (j, best_k, best_distance)
            )

    return pairs


def _curve_float(
    states,
    pairs,
    max_horizon,
):
    m = states.shape[0]

    if max_horizon is None:
        horizon_cap = m - 1
    else:
        horizon_cap = min(
            max_horizon,
            m - 1,
        )

    horizons = []
    mean_log_distance = []
    pair_counts = []

    for h in range(horizon_cap + 1):
        values = []

        for j, k, _ in pairs:
            jj = j + h
            kk = k + h

            if jj >= m or kk >= m:
                continue

            distance = _dist_float(
                states[jj],
                states[kk],
            )

            if (
                not np.isfinite(distance)
                or distance <= 0.0
            ):
                continue

            values.append(
                np.log(distance)
            )

        if len(values) == 0:
            break

        horizons.append(h)
        mean_log_distance.append(
            float(np.mean(values))
        )
        pair_counts.append(
            len(values)
        )

    return (
        np.asarray(horizons, dtype=int),
        np.asarray(mean_log_distance, dtype=float),
        np.asarray(pair_counts, dtype=int),
    )


def _curve_mp(
    states,
    pairs,
    max_horizon,
):
    m = len(states)

    if max_horizon is None:
        horizon_cap = m - 1
    else:
        horizon_cap = min(
            max_horizon,
            m - 1,
        )

    horizons = []
    mean_log_distance = []
    pair_counts = []

    for h in range(horizon_cap + 1):
        values = []

        for j, k, _ in pairs:
            jj = j + h
            kk = k + h

            if jj >= m or kk >= m:
                continue

            distance = _dist_mp(
                states[jj],
                states[kk],
            )

            if (
                not mp.isfinite(distance)
                or distance <= 0
            ):
                continue

            values.append(
                mp.log(distance)
            )

        if len(values) == 0:
            break

        horizons.append(h)
        mean_log_distance.append(
            mp.fsum(values)
            / len(values)
        )
        pair_counts.append(
            len(values)
        )

    return (
        horizons,
        mean_log_distance,
        pair_counts,
    )


def _slope_float(
    horizons,
    y,
    sample_time,
):
    if len(horizons) < 2:
        raise ValueError(
            "At least two divergence horizons are required."
        )

    t = (
        np.asarray(horizons, dtype=float)
        * float(sample_time)
    )

    design = np.column_stack(
        (
            np.ones_like(t),
            t,
        )
    )

    coef, _, rank, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    if rank < 2:
        raise ValueError(
            "The divergence-curve regression is rank deficient."
        )

    intercept = float(coef[0])
    slope = float(coef[1])

    return slope, intercept


def _slope_mp(
    horizons,
    y,
    sample_time,
):
    if len(horizons) < 2:
        raise ValueError(
            "At least two divergence horizons are required."
        )

    dt = _to_mpf(sample_time)

    t = [
        dt * h
        for h in horizons
    ]

    t_bar = (
        mp.fsum(t)
        / len(t)
    )

    y_bar = (
        mp.fsum(y)
        / len(y)
    )

    denominator = mp.fsum(
        [
            (ti - t_bar) ** 2
            for ti in t
        ]
    )

    if denominator == 0:
        raise ValueError(
            "The divergence-curve regression is rank deficient."
        )

    numerator = mp.fsum(
        [
            (ti - t_bar)
            * (yi - y_bar)
            for ti, yi in zip(t, y)
        ]
    )

    slope = (
        numerator
        / denominator
    )

    intercept = (
        y_bar
        - slope * t_bar
    )

    return slope, intercept


def rosenstein_le_estimator(
    x: Any,
    embedding_dim: int = 1,
    lag: int = 1,
    temporal_exclusion: int = 0,
    sample_time=1.0,
    max_horizon: int | None = None,
    numeric_backend: str = "float64",
):
    """
    Estimate the largest Lyapunov exponent using a reproducible Rosenstein
    adaptation.

    Parameters
    ----------
    x
        Scalar time series.

    embedding_dim : int, default=1
        Delay-embedding dimension. For the present scalar full-state map
        benchmark, use 1.

    lag : int, default=1
        Delay-coordinate lag. With embedding_dim=1 it does not affect the
        reconstructed state.

    temporal_exclusion : int, default=0
        Candidate nearest neighbors must satisfy
            |j-k| > temporal_exclusion.
        The original paper uses a mean-period exclusion. A convergent fixed
        point has no meaningful mean period, so the fixed-point adaptation
        uses 0: exclude only the reference point itself.

    sample_time : positive number, default=1
        Sampling interval.

    max_horizon : int or None, default=None
        If None, fit every available horizon. No subjective linear-region
        selection is performed.

    numeric_backend : {"float64", "mpmath"}
        Arithmetic backend.

    Returns
    -------
    dict
        lambda_hat
        intercept
        horizons
        mean_log_distance
        pair_counts
        neighbor_pairs
        n_reconstructed_states
        n_neighbor_pairs
        embedding_dim
        lag
        temporal_exclusion
        sample_time
        max_horizon
        numeric_backend
        runtime_sec
        valid
        failure_reason
    """

    start_time = time.perf_counter()

    try:
        _validate_common(
            embedding_dim=embedding_dim,
            lag=lag,
            temporal_exclusion=temporal_exclusion,
            sample_time=sample_time,
            max_horizon=max_horizon,
        )

        if numeric_backend == "float64":
            states = _build_embedding_float(
                x=x,
                embedding_dim=embedding_dim,
                lag=lag,
            )

            pairs = _nearest_pairs_float(
                states=states,
                temporal_exclusion=temporal_exclusion,
            )

            if len(pairs) == 0:
                raise ValueError(
                    "No valid nearest-neighbor pairs were found."
                )

            (
                horizons,
                mean_log_distance,
                pair_counts,
            ) = _curve_float(
                states=states,
                pairs=pairs,
                max_horizon=max_horizon,
            )

            (
                lambda_hat,
                intercept,
            ) = _slope_float(
                horizons=horizons,
                y=mean_log_distance,
                sample_time=sample_time,
            )

        elif numeric_backend == "mpmath":
            states = _build_embedding_mp(
                x=x,
                embedding_dim=embedding_dim,
                lag=lag,
            )

            pairs = _nearest_pairs_mp(
                states=states,
                temporal_exclusion=temporal_exclusion,
            )

            if len(pairs) == 0:
                raise ValueError(
                    "No valid nearest-neighbor pairs were found."
                )

            (
                horizons,
                mean_log_distance,
                pair_counts,
            ) = _curve_mp(
                states=states,
                pairs=pairs,
                max_horizon=max_horizon,
            )

            (
                lambda_hat,
                intercept,
            ) = _slope_mp(
                horizons=horizons,
                y=mean_log_distance,
                sample_time=sample_time,
            )

        else:
            raise ValueError(
                "numeric_backend must be 'float64' or 'mpmath'."
            )

        return {
            "lambda_hat":
                lambda_hat,

            "intercept":
                intercept,

            "horizons":
                horizons,

            "mean_log_distance":
                mean_log_distance,

            "pair_counts":
                pair_counts,

            "neighbor_pairs":
                pairs,

            "n_reconstructed_states":
                len(states),

            "n_neighbor_pairs":
                len(pairs),

            "embedding_dim":
                embedding_dim,

            "lag":
                lag,

            "temporal_exclusion":
                temporal_exclusion,

            "sample_time":
                sample_time,

            "max_horizon":
                max_horizon,

            "numeric_backend":
                numeric_backend,

            "runtime_sec":
                time.perf_counter()
                - start_time,

            "valid":
                True,

            "failure_reason":
                None,
        }

    except Exception as exc:
        invalid = (
            mp.nan
            if numeric_backend == "mpmath"
            else np.nan
        )

        return {
            "lambda_hat":
                invalid,

            "intercept":
                invalid,

            "horizons":
                [],

            "mean_log_distance":
                [],

            "pair_counts":
                [],

            "neighbor_pairs":
                [],

            "n_reconstructed_states":
                0,

            "n_neighbor_pairs":
                0,

            "embedding_dim":
                embedding_dim,

            "lag":
                lag,

            "temporal_exclusion":
                temporal_exclusion,

            "sample_time":
                sample_time,

            "max_horizon":
                max_horizon,

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
