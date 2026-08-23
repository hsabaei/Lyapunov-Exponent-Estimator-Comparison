"""
Reproducible scalar adaptation of the Sun-Wu MISO RBF Lyapunov estimator.

Reference
---------
Y. Sun and C.Q. Wu,
"A radial-basis-function network-based method of estimating Lyapunov
exponents from a scalar time series for analyzing nonlinear systems
stability," Nonlinear Dynamics 70, 1689-1708 (2012).

Scope of this implementation
----------------------------
This project currently benchmarks one-dimensional discrete-time maps,
so the delay reconstruction specializes to

    x_t -> x_{t+1}.

The paper specifies:
- Gaussian radial basis functions,
- centers from K-means clustering,
- widths from a K-nearest-center heuristic,
- analytical differentiation of the trained RBF model.

Two details are not fixed by a universal rule in the paper: the number of
hidden nodes and the particular K used in the width heuristic. Also, the
paper trains output weights by gradient descent with momentum, but does
not give universal learning-rate/momentum values.

To make the baseline reproducible and truth-independent, this adaptation:
1. selects the hidden-node count by leave-one-pair-out one-step prediction
   MSE, never by Lyapunov-exponent error;
2. solves the linear output weights by least squares after centers and
   widths are fixed;
3. by default averages distances to all other centers for each width,
   which removes an otherwise unspecified nearest-center count.

The default maximum hidden-node count is 20, matching the hidden-node
count used in two reported scalar-time-series MISO examples (Henon and
biped) in Sun's thesis/paper. The actual selected value is still chosen
from prediction error only.

For high-precision synthetic trajectories, this module preserves the
relative information before converting the normalized series to float64.
The RBF fit itself is carried out in float64.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
from mpmath import mp


def _is_mpf(value: Any) -> bool:
    return isinstance(value, mp.mpf)


def _normalize_series_preserving_resolution(x: Iterable[Any]):
    """Normalize one scalar record with one affine transform.

    The same center and scale are used for both x_t and x_{t+1}; therefore
    d(normalized output)/d(normalized input) equals d(output)/d(input).

    If the input contains mpmath values, subtraction and scaling are done
    in mpmath *before* conversion to float64. This avoids losing tiny late
    fixed-point residuals merely because the absolute state is near L.
    """

    values = list(x)

    if len(values) < 5:
        raise ValueError("x must contain at least five observations.")

    use_mp = any(_is_mpf(v) for v in values)

    if use_mp:
        mp_values = [
            v if _is_mpf(v) else mp.mpf(str(v))
            for v in values
        ]

        if not all(mp.isfinite(v) for v in mp_values):
            raise ValueError("x contains nonfinite observations.")

        lo = min(mp_values)
        hi = max(mp_values)
        scale = hi - lo

        if scale == 0:
            raise ValueError("All observations are identical.")

        center = (hi + lo) / 2

        normalized = np.asarray(
            [float((v - center) / scale) for v in mp_values],
            dtype=float,
        )

        return normalized, center, scale, "mpmath_to_scaled_float64"

    arr = np.asarray(values, dtype=float)

    if arr.ndim != 1:
        raise ValueError("x must be a one-dimensional scalar time series.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains nonfinite observations.")

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    scale = hi - lo

    if scale == 0.0:
        raise ValueError("All observations are identical.")

    center = 0.5 * (hi + lo)
    normalized = (arr - center) / scale

    return normalized, center, scale, "scaled_float64"


def _kmeans_1d(values: np.ndarray, n_centers: int) -> np.ndarray:
    """Deterministic one-dimensional Lloyd K-means."""

    x = np.asarray(values, dtype=float)

    if x.ndim != 1:
        raise ValueError("K-means input must be one-dimensional.")

    if n_centers < 2:
        raise ValueError("n_centers must be at least 2.")

    unique_count = np.unique(x).size

    if n_centers > unique_count:
        raise ValueError(
            "n_centers exceeds the number of unique training inputs."
        )

    sorted_x = np.sort(x)
    n = sorted_x.size

    # Deterministic equal-count-bin initialization.
    centers = np.empty(n_centers, dtype=float)

    for group in range(n_centers):
        start = (group * n) // n_centers
        stop = ((group + 1) * n) // n_centers

        if stop <= start:
            stop = start + 1

        centers[group] = float(np.mean(sorted_x[start:stop]))

    # Lloyd iterations. Exact label stability is the stopping condition;
    # max_iter is only a safety guard, not an estimation threshold.
    previous_labels = None

    for _ in range(1000):
        distances = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)

        if previous_labels is not None and np.array_equal(
            labels,
            previous_labels,
        ):
            break

        previous_labels = labels.copy()

        new_centers = centers.copy()

        for j in range(n_centers):
            members = x[labels == j]

            if members.size > 0:
                new_centers[j] = float(np.mean(members))

        centers = np.sort(new_centers)
    else:
        raise RuntimeError("Deterministic 1D K-means did not converge.")

    if np.unique(centers).size != n_centers:
        raise ValueError("K-means produced duplicate centers.")

    return centers


def _center_widths(
    centers: np.ndarray,
    width_neighbors: int | None,
) -> tuple[np.ndarray, int]:
    """Sun-Wu K-nearest-center width heuristic.

    sigma_i is the mean distance from center i to its q nearest *other*
    centers. If width_neighbors is None, all other centers are used so that
    no extra nearest-center count must be tuned.
    """

    centers = np.asarray(centers, dtype=float)
    n_centers = centers.size

    if n_centers < 2:
        raise ValueError("At least two centers are required for widths.")

    if width_neighbors is None:
        q = n_centers - 1
    else:
        if not isinstance(width_neighbors, (int, np.integer)):
            raise ValueError("width_neighbors must be an integer or None.")

        q = int(width_neighbors)

        if q < 1 or q > n_centers - 1:
            raise ValueError(
                "width_neighbors must lie between 1 and n_centers - 1."
            )

    pairwise = np.abs(centers[:, None] - centers[None, :])
    np.fill_diagonal(pairwise, np.inf)

    nearest = np.sort(pairwise, axis=1)[:, :q]
    widths = np.mean(nearest, axis=1)

    if not np.all(np.isfinite(widths)):
        raise ValueError("Nonfinite RBF width encountered.")

    if np.any(widths <= 0.0):
        raise ValueError("RBF widths must be strictly positive.")

    return widths, q


def _design_matrix(
    inputs: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=float)

    phi = np.exp(
        -(
            (inputs[:, None] - centers[None, :]) ** 2
        )
        / (
            2.0 * widths[None, :] ** 2
        )
    )

    return np.column_stack(
        [
            np.ones(inputs.size, dtype=float),
            phi,
        ]
    )


def _fit_rbf(
    inputs: np.ndarray,
    outputs: np.ndarray,
    n_hidden: int,
    width_neighbors: int | None,
):
    centers = _kmeans_1d(
        values=inputs,
        n_centers=n_hidden,
    )

    widths, q = _center_widths(
        centers=centers,
        width_neighbors=width_neighbors,
    )

    design = _design_matrix(
        inputs=inputs,
        centers=centers,
        widths=widths,
    )

    weights, _, rank, _ = np.linalg.lstsq(
        design,
        outputs,
        rcond=None,
    )

    if rank < 1:
        raise ValueError("RBF least-squares fit has zero rank.")

    if not np.all(np.isfinite(weights)):
        raise ValueError("Nonfinite RBF weight encountered.")

    return centers, widths, weights, q


def _predict_and_derivative(
    inputs: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    weights: np.ndarray,
):
    """Evaluate fitted normalized RBF and its analytical derivative."""

    inputs = np.asarray(inputs, dtype=float)

    phi = np.exp(
        -(
            (inputs[:, None] - centers[None, :]) ** 2
        )
        / (
            2.0 * widths[None, :] ** 2
        )
    )

    predictions = (
        weights[0]
        + phi @ weights[1:]
    )

    derivatives = (
        phi
        * (
            (centers[None, :] - inputs[:, None])
            / (widths[None, :] ** 2)
        )
    ) @ weights[1:]

    return predictions, derivatives


def _candidate_hidden_counts(
    inputs: np.ndarray,
    max_hidden_nodes: int,
) -> list[int]:
    n_pairs = inputs.size
    unique_count = np.unique(inputs).size

    # Leave-one-out training sets contain n_pairs - 1 observations and may
    # contain one fewer unique input than the full set. Keep one degree of
    # feasibility margin for every fold.
    upper = min(
        int(max_hidden_nodes),
        n_pairs - 2,
        unique_count - 1,
    )

    if upper < 2:
        return []

    return list(range(2, upper + 1))


def _select_hidden_nodes_loo(
    inputs: np.ndarray,
    outputs: np.ndarray,
    candidates: list[int],
    width_neighbors: int | None,
):
    """Truth-free model selection using one-step leave-one-pair-out MSE."""

    n_pairs = inputs.size
    scores: dict[int, float] = {}

    for n_hidden in candidates:
        squared_errors = []
        candidate_valid = True

        for holdout in range(n_pairs):
            mask = np.ones(n_pairs, dtype=bool)
            mask[holdout] = False

            try:
                centers, widths, weights, _ = _fit_rbf(
                    inputs=inputs[mask],
                    outputs=outputs[mask],
                    n_hidden=n_hidden,
                    width_neighbors=width_neighbors,
                )

                prediction, _ = _predict_and_derivative(
                    inputs=np.asarray([inputs[holdout]], dtype=float),
                    centers=centers,
                    widths=widths,
                    weights=weights,
                )

                error = float(prediction[0] - outputs[holdout])

                if not np.isfinite(error):
                    candidate_valid = False
                    break

                squared_errors.append(error * error)

            except Exception:
                candidate_valid = False
                break

        if candidate_valid and squared_errors:
            scores[n_hidden] = float(np.mean(squared_errors))

    if not scores:
        raise ValueError(
            "No hidden-node count produced a valid leave-one-out RBF fit."
        )

    # Lowest prediction MSE wins. If machine-identical scores tie, choose
    # the smaller network.
    selected = min(
        scores,
        key=lambda k: (scores[k], k),
    )

    return selected, scores


def sun_wu_scalar_rbf_le_estimator(
    x,
    max_hidden_nodes: int = 20,
    width_neighbors: int | None = None,
):
    """Estimate a scalar discrete-time LE using a Sun-Wu-style MISO RBF.

    Parameters
    ----------
    x : one-dimensional sequence
        Observations x_0, ..., x_M. Consecutive pairs train x_t -> x_{t+1}.
        mpmath values are supported; normalization occurs before float64
        conversion so tiny relative fixed-point residuals can be retained.

    max_hidden_nodes : int, default=20
        Largest hidden-node count considered during truth-free leave-one-pair-
        out prediction-error selection. The actual chosen count is returned.

    width_neighbors : int or None, default=None
        Number of nearest *other* centers averaged in the RBF width heuristic.
        None uses all other centers, avoiding an additional tuned parameter.

    Returns
    -------
    dict
        lambda_hat, selected_hidden_nodes, cv_mse_by_hidden_nodes,
        centers, widths, weights, local_derivatives, normalized_inputs,
        n_pairs, width_neighbors_used, runtime_sec, valid, failure_reason.
    """

    start_time = time.perf_counter()

    try:
        if not isinstance(max_hidden_nodes, (int, np.integer)):
            raise ValueError("max_hidden_nodes must be an integer.")

        if max_hidden_nodes < 2:
            raise ValueError("max_hidden_nodes must be at least 2.")

        normalized, center, scale, scaling_mode = (
            _normalize_series_preserving_resolution(x)
        )

        inputs = normalized[:-1]
        outputs = normalized[1:]

        if inputs.size < 4:
            raise ValueError("At least four one-step pairs are required.")

        candidates = _candidate_hidden_counts(
            inputs=inputs,
            max_hidden_nodes=max_hidden_nodes,
        )

        if not candidates:
            raise ValueError(
                "Not enough distinct one-step inputs for at least two RBF centers."
            )

        selected_hidden, cv_scores = _select_hidden_nodes_loo(
            inputs=inputs,
            outputs=outputs,
            candidates=candidates,
            width_neighbors=width_neighbors,
        )

        centers, widths, weights, width_q = _fit_rbf(
            inputs=inputs,
            outputs=outputs,
            n_hidden=selected_hidden,
            width_neighbors=width_neighbors,
        )

        predictions, derivatives = _predict_and_derivative(
            inputs=inputs,
            centers=centers,
            widths=widths,
            weights=weights,
        )

        if not np.all(np.isfinite(derivatives)):
            raise ValueError("Nonfinite analytical RBF derivative encountered.")

        zero_derivative_count = int(np.sum(derivatives == 0.0))

        if zero_derivative_count > 0:
            lambda_hat = -np.inf
        else:
            lambda_hat = float(
                np.mean(
                    np.log(
                        np.abs(derivatives)
                    )
                )
            )

        training_mse = float(
            np.mean(
                (predictions - outputs) ** 2
            )
        )

        return {
            "lambda_hat": lambda_hat,
            "selected_hidden_nodes": selected_hidden,
            "candidate_hidden_nodes": candidates,
            "cv_mse_by_hidden_nodes": cv_scores,
            "training_mse": training_mse,
            "centers": centers,
            "widths": widths,
            "weights": weights,
            "local_derivatives": derivatives,
            "normalized_inputs": inputs,
            "normalized_outputs": outputs,
            "n_pairs": int(inputs.size),
            "width_neighbors_used": width_q,
            "normalization_center": center,
            "normalization_scale": scale,
            "scaling_mode": scaling_mode,
            "zero_derivative_count": zero_derivative_count,
            "runtime_sec": time.perf_counter() - start_time,
            "valid": True,
            "failure_reason": None,
        }

    except Exception as exc:
        return {
            "lambda_hat": np.nan,
            "selected_hidden_nodes": None,
            "candidate_hidden_nodes": [],
            "cv_mse_by_hidden_nodes": {},
            "training_mse": np.nan,
            "centers": np.asarray([], dtype=float),
            "widths": np.asarray([], dtype=float),
            "weights": np.asarray([], dtype=float),
            "local_derivatives": np.asarray([], dtype=float),
            "normalized_inputs": np.asarray([], dtype=float),
            "normalized_outputs": np.asarray([], dtype=float),
            "n_pairs": 0,
            "width_neighbors_used": None,
            "normalization_center": None,
            "normalization_scale": None,
            "scaling_mode": None,
            "zero_derivative_count": 0,
            "runtime_sec": time.perf_counter() - start_time,
            "valid": False,
            "failure_reason": str(exc),
        }
