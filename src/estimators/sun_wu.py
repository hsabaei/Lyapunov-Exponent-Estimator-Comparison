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
This project benchmarks one-dimensional discrete-time maps, so the delay
reconstruction specializes to

    x_t -> x_{t+1}.

The source specifies:
- Gaussian radial basis functions,
- centers from K-means clustering,
- widths from a K-nearest-center heuristic,
- analytical differentiation of the trained RBF model.

Two model-selection details are not fixed by a universal rule in the source:
(1) the number of hidden nodes and (2) the number q of nearest centers used
in the width heuristic. The source also uses gradient descent with momentum
for output weights but does not supply universal learning-rate/momentum
choices.

To make the baseline reproducible and independent of the known Lyapunov
truth, this adaptation:
1. jointly selects hidden-node count l and width-neighbor count q using
   leave-one-pair-out one-step prediction MSE;
2. never uses Lyapunov-exponent error during model selection;
3. solves the linear output weights by least squares once centers and widths
   are fixed.

If width_neighbors is explicitly supplied, q is held fixed and only l is
selected by prediction CV. The default width_neighbors=None performs the
joint (l, q) selection.

The default maximum hidden-node count is 20, matching the scale used in
reported scalar-time-series MISO examples in Sun's thesis/paper. The actual
selected model is determined by prediction error only.

For high-precision synthetic trajectories, this module preserves relative
information before converting the normalized series to float64. The RBF fit
itself is carried out in float64.
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
    in mpmath before conversion to float64. This avoids losing tiny late
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

    centers = np.empty(n_centers, dtype=float)

    for group in range(n_centers):
        start = (group * n) // n_centers
        stop = ((group + 1) * n) // n_centers

        if stop <= start:
            stop = start + 1

        centers[group] = float(np.mean(sorted_x[start:stop]))

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
    width_neighbors: int,
) -> np.ndarray:
    """Sun-Wu K-nearest-center width heuristic.

    sigma_i is the mean distance from center i to its q nearest other centers,
    where q = width_neighbors.
    """

    centers = np.asarray(centers, dtype=float)
    n_centers = centers.size

    if n_centers < 2:
        raise ValueError("At least two centers are required for widths.")

    if not isinstance(width_neighbors, (int, np.integer)):
        raise ValueError("width_neighbors must be an integer.")

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

    return widths


def _design_matrix(
    inputs: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=float)

    phi = np.exp(
        -((inputs[:, None] - centers[None, :]) ** 2)
        / (2.0 * widths[None, :] ** 2)
    )

    return np.column_stack(
        [
            np.ones(inputs.size, dtype=float),
            phi,
        ]
    )


def _fit_rbf_with_centers(
    inputs: np.ndarray,
    outputs: np.ndarray,
    centers: np.ndarray,
    width_neighbors: int,
):
    widths = _center_widths(
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

    return widths, weights


def _fit_rbf(
    inputs: np.ndarray,
    outputs: np.ndarray,
    n_hidden: int,
    width_neighbors: int,
):
    centers = _kmeans_1d(
        values=inputs,
        n_centers=n_hidden,
    )

    widths, weights = _fit_rbf_with_centers(
        inputs=inputs,
        outputs=outputs,
        centers=centers,
        width_neighbors=width_neighbors,
    )

    return centers, widths, weights


def _predict_and_derivative(
    inputs: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    weights: np.ndarray,
):
    """Evaluate fitted normalized RBF and its analytical derivative."""

    inputs = np.asarray(inputs, dtype=float)

    phi = np.exp(
        -((inputs[:, None] - centers[None, :]) ** 2)
        / (2.0 * widths[None, :] ** 2)
    )

    predictions = weights[0] + phi @ weights[1:]

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
    fixed_width_neighbors: int | None,
) -> list[int]:
    n_pairs = inputs.size
    unique_count = np.unique(inputs).size

    # Every LOO training fold contains n_pairs - 1 observations. Keep one
    # feasibility margin so every fold can still form the requested centers.
    upper = min(
        int(max_hidden_nodes),
        n_pairs - 2,
        unique_count - 1,
    )

    lower = 2

    if fixed_width_neighbors is not None:
        lower = max(lower, int(fixed_width_neighbors) + 1)

    if upper < lower:
        return []

    return list(range(lower, upper + 1))


def _candidate_width_counts(
    n_hidden: int,
    fixed_width_neighbors: int | None,
) -> list[int]:
    if fixed_width_neighbors is None:
        return list(range(1, n_hidden))

    q = int(fixed_width_neighbors)

    if q < 1 or q > n_hidden - 1:
        return []

    return [q]


def _select_configuration_loo(
    inputs: np.ndarray,
    outputs: np.ndarray,
    hidden_candidates: list[int],
    fixed_width_neighbors: int | None,
):
    """Joint truth-free selection of (hidden nodes, width-neighbor count).

    For each candidate hidden-node count l and each admissible q, perform
    leave-one-pair-out one-step prediction. K-means centers are recomputed on
    each training fold, as they would be in a genuine held-out fit. Within a
    fold, centers are reused across q because q changes widths, not centers.
    """

    n_pairs = inputs.size
    squared_errors: dict[tuple[int, int], list[float]] = {}
    invalid_configurations: set[tuple[int, int]] = set()

    for n_hidden in hidden_candidates:
        q_candidates = _candidate_width_counts(
            n_hidden=n_hidden,
            fixed_width_neighbors=fixed_width_neighbors,
        )

        for q in q_candidates:
            squared_errors[(n_hidden, q)] = []

        for holdout in range(n_pairs):
            mask = np.ones(n_pairs, dtype=bool)
            mask[holdout] = False

            try:
                centers = _kmeans_1d(
                    values=inputs[mask],
                    n_centers=n_hidden,
                )
            except Exception:
                invalid_configurations.update(
                    (n_hidden, q) for q in q_candidates
                )
                break

            for q in q_candidates:
                configuration = (n_hidden, q)

                if configuration in invalid_configurations:
                    continue

                try:
                    widths, weights = _fit_rbf_with_centers(
                        inputs=inputs[mask],
                        outputs=outputs[mask],
                        centers=centers,
                        width_neighbors=q,
                    )

                    prediction, _ = _predict_and_derivative(
                        inputs=np.asarray([inputs[holdout]], dtype=float),
                        centers=centers,
                        widths=widths,
                        weights=weights,
                    )

                    error = float(prediction[0] - outputs[holdout])

                    if not np.isfinite(error):
                        invalid_configurations.add(configuration)
                        continue

                    squared_errors[configuration].append(error * error)

                except Exception:
                    invalid_configurations.add(configuration)

    scores: dict[tuple[int, int], float] = {}

    for configuration, errors in squared_errors.items():
        if configuration in invalid_configurations:
            continue

        if len(errors) != n_pairs:
            continue

        scores[configuration] = float(np.mean(errors))

    if not scores:
        raise ValueError(
            "No (hidden-node count, width-neighbor count) configuration "
            "produced a valid leave-one-out RBF fit."
        )

    # Lowest prediction MSE wins. If machine-identical scores tie, choose
    # the simpler model: fewer hidden nodes, then smaller q.
    selected = min(
        scores,
        key=lambda config: (
            scores[config],
            config[0],
            config[1],
        ),
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
        out prediction-error selection.

    width_neighbors : int or None, default=None
        If None, jointly select q in {1, ..., l-1} together with the hidden-
        node count l using leave-one-pair-out one-step prediction MSE.
        If an integer is supplied, hold q fixed and select only l.

    Returns
    -------
    dict
        lambda_hat, selected_hidden_nodes, selected_width_neighbors,
        cv_mse_by_configuration, centers, widths, weights,
        local_derivatives, n_pairs, runtime_sec, valid, failure_reason,
        plus compatibility summaries such as cv_mse_by_hidden_nodes.
    """

    start_time = time.perf_counter()

    try:
        if not isinstance(max_hidden_nodes, (int, np.integer)):
            raise ValueError("max_hidden_nodes must be an integer.")

        if max_hidden_nodes < 2:
            raise ValueError("max_hidden_nodes must be at least 2.")

        if width_neighbors is not None:
            if not isinstance(width_neighbors, (int, np.integer)):
                raise ValueError(
                    "width_neighbors must be a positive integer or None."
                )

            if int(width_neighbors) < 1:
                raise ValueError(
                    "width_neighbors must be a positive integer or None."
                )

            fixed_q = int(width_neighbors)
        else:
            fixed_q = None

        normalized, center, scale, scaling_mode = (
            _normalize_series_preserving_resolution(x)
        )

        inputs = normalized[:-1]
        outputs = normalized[1:]

        if inputs.size < 4:
            raise ValueError("At least four one-step pairs are required.")

        hidden_candidates = _candidate_hidden_counts(
            inputs=inputs,
            max_hidden_nodes=max_hidden_nodes,
            fixed_width_neighbors=fixed_q,
        )

        if not hidden_candidates:
            raise ValueError(
                "Not enough distinct one-step inputs for the requested "
                "RBF configuration search."
            )

        selected_configuration, cv_scores = _select_configuration_loo(
            inputs=inputs,
            outputs=outputs,
            hidden_candidates=hidden_candidates,
            fixed_width_neighbors=fixed_q,
        )

        selected_hidden, selected_q = selected_configuration

        centers, widths, weights = _fit_rbf(
            inputs=inputs,
            outputs=outputs,
            n_hidden=selected_hidden,
            width_neighbors=selected_q,
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

        # Compatibility / compact summaries.
        cv_by_hidden: dict[int, float] = {}

        for (n_hidden, q), score in cv_scores.items():
            if n_hidden not in cv_by_hidden or score < cv_by_hidden[n_hidden]:
                cv_by_hidden[n_hidden] = score

        candidate_configurations = sorted(cv_scores.keys())

        return {
            "lambda_hat": lambda_hat,
            "selected_hidden_nodes": selected_hidden,
            "selected_width_neighbors": selected_q,
            "width_neighbors_used": selected_q,
            "candidate_hidden_nodes": hidden_candidates,
            "candidate_configurations": candidate_configurations,
            "cv_mse_by_configuration": cv_scores,
            "cv_mse_by_hidden_nodes": cv_by_hidden,
            "selected_cv_mse": cv_scores[selected_configuration],
            "training_mse": training_mse,
            "centers": centers,
            "widths": widths,
            "weights": weights,
            "local_derivatives": derivatives,
            "normalized_inputs": inputs,
            "normalized_outputs": outputs,
            "n_pairs": int(inputs.size),
            "normalization_center": center,
            "normalization_scale": scale,
            "scaling_mode": scaling_mode,
            "zero_derivative_count": zero_derivative_count,
            "selection_mode": (
                "joint_hidden_and_width_cv"
                if fixed_q is None
                else "hidden_cv_fixed_width_neighbors"
            ),
            "runtime_sec": time.perf_counter() - start_time,
            "valid": True,
            "failure_reason": None,
        }

    except Exception as exc:
        return {
            "lambda_hat": np.nan,
            "selected_hidden_nodes": None,
            "selected_width_neighbors": None,
            "width_neighbors_used": None,
            "candidate_hidden_nodes": [],
            "candidate_configurations": [],
            "cv_mse_by_configuration": {},
            "cv_mse_by_hidden_nodes": {},
            "selected_cv_mse": np.nan,
            "training_mse": np.nan,
            "centers": np.asarray([], dtype=float),
            "widths": np.asarray([], dtype=float),
            "weights": np.asarray([], dtype=float),
            "local_derivatives": np.asarray([], dtype=float),
            "normalized_inputs": np.asarray([], dtype=float),
            "normalized_outputs": np.asarray([], dtype=float),
            "n_pairs": 0,
            "normalization_center": None,
            "normalization_scale": None,
            "scaling_mode": None,
            "zero_derivative_count": 0,
            "selection_mode": None,
            "runtime_sec": time.perf_counter() - start_time,
            "valid": False,
            "failure_reason": str(exc),
        }
