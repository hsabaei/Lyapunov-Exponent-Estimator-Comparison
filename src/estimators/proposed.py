import time
import numpy as np


def proposed_le_estimator(x, K, end_index=None):
    """
    Rolling proposed Lyapunov-exponent estimator.

    At estimation endpoint n, the estimator uses

        boundary:       x_{n-K-1}
        K-step window:  x_{n-K}, ..., x_n
        limit estimate: L_hat = x_{n+1}

    The K-step window contains K iterations and K+1 states.

    Parameters
    ----------
    x : array_like
        Observed scalar trajectory.

    K : int
        Number of iterations in the estimation window.

    end_index : int or None
        Index n of the final state in the K-step window.

        If None, the latest possible estimate is returned:
            n = len(x) - 2,
        so that x_{n+1} is available as L_hat.

    Returns
    -------
    result : dict
        lambda_hat
        ell_err_hat
        id_fie_hat
        L_hat
        end_index
        window_start
        window_end
        boundary_index
        L_hat_index
        K
        runtime_sec
        valid
        failure_reason
    """

    start_time = time.perf_counter()

    x = np.asarray(x, dtype=float)

    # --------------------------------------------------
    # Input checks
    # --------------------------------------------------
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if len(x) < 4:
        raise ValueError("At least four observations are required.")

    if K < 1:
        raise ValueError("K must be positive.")

    # Latest possible endpoint if not specified
    if end_index is None:
        n = len(x) - 2
    else:
        n = int(end_index)

    # Need x_{n+1} for L_hat
    if n + 1 >= len(x):
        raise ValueError(
            f"x_[n+1] is unavailable for end_index={n}."
        )

    # Start of K-step estimation window
    t0 = n - K

    # FIE requires one observation before the window
    if t0 < 1:
        raise ValueError(
            "FIE requires one observation before the K-step window. "
            f"For K={K}, end_index must satisfy end_index >= {K + 1}."
        )

    boundary_index = t0 - 1
    L_hat_index = n + 1

    # Each rolling estimate uses its own next observation
    L_hat = x[L_hat_index]

    tiny = np.finfo(float).tiny

    def failure(
        reason,
        ell_err_hat=np.nan,
        id_fie_hat=np.nan,
    ):
        return {
            "lambda_hat": np.nan,
            "ell_err_hat": ell_err_hat,
            "id_fie_hat": id_fie_hat,
            "L_hat": float(L_hat),
            "end_index": int(n),
            "window_start": int(t0),
            "window_end": int(n),
            "boundary_index": int(boundary_index),
            "L_hat_index": int(L_hat_index),
            "K": int(K),
            "runtime_sec": time.perf_counter() - start_time,
            "valid": False,
            "failure_reason": reason,
        }

    # ==================================================
    # 1. Error-decay component
    # ==================================================
    #
    # ell_err_hat
    # =
    # (1/K) sum_{j=1}^K
    # (1/j) log(
    #   |x_{t0+j} - L_hat|
    #   /
    #   |x_{t0} - L_hat|
    # )
    # ==================================================

    r0 = abs(x[t0] - L_hat)

    if (not np.isfinite(r0)) or r0 <= tiny:
        return failure(
            "Initial window distance from L_hat is zero or nonfinite."
        )

    ell_values = []

    for j in range(1, K + 1):

        rj = abs(x[t0 + j] - L_hat)

        if (not np.isfinite(rj)) or rj <= tiny:
            return failure(
                f"Window distance is zero or nonfinite at j={j}."
            )

        ell_j = np.log(rj / r0) / j

        if not np.isfinite(ell_j):
            return failure(
                f"Error-decay term is nonfinite at j={j}."
            )

        ell_values.append(ell_j)

    ell_err_hat = float(np.mean(ell_values))

    # ==================================================
    # 2. FIE estimate
    # ==================================================
    #
    # numerator:
    #
    # sum_{i=1}^K log |
    #   (x_{t0+i} - L_hat)
    #   /
    #   (x_{t0} - L_hat)
    # |
    #
    # denominator:
    #
    # sum_{i=1}^K log |
    #   (x_{t0-1+i} - L_hat)
    #   /
    #   (x_{t0-1} - L_hat)
    # |
    #
    # ==================================================

    numerator_base = abs(
        x[t0] - L_hat
    )

    denominator_base = abs(
        x[t0 - 1] - L_hat
    )

    if (
        not np.isfinite(numerator_base)
        or numerator_base <= tiny
    ):
        return failure(
            "FIE numerator base is zero or nonfinite.",
            ell_err_hat=ell_err_hat,
        )

    if (
        not np.isfinite(denominator_base)
        or denominator_base <= tiny
    ):
        return failure(
            "FIE denominator base is zero or nonfinite.",
            ell_err_hat=ell_err_hat,
        )

    numerator = 0.0
    denominator = 0.0

    for i in range(1, K + 1):

        numerator_distance = abs(
            x[t0 + i] - L_hat
        )

        denominator_distance = abs(
            x[t0 - 1 + i] - L_hat
        )

        if (
            not np.isfinite(numerator_distance)
            or numerator_distance <= tiny
        ):
            return failure(
                f"FIE numerator distance invalid at i={i}.",
                ell_err_hat=ell_err_hat,
            )

        if (
            not np.isfinite(denominator_distance)
            or denominator_distance <= tiny
        ):
            return failure(
                f"FIE denominator distance invalid at i={i}.",
                ell_err_hat=ell_err_hat,
            )

        numerator += np.log(
            numerator_distance / numerator_base
        )

        denominator += np.log(
            denominator_distance / denominator_base
        )

    if not np.isfinite(numerator):
        return failure(
            "FIE numerator is nonfinite.",
            ell_err_hat=ell_err_hat,
        )

    if not np.isfinite(denominator):
        return failure(
            "FIE denominator is nonfinite.",
            ell_err_hat=ell_err_hat,
        )

    if denominator == 0.0:
        return failure(
            "FIE denominator is zero.",
            ell_err_hat=ell_err_hat,
        )

    id_fie_hat = float(
        numerator / denominator
    )

    if (
        not np.isfinite(id_fie_hat)
        or id_fie_hat == 0.0
    ):
        return failure(
            "FIE estimate is zero or nonfinite.",
            ell_err_hat=ell_err_hat,
            id_fie_hat=id_fie_hat,
        )

    # ==================================================
    # 3. Final LE estimate
    # ==================================================

    lambda_hat = (
        ell_err_hat
        + np.log(abs(id_fie_hat))
    )

    if not np.isfinite(lambda_hat):
        return failure(
            "Final Lyapunov estimate is nonfinite.",
            ell_err_hat=ell_err_hat,
            id_fie_hat=id_fie_hat,
        )

    return {
        "lambda_hat": float(lambda_hat),
        "ell_err_hat": float(ell_err_hat),
        "id_fie_hat": float(id_fie_hat),
        "L_hat": float(L_hat),
        "end_index": int(n),
        "window_start": int(t0),
        "window_end": int(n),
        "boundary_index": int(boundary_index),
        "L_hat_index": int(L_hat_index),
        "K": int(K),
        "runtime_sec": float(
            time.perf_counter() - start_time
        ),
        "valid": True,
        "failure_reason": "",
    }


def rolling_proposed_le_estimator(x, K):
    """
    Compute every available rolling K-step LE estimate.

    The first possible estimate has

        boundary      = x_0
        window        = x_1, ..., x_{K+1}
        L_hat         = x_{K+2}

    because FIE requires one observation before the
    estimation window.

    The final possible endpoint is len(x)-2.
    """

    x = np.asarray(x, dtype=float)

    first_end = K + 1
    last_end = len(x) - 2

    if first_end > last_end:
        raise ValueError(
            "Trajectory is too short for the requested K."
        )

    results = []

    for n in range(first_end, last_end + 1):

        result = proposed_le_estimator(
            x=x,
            K=K,
            end_index=n,
        )

        results.append(result)

    return results