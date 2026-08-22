import time
import numpy as np


def proposed_le_estimator(x, K):
    """
    Proposed Lyapunov-exponent estimator based on

        lambda_hat
        =
        ell_err_hat
        +
        log(|ID_F_star_hat|),

    where ID_F_star_hat is estimated using FIE.

    Parameters
    ----------
    x : array_like
        Observed trajectory

            x_0, ..., x_N, x_{N+1}.

        Therefore len(x) = N + 2.

    K : int
        Tail length.

    Returns
    -------
    result : dict
        Dictionary containing

        lambda_hat
        ell_err_hat
        id_fie_hat
        L_hat
        t0
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

    # x = x_0, ..., x_N, x_{N+1}
    N = len(x) - 2

    # Common tail:
    # T = {t0, ..., N-1}
    # K = N - t0
    t0 = N - K

    # FIE also requires x_{t0-1}
    if t0 < 1:
        raise ValueError(
            f"K={K} is too large for N={N}. "
            "FIE requires t0 >= 1."
        )

    # Practical limit estimate
    L_hat = x[N + 1]

    # Numerical safeguard
    scale = max(1.0, float(np.max(np.abs(x))))
    tol = 100.0 * np.finfo(float).eps * scale

    def failure(reason, ell_err_hat=np.nan, id_fie_hat=np.nan):
        return {
            "lambda_hat": np.nan,
            "ell_err_hat": ell_err_hat,
            "id_fie_hat": id_fie_hat,
            "L_hat": float(L_hat),
            "t0": int(t0),
            "K": int(K),
            "runtime_sec": time.perf_counter() - start_time,
            "valid": False,
            "failure_reason": reason,
        }

    # ==================================================
    # 1. Error-decay component
    # ==================================================

    r0 = abs(x[t0] - L_hat)

    if (not np.isfinite(r0)) or r0 <= tol:
        return failure(
            "Initial tail distance from L_hat is numerically zero."
        )

    ell_err_values = []

    for j in range(1, K + 1):

        rj = abs(x[t0 + j] - L_hat)

        if (not np.isfinite(rj)) or rj <= tol:
            return failure(
                f"Tail distance is numerically invalid at j={j}."
            )

        ell_j = np.log(rj / r0) / j

        if not np.isfinite(ell_j):
            return failure(
                f"Error-decay estimate is nonfinite at j={j}."
            )

        ell_err_values.append(ell_j)

    ell_err_hat = float(np.mean(ell_err_values))

    # ==================================================
    # 2. FIE estimate of limiting LID
    # ==================================================

    numerator = 0.0
    denominator = 0.0

    numerator_base = abs(x[t0] - L_hat)
    denominator_base = abs(x[t0 - 1] - L_hat)

    if numerator_base <= tol:
        return failure(
            "FIE numerator base is numerically zero.",
            ell_err_hat=ell_err_hat,
        )

    if denominator_base <= tol:
        return failure(
            "FIE denominator base is numerically zero.",
            ell_err_hat=ell_err_hat,
        )

    for i in range(1, K + 1):

        numerator_distance = abs(
            x[t0 + i] - L_hat
        )

        denominator_distance = abs(
            x[t0 - 1 + i] - L_hat
        )

        if numerator_distance <= tol:
            return failure(
                f"FIE numerator distance invalid at i={i}.",
                ell_err_hat=ell_err_hat,
            )

        if denominator_distance <= tol:
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

    denominator_tol = (
        100.0
        * np.finfo(float).eps
        * max(1.0, abs(numerator), abs(denominator))
    )

    if abs(denominator) <= denominator_tol:
        return failure(
            "FIE denominator is numerically zero.",
            ell_err_hat=ell_err_hat,
        )

    id_fie_hat = float(numerator / denominator)

    if not np.isfinite(id_fie_hat):
        return failure(
            "FIE estimate is nonfinite.",
            ell_err_hat=ell_err_hat,
            id_fie_hat=id_fie_hat,
        )

    if abs(id_fie_hat) <= np.finfo(float).tiny:
        return failure(
            "FIE estimate is numerically zero.",
            ell_err_hat=ell_err_hat,
            id_fie_hat=id_fie_hat,
        )

    # ==================================================
    # 3. Final Lyapunov estimate
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

    runtime = time.perf_counter() - start_time

    return {
        "lambda_hat": float(lambda_hat),
        "ell_err_hat": float(ell_err_hat),
        "id_fie_hat": float(id_fie_hat),
        "L_hat": float(L_hat),
        "t0": int(t0),
        "K": int(K),
        "runtime_sec": float(runtime),
        "valid": True,
        "failure_reason": "",
    }