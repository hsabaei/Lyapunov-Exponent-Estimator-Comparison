import numpy as np


def generate_linear_trajectory(a, L, x0, N):
    """
    Generate x_0, ..., x_N, x_{N+1} for

        x_{t+1} = L + a (x_t - L),

    with 0 < |a| < 1.

    Parameters
    ----------
    a : float
        Linear contraction factor.
    L : float
        Fixed point.
    x0 : float
        Initial condition.
    N : int
        The returned trajectory contains N + 2 observations:
        x_0, ..., x_N, x_{N+1}.

    Returns
    -------
    x : np.ndarray
        Array of shape (N + 2,).
    """

    if not (0.0 < abs(a) < 1.0):
        raise ValueError("Linear benchmark requires 0 < |a| < 1.")

    if N < 1:
        raise ValueError("N must be at least 1.")

    x = np.empty(N + 2, dtype=float)
    x[0] = x0

    for t in range(N + 1):
        x[t + 1] = L + a * (x[t] - L)

    return x


def linear_true_lyapunov(a):
    """
    Exact asymptotic Lyapunov exponent for the linear benchmark.
    """
    if not (0.0 < abs(a) < 1.0):
        raise ValueError("Requires 0 < |a| < 1.")

    return float(np.log(abs(a)))