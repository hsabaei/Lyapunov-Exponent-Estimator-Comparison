import numpy as np
import mpmath as mp


def linear_map(x, a, L):
    """
    Linear fixed-point map

        U(x) = L + a(x - L)

    so that

        x_{t+1} = U(x_t)

    and L is the fixed point.
    """
    return L + a * (x - L)


def generate_linear_trajectory(
    a,
    L,
    x0,
    N,
    high_precision=False,
):
    """
    Generate

        x_0, ..., x_N, x_{N+1}

    from

        x_{t+1} = U(x_t),

    where

        U(x) = L + a(x - L).

    Parameters
    ----------
    a : float
        Linear contraction factor.

    L : float
        Fixed point.

    x0 : float
        Initial condition.

    N : int
        Returned trajectory contains
        x_0, ..., x_N, x_{N+1}.

    high_precision : bool
        If True, use mpmath arbitrary-precision arithmetic.
    """

    if not (0.0 < abs(float(a)) < 1.0):
        raise ValueError(
            "Linear benchmark requires 0 < |a| < 1."
        )

    if N < 1:
        raise ValueError(
            "N must be at least 1."
        )

    # ======================================================
    # High-precision trajectory
    # ======================================================

    if high_precision:

        a_mp = mp.mpf(str(a))
        L_mp = mp.mpf(str(L))
        x0_mp = mp.mpf(str(x0))

        x = [mp.mpf("0") for _ in range(N + 2)]

        x[0] = x0_mp

        for t in range(N + 1):

            x[t + 1] = linear_map(
                x=x[t],
                a=a_mp,
                L=L_mp,
            )

        return x

    # ======================================================
    # Standard float64 trajectory
    # ======================================================

    x = np.empty(
        N + 2,
        dtype=float,
    )

    x[0] = x0

    for t in range(N + 1):

        x[t + 1] = linear_map(
            x=x[t],
            a=a,
            L=L,
        )

    return x


def linear_true_lyapunov(
    a,
    high_precision=False,
):
    """
    Exact asymptotic Lyapunov exponent.

    Since

        U'(x) = a,

    we have

        lambda = log(|a|).
    """

    if not (0.0 < abs(float(a)) < 1.0):
        raise ValueError(
            "Requires 0 < |a| < 1."
        )

    if high_precision:

        a_mp = mp.mpf(str(a))

        return mp.log(
            mp.fabs(a_mp)
        )

    return float(
        np.log(abs(a))
    )