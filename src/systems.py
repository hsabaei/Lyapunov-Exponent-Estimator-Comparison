import numpy as np
import mpmath as mp


# ============================================================
# Utility
# ============================================================

def _to_mpf(value):
    if isinstance(value, mp.mpf):
        return value

    return mp.mpf(str(value))


# ============================================================
# Linear fixed-point system
# ============================================================

def linear_map(x, a, L):
    """
    Linear fixed-point map

        U(x) = L + a(x - L)

    with fixed point L.
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

    according to

        x_{t+1} = U(x_t),

    where

        U(x) = L + a(x - L).
    """

    if not (0.0 < abs(float(a)) < 1.0):
        raise ValueError(
            "Requires 0 < |a| < 1."
        )

    if N < 1:
        raise ValueError(
            "N must be at least 1."
        )

    # --------------------------------------------------------
    # Arbitrary precision
    # --------------------------------------------------------

    if high_precision:

        a_mp = _to_mpf(a)
        L_mp = _to_mpf(L)
        x0_mp = _to_mpf(x0)

        x = [
            mp.mpf("0")
            for _ in range(N + 2)
        ]

        x[0] = x0_mp

        for t in range(N + 1):

            x[t + 1] = linear_map(
                x=x[t],
                a=a_mp,
                L=L_mp,
            )

        return x

    # --------------------------------------------------------
    # float64
    # --------------------------------------------------------

    x = np.empty(
        N + 2,
        dtype=float,
    )

    x[0] = float(x0)

    for t in range(N + 1):

        x[t + 1] = linear_map(
            x=x[t],
            a=float(a),
            L=float(L),
        )

    return x


def linear_true_lyapunov(
    a,
    high_precision=False,
):
    """
    Exact asymptotic LE:

        lambda_infinity = log(|a|).
    """

    if not (0.0 < abs(float(a)) < 1.0):
        raise ValueError(
            "Requires 0 < |a| < 1."
        )

    if high_precision:

        return mp.log(
            mp.fabs(_to_mpf(a))
        )

    return float(
        np.log(abs(float(a)))
    )


# ============================================================
# Nonlinear fixed-point system
# ============================================================

def nonlinear_map(
    x,
    a,
    beta,
    L,
):
    """
    Nonlinear fixed-point map

        U(x)
        =
        L
        + a (x - L)
        + beta (x - L)^2.

    Since

        U(L) = L,

    L is a fixed point.
    """

    error = x - L

    return (
        L
        + a * error
        + beta * error**2
    )


def nonlinear_map_derivative(
    x,
    a,
    beta,
    L,
):
    """
    Derivative of the nonlinear map:

        U'(x)
        =
        a + 2 beta (x - L).

    Therefore

        U'(L) = a.
    """

    return (
        a
        + 2 * beta * (x - L)
    )


def generate_nonlinear_trajectory(
    a,
    beta,
    L,
    x0,
    N,
    high_precision=False,
):
    """
    Generate

        x_0, ..., x_N, x_{N+1}

    according to

        x_{t+1} = U(x_t),

    where

        U(x)
        =
        L
        + a(x-L)
        + beta(x-L)^2.
    """

    if not (0.0 < abs(float(a)) < 1.0):
        raise ValueError(
            "Requires 0 < |a| < 1."
        )

    if N < 1:
        raise ValueError(
            "N must be at least 1."
        )

    # --------------------------------------------------------
    # Arbitrary precision
    # --------------------------------------------------------

    if high_precision:

        a_mp = _to_mpf(a)
        beta_mp = _to_mpf(beta)
        L_mp = _to_mpf(L)
        x0_mp = _to_mpf(x0)

        x = [
            mp.mpf("0")
            for _ in range(N + 2)
        ]

        x[0] = x0_mp

        for t in range(N + 1):

            x[t + 1] = nonlinear_map(
                x=x[t],
                a=a_mp,
                beta=beta_mp,
                L=L_mp,
            )

        return x

    # --------------------------------------------------------
    # float64
    # --------------------------------------------------------

    x = np.empty(
        N + 2,
        dtype=float,
    )

    x[0] = float(x0)

    for t in range(N + 1):

        x[t + 1] = nonlinear_map(
            x=x[t],
            a=float(a),
            beta=float(beta),
            L=float(L),
        )

    return x


def nonlinear_true_lyapunov(
    a,
    high_precision=False,
):
    """
    Exact asymptotic Lyapunov exponent.

    Since

        U'(L) = a,

    then

        lambda_infinity = log(|a|).
    """

    if not (0.0 < abs(float(a)) < 1.0):
        raise ValueError(
            "Requires 0 < |a| < 1."
        )

    if high_precision:

        return mp.log(
            mp.fabs(_to_mpf(a))
        )

    return float(
        np.log(abs(float(a)))
    )


def nonlinear_finite_window_lyapunov(
    x,
    a,
    beta,
    L,
    K,
    end_index,
    high_precision=False,
):
    """
    Exact finite-window tangent quantity:

        lambda_FT
        =
        (1/K)
        sum_{t=n-K}^{n-1}
        log |U'(x_t)|.

    This is an ORACLE diagnostic because it uses
    the known derivative of U.

    It is not a competing estimator.
    """

    n = int(end_index)

    if n < K:
        raise ValueError(
            "end_index must be >= K."
        )

    if high_precision:

        a_mp = _to_mpf(a)
        beta_mp = _to_mpf(beta)
        L_mp = _to_mpf(L)

        values = []

        for t in range(
            n - K,
            n,
        ):

            derivative = (
                nonlinear_map_derivative(
                    x=x[t],
                    a=a_mp,
                    beta=beta_mp,
                    L=L_mp,
                )
            )

            magnitude = mp.fabs(
                derivative
            )

            if magnitude == 0:
                return mp.ninf

            values.append(
                mp.log(magnitude)
            )

        return (
            mp.fsum(values)
            / mp.mpf(K)
        )

    values = []

    for t in range(
        n - K,
        n,
    ):

        derivative = (
            nonlinear_map_derivative(
                x=x[t],
                a=float(a),
                beta=float(beta),
                L=float(L),
            )
        )

        magnitude = abs(
            derivative
        )

        if magnitude == 0.0:
            return -np.inf

        values.append(
            np.log(magnitude)
        )

    return float(
        np.mean(values)
    )