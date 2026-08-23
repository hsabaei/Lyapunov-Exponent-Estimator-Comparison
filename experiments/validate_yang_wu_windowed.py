from mpmath import mp

from src.estimators.yang_wu import (
    yang_wu_scalar_le_estimator,
)


MP_DPS = 200

N = 200

L = mp.mpf("2.0")

X0 = mp.mpf(
    "1.3578696273508724"
)

A_VALUES = [
    "-0.95",
    "-0.8",
    "-0.6",
    "-0.4",
    "-0.2",
    "0.2",
    "0.4",
    "0.6",
    "0.8",
    "0.95",
]

K_VALUES = [
    10,
    20,
    40,
    50,
]


def affine_trajectory(
    a,
):
    a = mp.mpf(a)

    x = [X0]

    # x_0,...,x_N,x_{N+1}
    for _ in range(
        N + 1
    ):
        x.append(
            L
            + a
            * (x[-1] - L)
        )

    return x


def nonlinear_trajectory(
    a,
    beta,
):
    a = mp.mpf(a)
    beta = mp.mpf(beta)

    x = [X0]

    for _ in range(
        N + 1
    ):
        e = x[-1] - L

        x.append(
            L
            + a * e
            + beta * e * e
        )

    return x


def run_window(
    x,
    K,
):
    # Proposed estimator at final
    # endpoint n=N uses:
    #
    # x_{N-K},...,x_N
    # plus x_{N+1}.
    #
    # Give Yang-Wu the same
    # observation record.
    data = x[
        N - K:
        N + 2
    ]

    return (
        yang_wu_scalar_le_estimator(
            x=data,
            evolution=1,
            r_min=None,
            numeric_backend="mpmath",
        )
    )


def main():
    mp.dps = MP_DPS

    print(
        "\n=== HIGH-PRECISION "
        "WINDOWED AFFINE VALIDATION ==="
    )

    for a_text in A_VALUES:
        a = mp.mpf(
            a_text
        )

        truth = mp.log(
            mp.fabs(a)
        )

        x = affine_trajectory(
            a
        )

        for K in K_VALUES:
            result = run_window(
                x=x,
                K=K,
            )

            error = (
                mp.fabs(
                    result["lambda_hat"]
                    - truth
                )
                if result["valid"]
                else mp.nan
            )

            print(
                f"a={float(a):+.2f} "
                f"K={K:2d} "
                f"truth="
                f"{float(truth):+.8f} "
                f"YW="
                f"{float(result['lambda_hat']):+.8f} "
                f"error="
                f"{float(error):.3e} "
                f"valid_maps="
                f"{result['n_valid_local_maps']}/"
                f"{result['n_attempted_local_maps']} "
                f"valid="
                f"{result['valid']}"
            )

    print(
        "\n=== HIGH-PRECISION "
        "WINDOWED NONLINEAR VALIDATION ==="
    )

    # Same representative
    # Experiment-2 case.
    a = mp.mpf("0.8")
    eta = mp.mpf("0.9")

    d = mp.fabs(
        X0 - L
    )

    beta = (
        eta
        * (
            1
            - mp.fabs(a)
        )
        / (
            2 * d
        )
    )

    x = nonlinear_trajectory(
        a=a,
        beta=beta,
    )

    truth = mp.log(
        mp.fabs(a)
    )

    print(
        f"a={float(a):+.2f} "
        f"eta={float(eta):.1f} "
        f"beta={float(beta):+.8f}"
    )

    for K in K_VALUES:
        result = run_window(
            x=x,
            K=K,
        )

        error = (
            mp.fabs(
                result["lambda_hat"]
                - truth
            )
            if result["valid"]
            else mp.nan
        )

        print(
            f"K={K:2d} "
            f"truth="
            f"{float(truth):+.8f} "
            f"YW="
            f"{float(result['lambda_hat']):+.8f} "
            f"error="
            f"{float(error):.3e} "
            f"valid_maps="
            f"{result['n_valid_local_maps']}/"
            f"{result['n_attempted_local_maps']} "
            f"valid="
            f"{result['valid']}"
        )


if __name__ == "__main__":
    main()
