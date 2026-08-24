from mpmath import mp

from src.estimators.rosenstein import (
    rosenstein_le_estimator,
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


def affine_trajectory(a):
    a = mp.mpf(a)

    x = [X0]

    # x_0,...,x_N,x_{N+1}
    for _ in range(N + 1):
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

    for _ in range(N + 1):
        e = x[-1] - L

        x.append(
            L
            + a * e
            + beta * e * e
        )

    return x


def run_window(x, K):
    # Same K+2 observed values supplied
    # to the other baselines:
    #
    # x_{N-K},...,x_N,x_{N+1}.
    data = x[
        N - K:
        N + 2
    ]

    return rosenstein_le_estimator(
        x=data,
        embedding_dim=1,
        lag=1,
        temporal_exclusion=0,
        sample_time=mp.mpf("1.0"),
        max_horizon=None,
        numeric_backend="mpmath",
    )


def main():
    mp.dps = MP_DPS

    print(
        "\n=== ROSENSTEIN WINDOWED "
        "AFFINE VALIDATION ==="
    )

    for a_text in A_VALUES:
        a = mp.mpf(a_text)

        truth = mp.log(
            mp.fabs(a)
        )

        x = affine_trajectory(a)

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

            counts = (
                result["pair_counts"]
            )

            first_count = (
                counts[0]
                if len(counts) > 0
                else 0
            )

            last_count = (
                counts[-1]
                if len(counts) > 0
                else 0
            )

            print(
                f"a={float(a):+.2f} "
                f"K={K:2d} "
                f"truth="
                f"{float(truth):+.8f} "
                f"ROS="
                f"{float(result['lambda_hat']):+.8f} "
                f"error="
                f"{float(error):.3e} "
                f"pairs="
                f"{result['n_neighbor_pairs']} "
                f"horizons="
                f"{len(result['horizons'])} "
                f"count0="
                f"{first_count} "
                f"countLast="
                f"{last_count} "
                f"valid="
                f"{result['valid']}"
            )

    print(
        "\n=== ROSENSTEIN WINDOWED "
        "NONLINEAR VALIDATION ==="
    )

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
            f"ROS="
            f"{float(result['lambda_hat']):+.8f} "
            f"error="
            f"{float(error):.3e} "
            f"pairs="
            f"{result['n_neighbor_pairs']} "
            f"horizons="
            f"{len(result['horizons'])} "
            f"valid="
            f"{result['valid']}"
        )


if __name__ == "__main__":
    main()
