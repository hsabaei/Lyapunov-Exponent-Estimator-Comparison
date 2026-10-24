"""Scalar loss-LE validation during actual multi-parameter regression training.

Run: python experiment3_ml.py --help
The supplied estimators are imported unchanged from src/estimators/.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from src.estimators.proposed import proposed_le_estimator


def hadamard(n):
    if n < 2 or n & (n - 1):
        raise ValueError('samples must be a power of two >= 2')
    H = [[1]]
    while len(H) < n:
        H = [r + r for r in H] + [r + [-v for v in r] for r in H]
    return H


def train_regression(a, steps, seed, samples=16, parameters=8, sigma='0.3'):
    """Compute predictions, loss, and full-batch gradients on the dataset.

    No scalar recurrence is used to generate the observed loss.
    X consists of Hadamard columns: X.T X / n = I exactly.
    Label residual sigma * H[:, d] is orthogonal to X; loss limit = sigma²/2.
    Random signed column permutation changes the dataset across seeds.
    """
    if not 1 < parameters < samples:
        raise ValueError('require 1 < parameters < samples')
    a, sigma = mp.mpf(str(a)), mp.mpf(str(sigma))
    if not 0 < abs(a) < 1 or sigma < 0:
        raise ValueError('require 0 < |a| < 1 and sigma >= 0')
    rng = random.Random(seed)
    H = hadamard(samples)
    cols = list(range(samples))
    rng.shuffle(cols)
    signs = [rng.choice([-1, 1]) for _ in range(parameters)]
    X = mp.matrix([[signs[j] * H[i][cols[j]] for j in range(parameters)]
                   for i in range(samples)])
    wstar = mp.matrix([mp.mpf(rng.randint(-20, 20)) / 10 for _ in range(parameters)])
    offset = mp.matrix([mp.mpf(rng.choice([-9, -5, 3, 7])) / 10
                        for _ in range(parameters)])
    w = wstar + offset
    noise = mp.matrix([sigma * H[i][cols[parameters]] for i in range(samples)])
    y = X * wstar + noise
    learning_rate = 1 - a  # h=1
    loss_limit = sigma**2 / 2
    losses, history = [], []
    # steps+2 observations: z_0,...,z_steps,z_(steps+1).
    for t in range(steps + 2):
        residual = X * w - y
        loss = (residual.T * residual)[0] / (2 * samples)
        losses.append(loss)
        history.append([t, loss] + list(w))
        if t < steps + 1:
            gradient = X.T * residual / samples
            w = w - learning_rate * gradient
    # Reference calculations are isolated from estimator inputs.
    q = a*a
    initial_excess = losses[0] - loss_limit
    recurrence_errors = []
    for t, loss in enumerate(losses):
        expected = initial_excess * q**t
        recurrence_errors.append(abs((loss-loss_limit)-expected)/expected)
    orthogonality_error = max(abs(v) for v in (X.T*X/samples-mp.eye(parameters)))
    return losses, history, {
        'loss_limit': loss_limit, 'loss_le_true': 2*mp.log(abs(a)),
        'parameter_le_true': mp.log(abs(a)), 'loss_multiplier': q,
        'learning_rate': learning_rate,
        'max_relative_recurrence_error': max(recurrence_errors),
        'orthogonality_error': orthogonality_error,
    }


def estimate(data, K, baselines=False):
    """Only scalar observations enter this function; no reference quantities."""
    methods = [
        ('Proposed-WB', lambda: proposed_le_estimator(
            data, K=K, end_index=K, limit_method='next', numeric_backend='mpmath')),
        ('Proposed-Aitken', lambda: proposed_le_estimator(
            data, K=K, end_index=K, limit_method='aitken', numeric_backend='mpmath')),
    ]
    if baselines:
        from src.estimators.yang_wu import yang_wu_scalar_le_estimator
        from src.estimators.sun_wu import sun_wu_scalar_rbf_le_estimator
        from src.estimators.rosenstein import rosenstein_le_estimator
        methods += [
            ('Yang-Wu', lambda: yang_wu_scalar_le_estimator(
                data, evolution=1, r_min=None, numeric_backend='mpmath')),
            ('Sun-Wu', lambda: sun_wu_scalar_rbf_le_estimator(
                data, max_hidden_nodes=20, width_neighbors=None)),
            ('Rosenstein', lambda: rosenstein_le_estimator(
                data, embedding_dim=1, lag=1, temporal_exclusion=0,
                sample_time=mp.mpf(1), max_horizon=None, numeric_backend='mpmath')),
        ]
    for name, function in methods:
        started = time.perf_counter()
        try:
            result = function()
            value = result.get('lambda_hat', mp.nan)
            valid = bool(result.get('valid', False) and mp.isfinite(value))
            reason = result.get('failure_reason', '')
            if not valid and not reason:
                reason = 'Estimator returned an invalid or nonfinite result.'
        except Exception as error:
            result, value, valid = {}, mp.nan, False
            reason = f'{type(error).__name__}: {error}'
        yield name, result, value, valid, reason, time.perf_counter()-started


def text_value(value):
    if isinstance(value, mp.mpf):
        return mp.nstr(value, 55)
    return value


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: text_value(v) for k, v in row.items()} for row in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--a-values', nargs='+', default=['-0.95', '-0.6', '-0.2', '0.2', '0.6', '0.95'])
    p.add_argument('--windows', nargs='+', type=int, default=[10, 20, 40, 50])
    p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027])
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--parameters', type=int, default=8)
    p.add_argument('--sigma', default='0.3')
    p.add_argument('--dps', type=int, default=None, help='Default: automatic precision budget')
    p.add_argument('--baselines', action='store_true', help='Include three supplied baseline methods')
    p.add_argument('--rolling', action='store_true', help='Evaluate all windows, not only final windows')
    p.add_argument('--output', type=Path, default=Path('results/experiment3_ml'))
    args = p.parse_args()
    if args.steps < 3 or not args.seeds or min(args.windows) < 2 or max(args.windows) > args.steps:
        p.error('require steps >= 3 and 2 <= K <= steps')
    for a in args.a_values:
        if not 0 < abs(mp.mpf(a)) < 1:
            p.error('each a must satisfy 0 < |a| < 1')
    # Loss residuals decay as |a|^(2t); retain 80 guard digits above that decay.
    needed = int(max(-2*(args.steps+1)*mp.log10(abs(mp.mpf(a))) for a in args.a_values))+80
    mp.mp.dps = args.dps or max(120, needed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = [], []
    for seed in args.seeds:
        for a in args.a_values:
            losses, history, reference = train_regression(
                a, args.steps, seed, args.samples, args.parameters, args.sigma)
            diagnostics.append({'seed': seed, 'a': a, **reference})
            if reference['max_relative_recurrence_error'] > mp.mpf('1e-30'):
                raise ArithmeticError('Training recurrence lost precision; increase --dps.')
            write_csv(args.output/f'training_seed{seed}_a{a}.csv', [
                dict(zip(['epoch', 'loss']+[f'w{j}' for j in range(args.parameters)], h))
                for h in history])
            for K in args.windows:
                ends = range(K, args.steps+1) if args.rolling else [args.steps]
                for n in ends:
                    # Exactly K+2 scalar observations; same input for every estimator.
                    data = losses[n-K:n+2]
                    for name, result, value, valid, reason, elapsed in estimate(data, K, args.baselines):
                        truth = reference['loss_le_true']
                        rows.append({
                            'seed': seed, 'a': a, 'samples': args.samples,
                            'parameters': args.parameters, 'K': K, 'end_index': n,
                            'window_start': n-K, 'latest_observation_index': n+1,
                            'estimator': name, 'precision_dps': mp.mp.dps,
                            'loss_le_true': truth, 'parameter_le_true': reference['parameter_le_true'],
                            'lambda_hat': value, 'absolute_error': abs(value-truth) if valid else mp.nan,
                            'correct_sign': bool(valid and value < 0), 'valid': valid,
                            'failure_reason': reason, 'loss_limit_true': reference['loss_limit'],
                            'L_hat': result.get('L_hat', mp.nan),
                            'limit_absolute_error': abs(result['L_hat']-reference['loss_limit'])
                                if valid and 'L_hat' in result else mp.nan,
                            'ell_err_hat': result.get('ell_err_hat', mp.nan),
                            'id_fie_hat': result.get('id_fie_hat', mp.nan),
                            'runtime_sec': elapsed,
                        })
            print(f'Completed seed={seed}, a={a}; recurrence checked', flush=True)
    write_csv(args.output/'estimates.csv', rows)
    write_csv(args.output/'training_diagnostics.csv', diagnostics)
    summary = []
    final = [r for r in rows if r['end_index'] == args.steps]
    for K, method in sorted({(r['K'], r['estimator']) for r in final}):
        group = [r for r in final if r['K'] == K and r['estimator'] == method]
        valid = [r for r in group if r['valid']]
        errors = sorted(r['absolute_error'] for r in valid)
        count = len(errors)
        median = ((errors[(count-1)//2]+errors[count//2])/2) if count else mp.nan
        summary.append({'K': K, 'estimator': method, 'total': len(group), 'valid': count,
                        'invalid': len(group)-count,
                        'mean_absolute_error': mp.fsum(errors)/count if count else mp.nan,
                        'median_absolute_error': median, 'max_absolute_error': max(errors) if count else mp.nan,
                        'correct_sign_fraction_all_cases': sum(r['correct_sign'] for r in group)/len(group)})
    write_csv(args.output/'final_summary.csv', summary)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(actual_dps=mp.mp.dps, python=platform.python_version(),
                    numpy=np.__version__, mpmath=mp.__version__,
                    target='LE of exact autonomous scalar loss map, per gradient update',
                    timing='single-call elapsed seconds; not a controlled runtime benchmark')
    (args.output/'config.json').write_text(json.dumps(metadata, indent=2))
    for r in summary:
        print(f"K={r['K']:2} {r['estimator']:18} valid={r['valid']}/{r['total']} "
              f"MAE={mp.nstr(r['mean_absolute_error'], 7)}")


if __name__ == '__main__':
    main()
