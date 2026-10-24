"""Scientific checks of the dataset, training trajectory and scalar target."""
import mpmath as mp
from experiments.experiment3_ml import train_regression, estimate

mp.mp.dps = 140
for sigma in ['0', '0.3']:
    for a in ['-0.6', '0.6', '0.95']:
        losses, history, ref = train_regression(a, 40, 9, sigma=sigma)
        assert ref['orthogonality_error'] == 0
        assert ref['max_relative_recurrence_error'] < mp.mpf('1e-90')
        assert abs(ref['loss_le_true'] - 2*ref['parameter_le_true']) < mp.mpf('1e-100')
        data = losses[20:42]
        estimates = list(estimate(data, 20))
        assert all(item[3] for item in estimates)
        aitken = estimates[1]
        assert abs(aitken[2]-ref['loss_le_true']) < mp.mpf('1e-70')
        # Ensure multiple actual parameters change, rather than a one-weight model.
        assert all(history[0][j] != history[1][j] for j in range(2, 10))
# Equal parameter multiplier magnitudes imply equal loss histories for the same data.
lp, _, _ = train_regression('0.6', 40, 9)
ln, _, _ = train_regression('-0.6', 40, 9)
assert max(abs(x-y) for x, y in zip(lp, ln)) < mp.mpf('1e-120')
print('Passed: dataset geometry, explicit training recurrence, loss target, Aitken recovery, nonzero/zero loss limits, multiple parameter updates, sign symmetry.')
