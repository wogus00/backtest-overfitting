from numbers import Real

import numpy as np


def generate_gaussian_returns(T=1000, N=100, rho=0.2, mu=0.0, seed=42):
    """Generate X[t,n] = mu[n] + sqrt(rho)*f[t] + sqrt(1-rho)*e[t,n].

    Independent standard-normal factors and shocks give iid rows, unit
    population variance per column, and pairwise correlation rho in [0, 1].
    mu is a scalar or length-N vector of nonannualized population Sharpes.
    No sample demeaning or rescaling is applied. The output is a T x N array
    of volatility-normalized observations, not percentage returns or prices.
    """
    for name, value in (("T", T), ("N", N)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be a positive integer.")
    if (isinstance(rho, (bool, np.bool_)) or not isinstance(rho, Real)
            or not np.isfinite(rho) or not 0 <= rho <= 1):
        raise ValueError("rho must be a finite number in [0, 1].")
    if np.iscomplexobj(mu):
        raise ValueError("mu must be real.")
    mu = np.asarray(mu, dtype=float)
    if mu.shape not in ((), (N,)) or not np.isfinite(mu).all():
        raise ValueError("mu must be a finite scalar or a finite length-N vector.")
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((T, 1))
    shocks = rng.standard_normal((T, N))
    with np.errstate(over="raise", invalid="raise"):
        return mu + np.sqrt(rho) * factor + np.sqrt(1 - rho) * shocks
