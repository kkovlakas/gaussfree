"""Module for estimating the success probability in a binomial distribution.

Author: Konstantinos Kovlakas
E-mail: kkovla@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import root


UNIMODAL_BETA_MAX_VAR = 1.0 / 12.0


def get_unimodal_beta(mode, variance, verbose=False):
    """Get a unimodal beta distribution with the given mode and variance.

    Parameters
    ----------
    mode : float
        The mode of the resulting beta distribution.
    variance : float
        The variance of the resulting beta distribution. Must be less than 1/12.
    verbose : bool
        If True, print verbose output and plot the prior distribution.

    Returns
    -------
    scipy.stats.beta
        The SciPy beta distribution object.

    """
    def verbose_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if variance >= UNIMODAL_BETA_MAX_VAR:
        raise ValueError(f"Not unimodal for variance = {variance:.6g} >= "
                         f"{UNIMODAL_BETA_MAX_VAR:.6g}.")

    verbose_print(f"Input pararms : mode={mode:.3g}, variance={variance:.3g}")

    def func(x):
        u, w = x
        a = 1 + np.exp(u)
        b = 1 + np.exp(w)
        new_mode = (a - 1.0) / (a + b - 2.0)
        new_variance = a * b / ((a + b)**2.0 * (a + b + 1.0))
        return [new_mode - mode, new_variance - variance]

    sol = root(func, [0, 0], method="lm")
    assert sol.success
    a, b = 1.0 + np.exp(sol.x)
    verbose_print(f"Beta params   : a={a:.6g}, b={b:.6g}")
    rec_mode = (a - 1.0) / (a + b - 2.0)
    rec_variance = a * b / ((a + b) ** 2.0 * (a + b + 1.0))
    verbose_print(
        f"Recovered par.: mode={rec_mode:.6g}, variance={rec_variance:.6g}")
    assert np.isclose(mode, rec_mode) and np.isclose(variance, rec_variance)

    dist = st.beta(a, b)
    if verbose:
        x = np.linspace(0.0, 1.0, 1001)
        plt.figure()
        plt.plot(x, dist.pdf(x), "k-")
        plt.xlim(0, 1)
        plt.show()
    return dist
