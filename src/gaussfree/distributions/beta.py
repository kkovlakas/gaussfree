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
    def vprint(*args, **kwargs):
        """Verbose print function."""
        if verbose:
            print(*args, **kwargs)

    vprint(f"Input properties: mode={mode:.3g}, var={variance:.3g}")

    if not 0 < mode < 1:
        raise ValueError(f"Mode must be in (0, 1), got {mode:.6g}.")
    if variance <= 0:
        raise ValueError(f"Variance must be positive, got {variance:.6g}.")
    if variance >= UNIMODAL_BETA_MAX_VAR:
        raise ValueError(f"Not unimodal for variance greater than or equal to "
                         f"{UNIMODAL_BETA_MAX_VAR:.6g}, got {variance:.6g}.")

    def func(x):
        a, b = 1 + np.exp(x)
        new_mode = (a - 1.0) / (a + b - 2.0)
        new_variance = a * b / ((a + b)**2.0 * (a + b + 1.0))
        return [new_mode - mode, new_variance - variance]

    sol = root(func, [0, 0], method="lm")
    if not sol.success:
        raise RuntimeError("Failed to find the beta parameters. "
                           "Root finder returned:\n" + str(sol))

    a, b = 1.0 + np.exp(sol.x)
    vprint(f"Beta parameters : a={a:.6g}, b={b:.6g}")

    rec_mode = (a - 1.0) / (a + b - 2.0)
    rec_variance = a * b / ((a + b) ** 2.0 * (a + b + 1.0))
    vprint(f"Recovered stats : mode={rec_mode:.6g}, var={rec_variance:.6g}")
    if not np.isclose(mode, rec_mode) and np.isclose(variance, rec_variance):
        raise AssertionError("Failed to recover the input mode and variance."
                             "Run with verbose=True to see the details.")


    dist = st.beta(a, b)
    if verbose:
        x = np.linspace(0.0, 1.0, 1001)
        plt.figure()
        plt.plot(x, dist.pdf(x), "k-")
        plt.xlim(0, 1)
        plt.ylim(ymin=0)
        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.show()
    return dist
