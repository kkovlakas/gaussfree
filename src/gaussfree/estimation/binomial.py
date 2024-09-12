"""Estimate the success probability in a binomial distribution."""


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# This avoids using the deprecated np.trapz for newer versions of numpy
try:
    from numpy import trapezoid as np_trapezoid
except ImportError:
    from numpy import trapz as np_trapezoid


PI = 3.14


class EstimateBinomialP:
    """Estimator for the success probability in a binomial distribution."""
    def __init__(self, n_trials, n_success, prior_dist=None,
                 n_points=1001, hpd_area=0.68):
        """Initialize the binomail probability estimator.

        Parameters
        ----------
        n_trials : int
            Number of trial in the binomial experiment.
        n_success : int
            Number of successful trials in the binomial experiment.
        prior_dist : None, or class containing `.pdf()` method., optional
            If None, a uniform prior will be used.
        n_points : int, optional
            Number of grid points for numerical oprations, default: 1001.
        hpd_area : float, optional
            _description_, by default 0.68

        """
        if not isinstance(n_trials, (int, np.integer)) or n_trials < 0:
            raise ValueError("`n_trials` must be a non-negative integer.")

        if not (isinstance(n_success, (int, np.integer))
                and 0 <= n_success <= n_trials):
            raise ValueError("`n_success` must between 0 and `n_trials`.")

        if not isinstance(n_points, (int, np.integer)) or n_points < 100:
            raise ValueError("`n_points` must be >= 100.")

        if not 0 < hpd_area < 1:
            raise ValueError("`hpd_area` must be in range (0, 1).")

        if prior_dist is None:
            prior_dist = st.uniform(0, 1)
        if not hasattr(prior_dist, "pdf"):
            raise ValueError("`prior_dist` must have a `.pdf()` method.")

        self._initialized = False
        self.n = n_trials
        self.k = n_success
        self.prior_dist = prior_dist
        self.n_points = n_points
        self.hpd_area = hpd_area
        self.mode = np.nan
        self.median = np.nan
        self.mean = np.nan
        self.variance = np.nan
        self.std = np.nan
        self.hpdi = (np.nan, np.nan)
        self.err_hi = np.nan
        self.err_lo = np.nan
        self.prior = None
        self.likelihood = None
        self.posterior = None

        self.compute_probabilities()
        self.compute_statistics()
        self._initialized = True

    def __setattr__(self, name, value):
        """Override the `__setattr__` method to make the object immutable."""
        if hasattr(self, "_initialized") and self._initialized:
            raise AttributeError("The object is immutable. You are not "
                                 f"allowed to set {name}={value}.")
        super().__setattr__(name, value)

    def compute_probabilities(self):
        """Compute the likelihood, prior, and posterior probabilities."""
        def norm_prob(prob):
            """Normalize a probability defined on the support `self.x`."""
            return prob / np_trapezoid(prob, self.x)

        self.x = np.linspace(0.0, 1.0, self.n_points)
        self.likelihood = norm_prob(st.binom.pmf(self.k, self.n, self.x))
        self.prior = norm_prob(self.prior_dist.pdf(self.x))
        self.posterior = norm_prob(self.likelihood * self.prior)

    def report(self):
        """Print statistics and plot the prior, likelihood, and posterior."""
        print(f"(n, k) = ({self.n}, {self.k})")
        print(f"# of support points: {self.n_points}")
        print("Statistics:")
        for statistic in ["mode", "median", "mean", "variance", "std"]:
            value = getattr(self, statistic)
            print(f"    {statistic:10s}: {value:.6f}")
        print(f"    {'HPDI':10s}: [{self.hpdi[0]:.6f}, {self.hpdi[1]:.6f}]")
        print(f"    {'errorbars':10s}: [{self.err_lo:.6f}, {self.err_hi:.6f}]")

        plt.figure()
        plt.plot(self.x, self.prior,
                 color="r", ls="-", alpha=0.5, label="Prior")
        plt.plot(self.x, self.likelihood,
                 color="g", ls="-", label="Likelihood")
        plt.plot(self.x, self.posterior,
                 color="b", ls="--", label="Posterior")
        hpdi_range = (self.x >= self.hpdi[0]) & (self.x <= self.hpdi[1])
        plt.fill_between(self.x[hpdi_range], 0.0, self.posterior[hpdi_range],
                         fc="b", alpha=0.5, ec="none", label="HPDI")
        plt.axhline(self.posterior[np.argmax(hpdi_range)],
                    color="0.5", ls=":", label="HPDI level")
        plt.xlim(0.0, 1.0)
        plt.legend(loc="best")
        plt.show()

    def get_hpdi(self):
        """Get the highest posterior density interval."""
        n = len(self.x)
        posterior_area = self.hpd_area
        y = self.posterior / np.sum(self.posterior)
        mode = np.argmax(y)
        area = y[mode]
        left, right = mode, mode
        while area < posterior_area and (left > 0 or right < n - 1):
            if left == 0 or (right != n - 1 and (y[right+1] > y[left-1])):
                right += 1
                area += y[right]
            else:
                left -= 1
                area += y[left]
        return self.x[left], self.x[right]

    def compute_statistics(self):
        """Compute statistics based on posterior probability."""
        x = self.x
        y = self.posterior
        self.mode = x[np.argmax(y)]

        cdf = np.cumsum(y)
        cdf -= cdf[0]
        cdf /= cdf[-1]
        self.median = x[np.argmin(cdf - 0.5)]

        norm = np.sum(y)
        self.mean = np.sum(y * x) / norm
        self.variance = np.sum(y * (x - self.mean)**2.0) / norm
        self.std = self.variance ** 0.5
        self.hpdi = self.get_hpdi()

        lo, hi = self.hpdi
        self.err_hi = hi - self.mode
        self.err_lo = self.mode - lo
