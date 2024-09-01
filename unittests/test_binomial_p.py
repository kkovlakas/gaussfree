"""Unit tests for beta distribution functions."""


from unittest import TestCase
from gaussfree.estimation.binomial import EstimateBinomialP


class TestBinomialP(TestCase):
    """Unit tests for the `EstimateBinomialP` function."""

    def test_n_trials_parameter(self):
        """Test that the number of trials is correctly set."""
        self.assertRaises(ValueError, EstimateBinomialP, -1, 0)
        self.assertRaises(ValueError, EstimateBinomialP, "a", 0)

    def test_n_success_parameter(self):
        """Test that the number of successful trials is correctly set."""
        self.assertRaises(ValueError, EstimateBinomialP, 1, -1)
        self.assertRaises(ValueError, EstimateBinomialP, 1, "a")
        self.assertRaises(ValueError, EstimateBinomialP, 1, 2)

    def test_n_points_parameter(self):
        """Test that the number of grid points is correctly set."""
        self.assertRaises(ValueError, EstimateBinomialP, 1, 0, n_points=99)
        self.assertRaises(ValueError, EstimateBinomialP, 1, 0, n_points="a")

    def test_hpd_area(self):
        """Test that the HPD area is correctly set."""
        self.assertRaises(ValueError, EstimateBinomialP, 1, 0, hpd_area=0)
        self.assertRaises(ValueError, EstimateBinomialP, 1, 0, hpd_area=1)

    def test_prior_choice(self):
        """Test that the prior distribution is correctly set."""
        self.assertRaises(ValueError, EstimateBinomialP, 1, 0, prior_dist=1)

    def test_immutability(self):
        """Test that an initialized object is immutable."""
        estimator = EstimateBinomialP(1, 0)
        self.assertRaises(AttributeError, setattr, estimator, "n", 2)
        self.assertRaises(AttributeError, setattr, estimator, "k", 1)
        self.assertRaises(AttributeError, setattr, estimator, "prior_dist", 1)
        self.assertRaises(AttributeError, setattr, estimator, "n_points", 1000)
        self.assertRaises(AttributeError, setattr, estimator, "hpd_area", 0.95)
        self.assertRaises(AttributeError, setattr, estimator, "mode", 0.5)
        self.assertRaises(AttributeError, setattr, estimator, "median", 0.5)
        self.assertRaises(AttributeError, setattr, estimator, "mean", 0.5)
        self.assertRaises(AttributeError, setattr, estimator, "variance", 0.5)
        self.assertRaises(AttributeError, setattr, estimator, "std", 0.5)
        self.assertRaises(AttributeError, setattr, estimator, "hpdi", (0, 1))
        self.assertRaises(AttributeError, setattr, estimator, "err_hi", 0.1)
        self.assertRaises(AttributeError, setattr, estimator, "err_lo", 0.1)
        self.assertRaises(AttributeError, setattr, estimator, "prior", 0.1)
        self.assertRaises(AttributeError, setattr, estimator, "likelihood", 0)
        self.assertRaises(AttributeError, setattr, estimator, "posterior", 0)
