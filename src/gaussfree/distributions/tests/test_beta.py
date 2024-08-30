"""Unit tests for beta distribution functions."""


from unittest import TestCase
from gaussfree.distributions.beta import get_unimodal_beta


class TestUnimodalBeta(TestCase):
    """Unit tests for the `get_unimodal_beta` function."""
    def test_mode_range(self):
        """Test that the mode is not allowed outside the correct range."""
        modes_to_test = [-0.1, 0.0, 1.0, 1.1]
        for mode_to_test in modes_to_test:
            with self.assertRaises(ValueError) as context:
                get_unimodal_beta(mode=mode_to_test, variance=0.01)
            self.assertIn("Mode must be in (0, 1)", str(context.exception))

    def test_positive_variance(self):
        """Test that the variance is not allowed to be negative."""
        with self.assertRaises(ValueError) as context:
            get_unimodal_beta(mode=0.5, variance=-0.01)
        self.assertIn("Variance must be positive", str(context.exception))

    def test_variance_not_extreme(self):
        """Test that the variance is not allowed outside the correct range."""
        with self.assertRaises(ValueError) as context:
            get_unimodal_beta(mode=0.5, variance=1.0/12.0)
        self.assertIn("Not unimodal for variance greater than or equal to", str(context.exception))

    def test_normal_function(self):
        """Test that the function runs normally for various inputs."""
        inputs = [(0.5, 0.08), (0.25, 0.001), (0.75, 0.001)]
        for input_params in inputs:
            try:
                get_unimodal_beta(mode=input_params[0], variance=input_params[1], verbose=False)
            except (ValueError, RuntimeError, AssertionError) as e:
                self.fail(f"Function failed with input {input}: {e}")
