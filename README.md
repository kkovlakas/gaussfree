# gaussfree

Tools for non-Gaussian statistics.

`gaussfree` is a library aiming at offering tools for

* Statistical operations in situations where the Gaussian ditribution is not be correct;
* Estimation of distirbution parameters avoiding the normal approximation;
* Bayesian inference; and
* Additional helper and visualization functions.

While it is far from complete, or generic, in its current implementation, this
is the ultimate goal of the project.

## Features

For now, the features are rather dedicated to a specific project, but there are plans to extend and generalize them.

* **Feature 1**: Estimation of the binomial parameter $p$ (success probability),
                 with arbitrary prior distribution.
* **Feature 2**: Construction of unimodal Beta distributions with a specific
                 mode and variance.

## Installation

Simply download or clone the source, switch to your favourite environment, and run

`pip install .`

from the top level of the repository.

*For the Sphinx autogenerated documentation,
install via `pip` the following packages:
`sphinx_mdinclude`, `sphinx-math-dolar` and `sphinx-rtd-theme`.*

## Examples

### Construct a unimodal Beta distribution

```python
from gaussfree.distributions.beta import get_unimodal_beta

my_prior = get_unimodal_beta(mode=0.75, variance=0.03, verbose=True)
```

### Estimate the binomial $p$

```python
from gaussfree.estimation.binomial import EstimateBinomialP

est = EstimateBinomial(3, 2)  # trials n=3, successes k=2
est.report()                  # print result and show posterior

# plot the 68% highest-posterior density interval
plt.errorbar(0.0, est.mode, yerr[est.err_lo, est.err_hi])

print(f"Posterior mean+/-std: {est.mean:.3f} +/- {est.std:.3f}")


with_prior = EstimateBinomial(3, 2, prior_dist=my_prior)

```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Authors

Konstantinos Kovlakas

## Acknowledgements

I would like to thank Grigoris Maravelias for suggesting publishing my contribution to his paper [link to be added], and his idea on being just the starting point for a more general package.