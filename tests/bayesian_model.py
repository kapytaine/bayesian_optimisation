import numpy as np
from scipy.optimize import approx_fprime

from module.bayesian_model import (
    BayesianLogisticRegression,
    BayesianNormalRegression,
)

# binary case
beta = np.random.normal(size=4)
X = np.random.normal(size=(10, 4))
y = np.random.choice([0, 1], size=10)
m = np.random.normal(size=4)
p = np.exp(np.random.normal(size=4))

np.testing.assert_allclose(
    approx_fprime(
        beta,
        BayesianLogisticRegression()._loss,
        1.4901161193847656e-08,
        *(X, y, m, p),
    ),
    BayesianLogisticRegression()._jac(beta, X, y, m, p),
    atol=1e-6,
    rtol=1e-6,
)

np.testing.assert_allclose(
    approx_fprime(
        beta,
        BayesianLogisticRegression()._jac,
        1.4901161193847656e-08,
        *(X, y, m, p),
    ),
    BayesianLogisticRegression()._hess(beta, X, y, m, p),
    atol=1e-6,
    rtol=1e-6,
)

# regression case
beta = np.random.normal(size=4)
X = np.random.normal(size=(10, 4))
y = np.random.normal(size=10)
m = np.random.normal(size=4)
p = np.exp(np.random.normal(size=4))


class MockBayesianNormalRegression(BayesianNormalRegression):
    def _get_args(
        self,
        *args,
        **kwargs,
    ) -> tuple:
        kwargs["scale"] = 1
        return super()._get_args(*args, **kwargs)


np.testing.assert_allclose(
    approx_fprime(
        beta,
        MockBayesianNormalRegression()._loss,
        1.4901161193847656e-08,
        *MockBayesianNormalRegression()._get_args(X, y, m, p),
    ),
    MockBayesianNormalRegression()._jac(
        beta, *MockBayesianNormalRegression()._get_args(X, y, m, p)
    ),
    atol=1e-6,
    rtol=1e-6,
)

np.testing.assert_allclose(
    approx_fprime(
        beta,
        MockBayesianNormalRegression()._jac,
        1.4901161193847656e-08,
        *MockBayesianNormalRegression()._get_args(X, y, m, p),
    ),
    MockBayesianNormalRegression()._hess(
        beta, *MockBayesianNormalRegression()._get_args(X, y, m, p)
    ),
    atol=1e-6,
    rtol=1e-6,
)
