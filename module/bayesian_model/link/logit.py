import numpy as np
from scipy.special import expit, logit

from ._interface import Link


class Logit(Link):
    def _inv_link(self, x: np.ndarray) -> np.ndarray:
        return expit(x)

    def _jac_inv_link(self, x: np.ndarray) -> np.ndarray:
        return expit(x) * (1 - expit(x))

    def _hess_inv_link(self, x: np.ndarray) -> np.ndarray:
        return expit(x) * (1 - expit(x)) * (1 - 2 * expit(x))

    def _link(self, y: np.ndarray) -> np.ndarray:
        return logit(y)

    def _inv_jac_link(self, y: np.ndarray) -> np.ndarray:
        return y * (1 - y)

    def _hess_link(self, y: np.ndarray) -> np.ndarray:
        return (2 * y - 1) / (y * (1 - y)) ** 2
