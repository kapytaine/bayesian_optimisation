import numpy as np

from ._interface import Link


class Identity(Link):
    def _inv_link(self, x: np.ndarray) -> np.ndarray:
        return x

    def _link(self, y: np.ndarray) -> np.ndarray:
        return y

    def _inv_jac_link(self, y: np.ndarray) -> np.ndarray:
        return np.ones_like(y)

    def _hess_link(self, y: np.ndarray) -> np.ndarray:
        return np.zeros_like(y)
