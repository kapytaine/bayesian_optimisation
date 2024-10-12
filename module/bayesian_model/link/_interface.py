from abc import ABCMeta, abstractmethod

import numpy as np


class Link(metaclass=ABCMeta):
    @abstractmethod
    def _inv_link(self, x: np.ndarray) -> np.ndarray:
        """Placeholder for train.
        Subclasses should implement this method!
        """

    def _jac_inv_link(self, x: np.ndarray) -> np.ndarray:
        return self._inv_jac_link(self._inv_link(x))

    def _hess_inv_link(self, x: np.ndarray) -> np.ndarray:
        return np.multiply(
            -self._hess_link(self._inv_link(x)),
            np.power(self._jac_inv_link(x), 3),
        )

    @abstractmethod
    def _link(self, y: np.ndarray) -> np.ndarray:
        """Placeholder for train.
        Subclasses should implement this method!
        """

    @abstractmethod
    def _inv_jac_link(self, y: np.ndarray) -> np.ndarray:
        """Placeholder for train.
        Subclasses should implement this method!
        This method has been chosen to avoid numerical error.
        When working with _jac_link, it may be close to 1/0 close to the
        optimum resulting in an zero division error.
        """

    @abstractmethod
    def _hess_link(self, y: np.ndarray) -> np.ndarray:
        """Placeholder for train.
        Subclasses should implement this method!
        """
