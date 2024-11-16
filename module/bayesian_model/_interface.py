from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.base import BaseEstimator


class NotPositiveHessian(Exception):
    pass


class OptimizationError(Exception):
    pass


class BayesianInterface(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        prior_parameters: dict = {},
        default_parameters: dict = {"m": 0, "p": 5},
        optimize_kwargs: dict = {},
        optimize_method: str = "L-BFGS-B",
    ) -> None:
        self.prior_parameters = prior_parameters
        self.default_parameters = default_parameters
        self.optimize_kwargs = optimize_kwargs
        self.optimize_method = optimize_method

    def _check_distribution_params(self, params: dict) -> None:
        if not (
            isinstance(params, dict)
            and set(params.keys()) == set(["m", "p"])
            and (
                isinstance(params.get("m"), Union[int, float])  # type: ignore
                and not np.isnan(params.get("m"))  # type: ignore
            )
            and (
                isinstance(params.get("p"), Union[int, float])  # type: ignore
                and params.get("p") >= 0  # type: ignore
            )
        ):
            raise ValueError

    @abstractmethod
    def _loss(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Placeholder for train.
        Subclasses should implement this method!
        """

    @abstractmethod
    def _jac(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Placeholder for train.
        Subclasses should implement this method!
        """

    @abstractmethod
    def _diag_hess(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Placeholder for train.
        Subclasses should implement this method!
        """

    def _get_prior_m_p(self, feat_names: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        m = np.array([])
        p = np.array([])
        for col in feat_names:
            m = np.append(
                m,
                self.prior_parameters.get(col, self.default_parameters)["m"],
            )
            p = np.append(
                p,
                self.prior_parameters.get(col, self.default_parameters)["p"],
            )
        return m, p

    def _update_m_p(
        self,
        res: optimize.OptimizeResult,
        X: pd.DataFrame,
        y: pd.Series,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs,
    ) -> None:
        # get new updates values
        m_updated = res.x
        p_updated = self._diag_hess(
            res.x,
            *self._get_args(X.to_numpy(), y.to_numpy(), m, p, **kwargs),
        )
        feat_names = X.columns.to_list()

        # check hessian positiveness
        if not (np.all(np.isfinite(p_updated)) and np.all(p_updated >= 0)):
            raise NotPositiveHessian

        # update parameters
        self.posterior_parameters_ = dict(
            self.prior_parameters,
            **{
                col: {"m": m, "p": p}
                for col, m, p in zip(feat_names, m_updated, p_updated)
            },
        )

        # check parameters
        for _, v in self.posterior_parameters_.items():
            self._check_distribution_params(v)

    @abstractmethod
    def _get_args(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs,
    ) -> tuple:
        """Placeholder for train.
        Subclasses should implement this method!
        """

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> BayesianInterface:
        logging.info(f"Shape of the training dataset: {X.shape}.")

        # get parameter prior distribution parameters (mean and precision)
        m, p = self._get_prior_m_p(X.columns.to_list())
        logging.info("Prior distribution parameters obtained.")

        # optimize
        res = optimize.minimize(
            fun=self._loss,
            x0=m,
            jac=self._jac,
            args=self._get_args(X.to_numpy(), y.to_numpy(), m, p, **kwargs),
            method=self.optimize_method,
            **self.optimize_kwargs,
        )

        # log optimization result
        initial_loss = self._loss(
            m, *self._get_args(X.to_numpy(), y.to_numpy(), m, p, **kwargs)
        )
        log_message = (
            f"initial fun: {initial_loss}\n"
            "The OptimizeResult instance is:\n"
            f"{res}"
        )

        if res.success:
            logging.info(f"Log likelihood optimized successfully.\n{log_message}")

            # calculate parameter posterior distribution parameters
            self._update_m_p(res, X, y, m, p, **kwargs)
            logging.info("Posterior distribution parameters computed.")
        else:
            logging.warning(f"Log likelihood optimization failed.\n{log_message}")
            if res.fun < initial_loss:
                # calculate parameter posterior distribution parameters
                self._update_m_p(res, X, y, m, p, **kwargs)
                logging.info(
                    "Posterior distribution parameters computed even "
                    "though the optimization failed."
                )
            else:
                raise OptimizationError

        return self

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Placeholder for train.
        Subclasses should implement this method!
        """

    @abstractmethod
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Placeholder for train.
        Subclasses should implement this method!
        """

    def get_params(self, deep=True) -> dict:
        return {
            "prior_parameters": self.prior_parameters,
            "default_parameters": self.default_parameters,
            "optimize_kwargs": self.optimize_kwargs,
            "optimize_method": self.optimize_method,
        }

    def set_params(self, **parameters) -> BayesianInterface:
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
