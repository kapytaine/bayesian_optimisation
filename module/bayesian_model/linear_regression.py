from typing import Optional

import numpy as np
import pandas as pd

from .link import Identity
from .link._interface import Link
from ._interface import BayesianInterface


class BayesianNormalRegression(BayesianInterface):
    def __init__(self, link: Link = Identity(), **kwargs):
        self.link = link
        super().__init__(**kwargs)

    def _likelihood(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        r: float,
        scale: Optional[float] = None,
        **kwargs,
    ) -> float:
        weights = np.ones(y.shape)
        weights[y == 0] = 1 / r

        return (
            self._loss(beta, X, y, m, p, r, **dict({"scale": scale}, **kwargs))
            - 0.5 * np.sum(np.log(weights))
            + 0.5 * (np.sum(np.log(1 / p)) + len(p) * np.log(2 * np.pi))
        )

    def _loss(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        scale: Optional[float] = None,
        **kwargs,
    ) -> float:
        linear_pred = np.dot(X, beta)
        y_pred = self.link._inv_link(linear_pred)

        if scale is None:
            scale = self._estimate_scale(X, y, y_pred, bool(np.all(p == 0)))

        weights = np.ones(y.shape)

        loss = (
            np.sum(
                np.multiply(
                    weights,
                    1 / 2 * np.divide(np.power(np.subtract(y, y_pred), 2), scale),
                )
            )
            + 0.5 * len(y) * (np.log(scale) + np.log(2 * np.pi))
            + np.sum(
                1
                / 2
                * np.multiply(
                    np.power(np.subtract(beta, m), 2),
                    p,
                )
            )
        )
        return loss

    def _jac(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        scale: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        linear_pred = np.dot(X, beta)
        y_pred = self.link._inv_link(linear_pred)

        if scale is None:
            scale = self._estimate_scale(X, y, y_pred, bool(np.all(p == 0)))

        weights = np.ones(y.shape)

        jac = -np.sum(
            np.multiply(
                np.expand_dims(
                    np.multiply(
                        weights,
                        np.divide(np.subtract(y, y_pred), scale),
                    ),
                    axis=1,
                ),
                np.multiply(
                    np.expand_dims(
                        self.link._jac_inv_link(linear_pred),
                        axis=1,
                    ),
                    X,
                ),
            ),
            axis=0,
        ) + np.multiply(np.subtract(beta, m), p)
        return jac

    def _diag_hess(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        scale: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        linear_pred = np.dot(X, beta)
        y_pred = self.link._inv_link(linear_pred)

        if scale is None:
            scale = self._estimate_scale(X, y, y_pred, bool(np.all(p == 0)))

        weights = np.ones(y.shape)

        diag_hess = np.add(
            np.sum(
                np.multiply(
                    np.power(X, 2),
                    np.expand_dims(
                        np.divide(
                            np.multiply(
                                weights,
                                np.subtract(
                                    np.power(
                                        self.link._jac_inv_link(linear_pred),
                                        2,
                                    ),
                                    np.multiply(
                                        np.subtract(y, y_pred),
                                        self.link._hess_inv_link(linear_pred),
                                    ),
                                ),
                            ),
                            scale,
                        ),
                        axis=1,
                    ),
                ),
                axis=0,
            ),
            p,
        )
        return diag_hess

    def _hess(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        scale: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        linear_pred = np.dot(X, beta)
        y_pred = self.link._inv_link(linear_pred)

        if scale is None:
            scale = self._estimate_scale(X, y, y_pred, bool(np.all(p == 0)))

        weights = np.ones(y.shape)

        hess = np.add(
            np.matmul(
                np.matmul(
                    X.T,
                    np.diag(
                        np.divide(
                            np.multiply(
                                weights,
                                np.subtract(
                                    np.power(
                                        self.link._jac_inv_link(linear_pred),
                                        2,
                                    ),
                                    np.multiply(
                                        np.subtract(y, y_pred),
                                        self.link._hess_inv_link(linear_pred),
                                    ),
                                ),
                            ),
                            scale,
                        )
                    ),
                ),
                X,
            ),
            np.diag(p),
        )
        return hess

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        m_posterior = np.array(
            [
                self.posterior_parameters_.get(col, self.default_parameters)["m"]
                for col in X.columns
            ]
        )
        m_prior = np.array(
            [
                self.prior_parameters.get(col, self.default_parameters)["m"]
                for col in X.columns
            ]
        )
        p_prior = np.array(
            [
                self.prior_parameters.get(col, self.default_parameters)["p"]
                for col in X.columns
            ]
        )
        return -self._loss(m_posterior, X.to_numpy(), y.to_numpy(), m_prior, p_prior)

    def get_params(self, deep=True):
        shared_params = super().get_params()
        return dict(shared_params, **{"link": self.link})

    def _estimate_scale(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        without_prior: bool,
    ):
        weights = np.ones(y.shape)
        if without_prior:
            df_resid = np.sum(weights) - X.shape[1]
        else:
            df_resid = np.sum(weights)

        if df_resid <= 0:
            raise ValueError(f"Scale estimation is wrong since df_resid={df_resid}.")

        return np.sum(np.multiply(np.power(y - y_pred, 2), weights)) / df_resid

    def _get_args(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs,
    ) -> tuple:
        return (
            X,
            y,
            m,
            p,
            kwargs.get("scale"),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        m = [
            self.posterior_parameters_.get(col, self.default_parameters)["m"]
            for col in X.columns
        ]
        linear_pred = np.dot(X, np.array(m))
        return self.link._inv_link(linear_pred)
