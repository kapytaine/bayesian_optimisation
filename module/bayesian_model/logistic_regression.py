import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


from .link import Logit
from .link._interface import Link
from ._interface import BayesianInterface


class BayesianLogisticRegression(BayesianInterface):
    def __init__(self, link: Link = Logit(), **kwargs):
        self.link = link
        super().__init__(**kwargs)

    def _loss(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Negative log likelihood
        """
        linear_pred = np.dot(X, beta)
        y_pred = self.link._inv_link(linear_pred)

        weights = np.ones(y.shape)

        loss = (
            log_loss(y, y_pred, sample_weight=weights, normalize=False)
            + 0.5 * np.dot(np.multiply(p, np.subtract(beta, m)), np.subtract(beta, m))
            + 0.5 * (np.sum(np.log(1 / p)) + len(p) * np.log(2 * np.pi))
        )

        return loss

    def _jac(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        LogLikelihood defines the regularized loss function
        :param omega: vector to optimize
        :param y: responses (1/-1) of training data
        :param X: dimensions of training data
        :param m: previous vector of means
        :param p: previous vector of inverse variances
        :return out: value of loss function
        :return grad: gradient of loss function
        """
        linear_pred = np.dot(X, beta)
        y_pred = self.link._inv_link(linear_pred)

        weights = np.ones(y.shape)

        jac = np.add(
            -np.dot(weights * (y - y_pred), X),
            np.multiply(p, np.subtract(beta, m)),
        )

        return jac

    def _diag_hess(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        linear_pred = np.dot(X, beta)
        y_pred = self.link._inv_link(linear_pred)

        weights = np.ones(y.shape)

        diag_hess = p + np.dot(
            np.power(X, 2).T,
            np.multiply(weights, np.multiply(y_pred, 1 - y_pred)),
        )
        return diag_hess

    def _hess(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        linear_pred = np.dot(X, beta)
        y_pred = self.link._inv_link(linear_pred)

        weights = np.ones(y.shape)

        hess = np.diag(p) + np.matmul(
            np.matmul(
                X.T,
                np.diag(np.multiply(weights, np.multiply(y_pred, 1 - y_pred))),
            ),
            X,
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

    def _get_args(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        p: np.ndarray,
        **kwargs,
    ) -> tuple:
        return (X, y, m, p)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        m = [
            self.posterior_parameters_.get(col, self.default_parameters)["m"]
            for col in X.columns
        ]
        linear_pred = np.dot(X.to_numpy(), np.array(m))
        return self.link._inv_link(linear_pred)
