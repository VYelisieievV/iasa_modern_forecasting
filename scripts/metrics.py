import numpy as np
import pandas as pd
import math
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred, model_params):
    return mse(y_true, y_pred) ** 0.5


def sse(y_true, y_pred):
    resid = y_true - y_pred
    return resid.T @ resid


def mape(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mae(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


def dispersion(distribution):
    return ((distribution - distribution.mean()) ** 2).mean()


def determination_coef(y_true, y_pred):
    dis = ((y_true - y_pred) ** 2).mean()
    testdis = ((y_true - y_true.mean()) ** 2).mean()
    return r2_score(y_true, y_pred)  # (testdis / (testdis + dis)) ** 2


def log_likelihood(y_true, y_pred):
    resid = y_true - y_pred
    return (-y_true.shape[0] / 2) * (
        1 + np.log(2 * math.pi) + np.log(resid.T @ resid / y_true.shape[0])
    )


def mean_dependent_var(y_true, y_pred):
    return (y_true - y_pred).mean()


def std_dependent_var(y_true, y_pred):
    return (y_true - y_pred).std()


def akkake_criteria(y_true, y_pred, model_params):
    return (
        -2 * log_likelihood(y_true, y_pred) / y_true.shape[0]
    ) + 2 * model_params / y_true.shape[0]


def shwarz_criteria(y_true, y_pred, model_params):
    return (-2 * log_likelihood(y_true, y_pred) / y_true.shape[0]) + np.log(
        y_true.shape[0]
    ) * model_params / y_true.shape[0]


def get_all_metrics(y_true, y_pred, model_params, caption):
    res = {
        "AIC": akkake_criteria(y_true, y_pred, model_params),
        "RMSE": rmse(y_true, y_pred, model_params),
        "MSE": mse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Durbin-Watson": durbin_watson(y_true - y_pred),
        "R-squared": determination_coef(y_true, y_pred),
        "Schwarz criteria": shwarz_criteria(y_true, y_pred, model_params),
        "SSE": sse(y_true, y_pred),
        "Log Likelihood": log_likelihood(y_true, y_pred),
    }
    all_key = list(res.keys())
    for k in all_key:
        res[k + f"_{caption}"] = res.pop(k)
    return res
