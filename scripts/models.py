import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import math


def differ(df, arr):
    arr.append(df.dropna().iloc[0])
    return df.diff()


def integrate(df, const):
    return pd.Series(np.r_[const, df.dropna().iloc[0:]].cumsum(), index=df.index)


def create_lags(s, lag=2, dropna=True, name="y"):
    s = pd.Series(s)
    if lag > 0:

        res = pd.concat([s.shift(i) for i in range(1, lag + 1)], axis=1)
        res.columns = ["name(k-%d)" % i for i in range(1, lag + 1)]
        if dropna:
            return res.dropna()
        else:
            return res
    else:
        return pd.Series()


class ARIMA(object):
    def __init__(self, order, window=5, ewm=False):
        self.window = window
        self.ewm = ewm

        self.p = order[0]
        self.d = order[1]
        self.q = order[2]

        self.coefs = None
        self.reserved = None
        self.model = None
        self.results = None

    def fit(self, X):
        self.reserved = X.iloc[-self.d - self.p - self.q - self.window :]
        # reserved values just in case
        consts = []
        for i in range(self.d):
            X = differ(X, consts)
        df = create_lags(X, self.p).dropna()

        if self.q > 0:
            if not self.ewm:
                df = df.join(
                    create_lags(
                        X.rolling(window=self.window).mean().dropna(), self.q, name="ma"
                    )
                ).dropna()
            else:
                df = df.join(
                    create_lags(
                        X.ewm(span=self.window, min_periods=self.window)
                        .mean()
                        .dropna(),
                        self.q,
                        name="ma",
                    )
                ).dropna()

        LR = LinearRegression()
        LR.fit(df, X.iloc[X.shape[0] - df.shape[0] :])
        self.coefs = [LR.intercept_] + list(LR.coef_)
        self.model = LR
        prediction = LR.predict(df)

        for i in range(self.d):
            X = integrate(X, consts[-1 - i])

        if self.d > 0:
            self.results = prediction + X.iloc[X.shape[0] - df.shape[0] :]
        else:
            self.results = prediction
        return self

    def static_predict(self, X):
        num = X.shape[0]
        X = pd.concat([self.reserved, X])
        consts = []
        for i in range(self.d):
            X = differ(X, consts)
        df = create_lags(X, self.p).dropna()
        if self.q > 0:
            if not self.ewm:
                df = df.join(
                    create_lags(
                        X.rolling(window=self.window).mean().dropna(), self.q, name="ma"
                    )
                ).dropna()
            else:
                df = df.join(
                    create_lags(
                        X.ewm(span=self.window, min_periods=self.window)
                        .mean()
                        .dropna(),
                        self.q,
                        name="ma",
                    )
                ).dropna()

        LR_pred = self.model.predict(df.iloc[df.shape[0] - num :])

        for i in range(self.d):
            X = integrate(X, consts[-1 - i])

        if self.d > 0:
            res = pd.Series(
                data=LR_pred + X.iloc[X.shape[0] - LR_pred.shape[0] :],
                index=(X.iloc[X.shape[0] - LR_pred.shape[0] :]).index,
            )
        else:
            res = pd.Series(
                data=LR_pred, index=(X.iloc[X.shape[0] - LR_pred.shape[0] :]).index
            )
        return res

    def dynamic_predict(self, steps=1):
        X = self.reserved
        idx = list(X.index)

        consts = []
        for i in range(self.d):
            X = differ(X, consts)

        prediction = []
        for i in range(steps):
            idx.append(max(idx) + 1)
            df = 0
            df = create_lags(X, self.p).dropna()
            if self.q > 0:
                if not self.ewm:
                    df = df.join(
                        create_lags(
                            X.rolling(window=self.window).mean().dropna(),
                            self.q,
                            name="ma",
                        )
                    ).dropna()
                else:
                    df = df.join(
                        create_lags(
                            X.ewm(span=self.window, min_periods=self.window)
                            .mean()
                            .dropna(),
                            self.q,
                            name="ma",
                        )
                    ).dropna()

            yhat = self.model.predict(df.iloc[-1].values.reshape(1, -1))
            prediction.append(yhat[0])
            X = pd.concat([X, pd.Series(yhat)], ignore_index=True)
        for i in range(self.d):
            X = integrate(X, consts[-1 - i])
        if self.d > 0:
            res = pd.Series(
                prediction + X[X.shape[0] - len(prediction)],
                index=idx[len(idx) - steps :],
            )
        else:
            res = pd.Series(prediction, index=idx[len(idx) - steps :])
        return res

    def predict(self, X):
        return self.static_predict(X)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def get_equation(self):
        string = str(self.p)
        for i in range(1, self.p + 1):
            if self.coefs[i] < 0:
                string += str(self.coefs[i]) + "y(k-" + str(i) + ")"
            else:
                string += "+" + str(self.coefs[i]) + "y(k-" + str(i) + ")"
        for j in range(1, self.q + 2):
            if self.coefs[self.p + j] < 0:
                string += str(self.coefs[self.p + j]) + "ma(k-" + str(j - 1) + ")"
            else:
                string += "+" + str(self.coefs[self.p + j]) + "ma(k-" + str(j - 1) + ")"
        return string


from statsmodels.tsa.stattools import pacf, acf
from sklearn.linear_model import LinearRegression


class ARMA(object):
    def __init__(
        self,
        window=5,
        ewm=False,
        ar_threshold=0.2,
        ma_threshold=0.2,
        nlags=12,
        resid=True,
        own_b=False,
    ):
        self.window = window
        self.ewm = ewm
        self.ar_threshold = ar_threshold
        self.ma_threshold = ma_threshold
        self.nlags = nlags
        self.resid = resid
        self.own_b = own_b

        self.pacf_ar = None
        self.pacf_ma = None
        self.pacf_ar_plot = None
        self.pacf_ma_plot = None

        self.order = None
        self.only_ar_coefs = None

        self.ar_results = None
        self.arma_coefs = None
        self.arma_results = None

    def fit_order(self, y, of):
        pacf_coefs = pd.Series(pacf(y, nlags=self.nlags,))
        if of == "ar":
            self.pacf_ar_plot = sm.graphics.tsa.plot_pacf(
                y, lags=self.nlags, title="AR PACF"
            )
            self.pacf_ar = pacf_coefs
            return pacf_coefs[pacf_coefs.abs() > self.ar_threshold].index.max()
        elif of == "ma":
            self.pacf_ma_plot = sm.graphics.tsa.plot_pacf(
                y, lags=self.nlags, title="MA PACF"
            )
            self.pacf_ma = pacf_coefs
            return pacf_coefs[pacf_coefs.abs() > self.ma_threshold].index.max()

    def predict(self, X):
        X = np.array(X)
        p = self.fit_order(X, of="ar")
        lagged_ar_df = create_lags(X, p)

        X = X[p:]
        LR = LinearRegression()
        LR.fit(lagged_ar_df, X)

        self.only_ar_coefs = [LR.intercept_] + list(LR.coef_)
        LR_pred = LR.predict(lagged_ar_df)

        self.ar_results = LR_pred, X

        if self.window == 0:
            return self.ar_results

        if self.resid:
            resid = pd.Series(X - LR_pred)
            q = self.fit_order(resid, of="ma")
            if self.ewm:
                lagged_ma_df = create_lags(
                    resid.ewm(span=self.window, min_periods=self.window).mean(),
                    q,
                    name="ma",
                )
                rolling = (
                    resid.ewm(span=self.window, min_periods=self.window).mean().dropna()
                )
            else:
                lagged_ma_df = create_lags(
                    resid.rolling(window=self.window).mean(), q, name="ma"
                )
                rolling = resid.rolling(window=self.window).mean().dropna()
        else:
            X = pd.Series(X)
            if self.ewm:
                rolling = (
                    X.ewm(span=self.window, min_periods=self.window).mean().dropna()
                )
            else:
                rolling = X.rolling(window=self.window).mean().dropna()

            q = self.fit_order(rolling, of="ma")
            lagged_ma_df = create_lags(rolling, q, name="ma")

        self.order = p, q

        if self.own_b:
            alpha = 2 / (q + 1)
            div = sum([(1 - alpha) ** j for j in range(1, q + 1)])
            b = np.array([((1 - alpha) ** i) / div for i in range(1, q + 1)])
            X_m_roll = X[X.shape[0] - lagged_ma_df.shape[0] :] - rolling
            for i in range(0, q):
                X_m_roll = (
                    X_m_roll[X_m_roll.shape[0] - lagged_ma_df.iloc[:, i].shape[0] :]
                    - b[i] * lagged_ma_df.iloc[:, i]
                )
            X_m_roll = X_m_roll.dropna()
            arma_df = lagged_ar_df
        else:
            if q > 0:
                X_m_roll = (
                    X[X.shape[0] - lagged_ma_df.shape[0] :]
                    - rolling[
                        rolling.shape[0]
                        - X[X.shape[0] - lagged_ma_df.shape[0] :].shape[0] :
                    ]
                ).dropna()
                arma_df = pd.concat(
                    [
                        lagged_ar_df.iloc[
                            lagged_ar_df.shape[0] - lagged_ma_df.shape[0] :
                        ].reset_index(drop=True),
                        lagged_ma_df.reset_index(drop=True),
                    ],
                    axis=1,
                )
            else:
                X_m_roll = X[X.shape[0] - rolling.shape[0] :] - rolling
                arma_df = lagged_ar_df

        LR.fit(arma_df[arma_df.shape[0] - X_m_roll.shape[0] :], X_m_roll)

        self.arma_coefs = (
            [LR.intercept_] + list(LR.coef_[:p]) + [1] + list(LR.coef_[p:])
        )

        if self.own_b:
            self.arma_coefs.extend(b)

        if self.own_b:
            csx = LR.predict(arma_df)
            tmp = [0, 0]
            tmp[0] = csx[csx.shape[0] - rolling.shape[0] :] + rolling
            tmp[1] = X[
                X.shape[0]
                - (csx[csx.shape[0] - rolling.shape[0] :] + rolling).shape[0] :
            ]
            for i in range(0, q):
                tmp[0] += b[i] * lagged_ma_df.iloc[:, i]
            tmp[0] = tmp[0].dropna()
            self.arma_results = tmp[0], tmp[1][tmp[1].shape[0] - tmp[0].shape[0] :]
            return self.arma_results
        else:
            # some intence programing skills
            self.arma_results = (
                LR.predict(arma_df)[len(LR.predict(arma_df)) - rolling[q:].shape[0]]
                + rolling[q:],
                X[
                    X.shape[0]
                    - (
                        LR.predict(arma_df)[
                            len(LR.predict(arma_df)) - rolling[q:].shape[0]
                        ]
                        + rolling[q:]
                    ).shape[0] :
                ],
            )
            return self.arma_results

    def get_equation(self):
        string = str(self.arma_coefs[0])
        for i in range(1, self.order[0] + 1):
            if self.arma_coefs[i] < 0:
                string += str(self.arma_coefs[i]) + "y(k-" + str(i) + ")"
            else:
                string += "+" + str(self.arma_coefs[i]) + "y(k-" + str(i) + ")"
        for j in range(1, self.order[1] + 2):
            if self.arma_coefs[self.order[0] + j] < 0:
                string += (
                    str(self.arma_coefs[self.order[0] + j]) + "v(k-" + str(j - 1) + ")"
                )
            else:
                string += (
                    "+"
                    + str(self.arma_coefs[self.order[0] + j])
                    + "v(k-"
                    + str(j - 1)
                    + ")"
                )
        return string
