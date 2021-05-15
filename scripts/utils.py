from numpy.core.numeric import False_
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import jarque_bera
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.stattools import adfuller, kpss
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from joblib import Parallel


def inference_statinarity(y, stat_test="adfuller", index_of_critical_values=4):
    if stat_test == "adfuller":
        result = adfuller(y)
    elif stat_test == "kpss":
        result = kpss(y, nlags="auto")
    else:
        raise ValueError(
            "Invalid test name:" + str(stat_test) + ". Try 'adfuller' or 'kpss'"
        )
    parsed_result = {
        "statistic": result[0],
        "p_value": result[1],
    }
    parsed_result.update(
        {"critical_value_" + k: v for k, v in result[index_of_critical_values].items()}
    )
    return {stat_test: parsed_result}


def get_stat_overview(y):
    return pd.Series(
        data=[
            y.mean(),
            y.median(),
            y.max(),
            y.min(),
            y.std(),
            y.skew(),
            y.kurtosis(),
            jarque_bera(y)[0],
            jarque_bera(y)[1],
        ],
        index=[
            "Mean",
            "Median",
            "Max",
            "Min",
            "Std",
            "Skewness",
            "Kurtosis",
            "Jarque_Bera",
            "Jarque_Bera_p",
        ],
    )


def plot_ts_hist(y, bins=0.2, rug=False):
    fig = ff.create_distplot([y], [y.name], bin_size=bins, show_rug=rug)
    fig.show()


def plot_ts(y):
    fig = go.Figure()

    if type(y) == type([]) or type(y) == type(list()):
        for el in y:
            fig.add_trace(go.Scatter({"x": el.index, "y": el, "name": el.name}))
    else:
        fig.add_trace(go.Scatter({"x": y.index, "y": y, "name": y.name}))
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        )
    )
    fig.show()


def compute_pacf_acf(input, acf_bound, pacf_bound, nlags=30, plot=False):
    acf_r = acf(input, nlags=nlags)
    pacf_r = pacf(input, nlags=nlags)
    if plot:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))

        fig.add_trace(
            go.Scatter(x=np.arange(nlags + 1), y=acf_r, showlegend=False), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=np.arange(nlags + 1), y=pacf_r, showlegend=False), row=2, col=1
        )
        fig.add_hrect(
            y0=-acf_bound,
            y1=acf_bound,
            line_width=0,
            fillcolor="turquoise",
            opacity=0.5,
            row=1,
            col=1,
        )
        fig.add_hrect(
            y0=-pacf_bound,
            y1=pacf_bound,
            line_width=0,
            fillcolor="yellowgreen",
            opacity=0.5,
            row=2,
            col=1,
        )

        fig.show()

    print("ACF Important lags: ", np.where(np.abs(acf_r) > acf_bound)[0])
    print("PACF Important lags: ", np.where(np.abs(pacf_r) > pacf_bound)[0])

    return acf_r, pacf_r


def plot_seasonal_decompose(result):
    trace1 = go.Scatter(x=result.observed.index, y=result.observed)

    trace2 = go.Scatter(x=result.trend.index, y=result.trend, yaxis="y2")

    trace3 = go.Scatter(x=result.seasonal.index, y=result.seasonal, yaxis="y3")

    trace4 = go.Scatter(
        x=result.resid.index, y=result.resid, mode="markers", yaxis="y4"
    )
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 3, 1)
    fig.append_trace(trace4, 4, 1)

    fig["layout"].update(
        showlegend=False,
        height=600,
        yaxis=dict(title="Observed"),
        yaxis2=dict(title="Trend"),
        yaxis3=dict(title="Seasonal"),
        yaxis4=dict(title="Residual"),
        xaxis4=dict(title="Month"),
    )

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            type="date",
        ),
        xaxis4_rangeslider_visible=True,
        xaxis4_rangeslider_thickness=0.075,
    )
    fig.show()


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
