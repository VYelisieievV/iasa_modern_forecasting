from numpy.core.numeric import False_
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


def inference_statinarity(y, stat_test="adfuller", index_of_critical_values=4):
    if stat_test == "adfuller":
        result = adfuller(y)
    elif stat_test == "kpss":
        result = kpss(y)
    else:
        raise ValueError(
            "Invalid test name:" + str(stat_test) + ". Try 'adfuller' or 'kpss'"
        )
    parsed_result = {
        "statistic": result[0],
        "p_value": result[1],
    }
    parsed_result.update(
        {
            "critical_value_" + k: v
            for k, v in result[index_of_critical_values].items()
        }
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
    fig = go.Figure([{"x": y.index, "y": y}])
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


def compute_pacf_acf(input, plot=True):
    acf_r = acf(input)
    pacf_r = pacf(input)

    if plot:
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(input, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(input, ax=ax2)

    return acf_r, pacf_r
