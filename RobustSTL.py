import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RobustSTL import RobustSTL


def decompose_time_series(data_series, plot=False):
    """
    Decompose a given time series using RobustSTL.

    Parameters:
    - data_series (pd.Series): The time series data to be decomposed.
    - plot (bool): Whether to plot the decomposed components.

    Returns:
    - tuple: A tuple containing the trend, seasonal, and residual components.
    """
    RSTL = RobustSTL(data_series, period=365, seasonal=11, robust=True)
    trend, seasonal, residual = RSTL.fit().stl

    if plot:
        plt.figure(figsize=(12, 8))

        plt.subplot(4, 1, 1)
        plt.plot(data_series, label="Original")
        plt.legend(loc="upper left")

        plt.subplot(4, 1, 2)
        plt.plot(trend, label="Trend")
        plt.legend(loc="upper left")

        plt.subplot(4, 1, 3)
        plt.plot(seasonal, label="Seasonal")
        plt.legend(loc="upper left")

        plt.subplot(4, 1, 4)
        plt.plot(residual, label="Residual")
        plt.legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    return trend, seasonal, residual


if __name__ == "__main__":
    # Example usage
    data = pd.read_excel("your file.xlsx")
    time_series_data = data["your_column_name"]  # replace with your column name
    trend, seasonal, residual = decompose_time_series(time_series_data, plot=True)