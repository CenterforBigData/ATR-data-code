import argparse
from sample_generator import *
from RobustSTL import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(input_file, output_file, season_len, reg1, reg2, K, H, ds1):
    # Load dataset
    data = pd.read_excel(input_file)
    data["tourist"] = data["tourist"].astype("float64")

    # Extract the 'tourist' column values as a 1D numpy array
    time_series = data['tourist'].values

    # Reshape the 1D array to make it compatible for further processing
    time_series = time_series.reshape(-1)

    # Standardize the time series data
    mean_value = np.mean(time_series)
    std_dev = np.std(time_series)
    normalized_series = (time_series - mean_value) / std_dev

    # Apply the RobustSTL algorithm
    result = RobustSTL(normalized_series, season_len, reg1=reg1, reg2=reg2, K=K, H=H, ds1=ds1)

    # Extract the trend, seasonal, and residual components
    trend = result[1]
    seasonal = result[2]
    resid = result[3]

    # De-standardize the components
    original_trend = trend * std_dev + mean_value
    original_seasonal = seasonal * std_dev + mean_value
    original_resid = resid * std_dev + mean_value

    # Add the components to the DataFrame
    data['Trend'] = original_trend
    data['Seasonal'] = original_seasonal
    data['Resid'] = original_resid

    # Write to a new excel file
    data.to_excel(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RobustSTL on a time series dataset.')
    parser.add_argument('--input_file', type=str, default='Jiuzhaigou data.xlsx', help='Input EXCEL file containing the time series data')
    parser.add_argument('--output_file', type=str, default='decomposed_series.xlsx', help='Output EXCEL file to save the results')
    parser.add_argument('--season_len', type=int, default=50, help='Length of seasonal period')
    parser.add_argument('--reg1', type=float, default=10.0, help='First order regularization parameter for trend extraction')
    parser.add_argument('--reg2', type=float, default=0.5, help='Second order regularization parameter for trend extraction')
    parser.add_argument('--K', type=int, default=2, help='Number of past season samples in seasonality extraction')
    parser.add_argument('--H', type=int, default=5, help='Number of neighborhood in seasonality extraction')
    parser.add_argument('--ds1', type=float, default=10, help='Hyperparameter of bilateral filter in seasonality extraction step')

    args = parser.parse_args()

    # Call the main function with provided command line arguments
    main(args.input_file, args.output_file, args.season_len, args.reg1, args.reg2, args.K, args.H, args.ds1)
