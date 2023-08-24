
# Enhancing Tourism Demand Forecasting with a Transformer-based Framework

This study introduces an innovative framework that harnesses the most recent Transformer architecture to enhance tourism demand forecasting. The proposed transformer-based model integrates the tree-structured Parzen estimator for hyperparameter optimization, a robust time series decomposition approach, and a temporal fusion transformer for multivariate time series prediction. Our approach initially employs the decomposition method to decompose the data series to effectively mitigate the influence of outliers. The temporal fusion transformer is subsequently utilized for forecasting, and its hyperparameters are meticulously fine-tuned by a Bayesian-based algorithm, culminating in a more efficient and precise model for tourism demand forecasting. Our model surpasses existing state-of-the-art methodologies in terms of forecasting accuracy and robustness.

## Prerequisites

Ensure the following Python libraries are installed:

- `os`
- `warnings`
- `tensorflow`
- `tensorboard`
- `numpy`
- `pandas`
- `pytorch_lightning`
- `torch`
- `pytorch_forecasting`
- `pickle`
- `pathlib`
- `RobustSTL`

You can install the necessary libraries using pip:

```bash
pip install tensorflow tensorboard numpy pandas pytorch_lightning torch pytorch_forecasting RobustSTL
```

Ensure you have a GPU setup for PyTorch, as the project is optimized to run on a GPU for faster training.

## Directory Structure

Place the following files in the same directory:

- `main.py`: Primary execution script.
- `imports.py`: Contains library imports.
- `data_preprocessing.py`: Handles data loading and preprocessing.
- `robust_stl_decomposition.py`: Decomposes the time series using RobustSTL.
- `dataset_creation.py`: Processes data for model training.
- `hyperparameter_optimization.py`: Contains the hyperparameter optimization routine.
- `model_training.py`: Defines and trains the Temporal Fusion Transformer model.
- `model_prediction.py`: Predicts using the trained model.

## Running the Project

1. **Data Setup**: Place your data file (e.g., `your file.xlsx`) in the same directory. Adjust paths in `data_preprocessing.py` and `robust_stl_decomposition.py` if needed.
2. **Time Series Decomposition**: Before running the main model, execute the `robust_stl_decomposition.py` to decompose the original time series:

```bash
python robust_stl_decomposition.py
```

3. **Execute Main Script**: Run the `main.py` script:

```bash
python main.py
```

4. **Model Output**: The best model will be saved under `saved_models`. Predictions are made using the best model checkpoint.
