
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
- `argparse`

You can install the necessary libraries using pip:

```bash
pip install tensorflow tensorboard numpy pandas pytorch_lightning torch pytorch_forecasting RobustSTL
```

Ensure you have a GPU setup for PyTorch, as the project is optimized to run on a GPU for faster training.


## Usage

### Data Loading and Preprocessing (`data_loader.py`)
The script is used for loading and preprocessing the dataset, converting columns to appropriate data types.

### Model Preparation and Training (`prepare_train_model.py`)
This script handles feature selection, dataset preparation, optional hyperparameter optimization, and model training.

**Command-line Arguments:**
- `--data_file`: Path to your data file.
- `--target_feature`: Target feature for the model (e.g., 'Trend', 'Seasonal', 'Resid').
- `--optimize_hyperparameters`: Enable hyperparameter optimization.
- Additional model hyperparameters (e.g., `--learning_rate`, `--hidden_size` , `--gradient_clip_val` , `--dropout` , `--hidden_continuous_size` , `--attention_head_size`).

**Execution Command:**
To train the model with default settings:
```
python prepare_train_model.py --data_file "path_to_your_data_file.xlsx" --target_feature "Trend" --learning_rate 0.01 --hidden_size 128
```
*Note: Similar to the above, you can use the system's Python interpreter or an interpreter from a virtual environment.*

*Showing an example on my computer:       D:\anaconda3\envs\myenv1\python.exe prepare_train_model.py --data_file "C:/Users/xuyechi/Desktop/科研/tft/夏威夷景区每日客流量（2009年7月至今）.xlsx" --target_feature "Trend" --learning_rate 0.00842077448532244 --hidden_size 125 --gradient_clip_val 0.03911626926390909 --dropout 0.15160823136480017 --hidden_continuous_size 17 --attention_head_size 1*

## Model Output
The trained model is saved as `tft_model.ckpt`, and the script outputs the forecasted values for the selected target feature.

