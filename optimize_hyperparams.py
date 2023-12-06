import argparse
import pickle
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# 解析命令行参数
parser = argparse.ArgumentParser(description="Prepare data and optimize hyperparameters for the Temporal Fusion Transformer model.")
parser.add_argument("--data_file", type=str, default="path_to_your_data.xlsx", help="Path to the data file.")
parser.add_argument("--target_feature", type=str, default="Trend", help="Target feature for the model (e.g., 'Trend', 'Seasonal', 'Resid').")
parser.add_argument("--optimize_hyperparameters", action="store_true", help="Run hyperparameter optimization")
args = parser.parse_args()

# Import and load data using the data loader module
from data_loader import load_data
data = load_data(args.data_file)

# Prepare the dataset for training the model
# Setting up the TimeSeriesDataSet for training
max_prediction_length = 3
max_encoder_length = 30
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= 4481],
    time_idx="time_idx",
    target=args.target_feature,
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_categoricals=["month", "day of the week", "day", "Holiday"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=[args.target_feature],
    group_ids=['destination'],
    target_normalizer=GroupNormalizer(groups=['destination'], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)

# Prepare validation dataset and data loaders for training and validation
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# This section is dedicated to hyperparameter optimization using the Tree-structured Parzen Estimator (TPE) 
# algorithm within the Optuna framework. TPE is an advanced Bayesian optimization technique that models
# the probability distribution of hyperparameters and efficiently narrows down the search space.
# This step, while optional due to its time-consuming nature, is crucial for fine-tuning the model 
# to achieve optimal performance. For more information on TPE and its workings within Optuna, 
# you can refer to the official documentation: 
# https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html

def optimize_hyperparams(train_dataloader, val_dataloader):
# Initialize the hyperparameter optimization study using Optuna's TPE algorithm
# to search for the best hyperparameters over a specified number of trials.
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=50,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False
    )
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    return study.best_trial.params


if __name__ == "__main__":
    if args.optimize_hyperparameters:
        best_params = optimize_hyperparams(train_dataloader, val_dataloader)
        print(f"Best hyperparameters: {best_params}")
    else:
        print("Hyperparameter optimization is disabled. Proceed with predefined or manual hyperparameters.")
