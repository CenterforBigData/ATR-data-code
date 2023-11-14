import argparse
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# Used to resolve the "module ‘tensorflow._api.v2.io.gfile’ has no attribute ‘get_filesystem’" error
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Parse command-line arguments for model configuration
parser = argparse.ArgumentParser(description="Prepare and train the Temporal Fusion Transformer model.")
parser.add_argument("--data_file", type=str, default="path_to_your_data.xlsx", help="Path to the data file.")
parser.add_argument("--target_feature", type=str, default="Trend", help="Target feature for the model (e.g., 'Trend', 'Seasonal', 'Resid').")
parser.add_argument("--optimize_hyperparameters", action="store_true", help="Run hyperparameter optimization")
parser.add_argument("--gradient_clip_val", type=float, default=0.1, help="Gradient clipping value")
parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for the model")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--hidden_continuous_size", type=int, default=50, help="Hidden continuous size")
parser.add_argument("--attention_head_size", type=int, default=1, help="Attention head size")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
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

# Function for hyperparameter optimization using Optuna
def optimize_hyperparams(train_dataloader, val_dataloader):
    # Optuna optimization configurations
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

# Function for training the Temporal Fusion Transformer model
def train_model(train_dataloader, val_dataloader, params):
    # Setting up callbacks and model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        filename="best_model_{epoch}",
        dirpath="saved_models"
    )

    # Model training configurations
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=50,
        gpus=1 if torch.cuda.is_available() else 0,
        enable_model_summary=True,
        gradient_clip_val=params.get("gradient_clip_val", args.gradient_clip_val),
        limit_train_batches=30,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=params.get("learning_rate", args.learning_rate),
        hidden_size=params.get("hidden_size", args.hidden_size),
        attention_head_size=params.get("attention_head_size", args.attention_head_size),
        dropout=params.get("dropout", args.dropout),
        hidden_continuous_size=params.get("hidden_continuous_size", args.hidden_continuous_size),
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    # Model instantiation and fitting
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Post-training operations including best model selection and prediction
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    Trend_forecasting = raw_predictions[0][:, :, 3]
    print(Trend_forecasting)

    # Saving the trained model
    trainer.save_checkpoint("tft_model.ckpt")

# Main execution logic
if __name__ == "__main__":
    # Determine whether to run hyperparameter optimization or use predefined parameters
    if args.optimize_hyperparameters:
        best_params = optimize_hyperparams(train_dataloader, val_dataloader)
    # Execute the model training function with appropriate parameters
    else:
        best_params = {
            "gradient_clip_val": args.gradient_clip_val,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "hidden_continuous_size": args.hidden_continuous_size,
            "attention_head_size": args.attention_head_size,
            "learning_rate": args.learning_rate
        }

    train_model(train_dataloader, val_dataloader, best_params)
