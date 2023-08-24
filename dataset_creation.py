
import os
import warnings
import tensorflow as tf
import tensorboard as tb
import copy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MAE, MAPE, RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pathlib import Path
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint

from data_preprocessing import data

max_prediction_length = 1
max_encoder_length = 30

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= 783],
    time_idx="time_idx",
    target="Seasonal",
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_categoricals=["month", "weekday", "day", "holiday"],
    time_varying_known_reals=["time_idx", "pc", "mob"],
    time_varying_unknown_reals=["Seasonal"],
    group_ids=['destination'],
    target_normalizer=GroupNormalizer(groups=['destination'], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
batch_size = 128

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)
