
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

from dataset_creation import training, train_dataloader, val_dataloader

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_last=True,
    save_top_k=1,
    filename="best_model_{epoch}",
    dirpath="saved_models",
)
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=100,
    gpus=1,
    enable_model_summary=True,
    gradient_clip_val=0.31330228284676426,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0011749674283127688,
    hidden_size=15,
    attention_head_size=2,
    dropout=0.2757684537904421,
    hidden_continuous_size=10,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
