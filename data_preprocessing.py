
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

data = pd.read_excel('your file')
data["year"] = data["year"].astype(str)
data["day"] = data["day"].astype(str)
data["month"] = data["month"].astype(str)
data['holiday'] = data['holiday'].astype(str)
data["tourist"] = data["tourist"].astype("float64")
data["Trend"] = data["Trend"].astype("float64")
data["Seasonal"] = data["Seasonal"].astype("float64")
data["Resid"] = data["Resid"].astype("float64")
