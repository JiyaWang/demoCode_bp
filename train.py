'''This file contains the main run function and setting of hyperparams'''

import sys
sys.path.insert(0, "./keops")
import os
os.environ["USE_KEOPS"] = "True"
import random
from argparse import ArgumentParser
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from data_loader import AmazonDataset, PlanetoidDataset
from bpgnn.model import BPGNN_Model


def run_training_process(run_params):
    '''Run function for training the model'''

    # Load datasets and construct data module.
    global val_data
    train_data = None
    test_data = None

    if run_params.dataset in ["Photo"]:
        train_data = AmazonDataset(split="train", name=run_params.dataset, device="cpu")
        val_data = AmazonDataset(
            split="val", name=run_params.dataset, samples_per_epoch=1
        )
        test_data = AmazonDataset(
            split="test", name=run_params.dataset, samples_per_epoch=1
        )

    if run_params.dataset in ["Cora", "CiteSeer", "PubMed"]:
        train_data = PlanetoidDataset(
            split="train", name=run_params.dataset, device="cuda"
        )
        val_data = PlanetoidDataset(
            split="val", name=run_params.dataset, samples_per_epoch=1
        )
        test_data = PlanetoidDataset(
            split="test", name=run_params.dataset, samples_per_epoch=1
        )


    if train_data is None:
        raise Exception("Dataset %s not supported" % run_params.dataset)

    train_loader = DataLoader(train_data, batch_size=1, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    class MyDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

        def test_dataloader(self):
            return test_loader

    # configure input feature size.
    if run_params.pre_mlp is None or len(run_params.pre_mlp) == 0:
        if len(run_params.BP_layers[0]) > 0:
            run_params.BP_layers[0][0] = train_data.n_features
        run_params.gnn_layers[0][0] = train_data.n_features
    else:
        run_params.pre_mlp[0] = train_data.n_features
    run_params.mlp_layers[-1] = train_data.num_classes

    model = BPGNN_Model(run_params)

    # Set checkpoint and early stopping.
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=15, verbose=False, mode="min"
    )
    callbacks = [checkpoint_callback, early_stop_callback]

    if val_data == test_data:
        callbacks = None

    trainer = pl.Trainer.from_argparse_args(
        run_params, callbacks=callbacks)
    
    # fit the model and test
    trainer.fit(model, datamodule=MyDataModule())
    trainer.test(model, datamodule=MyDataModule())




if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args(
        ["--accelerator", "gpu",
        "--devices", "1",
        "--log_every_n_steps", "100",
        "--max_epochs", "100",
        "--check_val_every_n_epoch", "1",]
    )

    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--gnn", default="gcn", help = "aggregate function in gnn module including gcn, gat and edgeconv")
    parser.add_argument("--gnn_layers", default=[[32, 32], [32, 16], [16, 8]], type=lambda x: eval(x), help = "output sizes of GNN ")

    parser.add_argument("--ffun", default="gcn", help = "feature aggregated functions for preprocessing features in latent graph inference module")
    parser.add_argument("--BP_layers", default=[[32, 4], [], []], type=lambda x: eval(x), help = "output feature sizes of feature aggregated functions")
    parser.add_argument("--k", default=5, type=int, help = "the degree of latent graph")
    parser.add_argument("--scheme", default="sparse", help = "the scheme of Boolean product")
    parser.add_argument("--distance", default="euclidean", help = " latent graph embedding space")

    parser.add_argument("--mlp_layers", default=[8, 8, 3], type=lambda x: eval(x), help = "output feature sizes of mlp for classification")
    parser.add_argument("--pre_mlp", default=[-1, 32], type=lambda x: eval(x), help = "output feature sizes of mlp for preprocessing")

    # parser.add_argument("--is_use_A_init", default=True, type=lambda x: eval(x), help = "")

    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--pooling", default="add", help = " the aggregation scheme in edgeconv")
    parser.add_argument("--lr", default=5e-3, type=float, help = "learning rate")
    parser.add_argument("--test_eval", default=10, type=int, help = "times of running on the test datasets")

    parser.set_defaults(default_root_path="./log")
    params = parser.parse_args(namespace=params)

    run_training_process(params)

