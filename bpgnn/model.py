"""THe module contains the main algorithm"""

import os
from argparse import Namespace
import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric
import torch_scatter
from torch import nn
from torch.nn import Module, ModuleList, Sequential
from torch.utils.data import DataLoader
from torch_geometric.nn import EdgeConv, GATConv, GCNConv


from bpgnn.layers import *


class BPGNN_Model(pl.LightningModule):
    """
    Runs the BPGNN algorithm with parameters specified by hparams.
    """

    def __init__(self, hparams):
        super(BPGNN_Model, self).__init__()

        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)

        # self.hparams=hparams
        self.save_hyperparameters(hparams)
        gnn_layers = hparams.gnn_layers
        mlp_layers = hparams.mlp_layers
        BP_layers = hparams.BP_layers
        k = hparams.k
        scheme = hparams.scheme
        # self.is_use_A_init = hparams.is_use_A_init
        self.graph_f = ModuleList()
        self.gnn = ModuleList()


        for i, (bp_l, gnn_l) in enumerate(zip(BP_layers, gnn_layers)):
            if len(bp_l) > 0:
                if "ffun" not in hparams or hparams.ffun == "gcn":
                    self.graph_f.append(
                        boolean_product(
                            GCNConv(bp_l[0], bp_l[-1]), k = k, distance=hparams.distance, method=scheme
                        )
                    )

            else:
                self.graph_f.append(Identity())

            if hparams.gnn == "edgeconv":
                gnn_l = gnn_l.copy()
                gnn_l[0] = gnn_l[0] * 2
                self.gnn.append(EdgeConv(MLP(gnn_l), hparams.pooling))
            if hparams.gnn == "gcn":
                self.gnn.append(GCNConv(gnn_l[0], gnn_l[1]))
            if hparams.gnn == "gat":
                self.gnn.append(GATConv(gnn_l[0], gnn_l[1]))

        self.fc = MLP(mlp_layers, final_activation=False)
        if hparams.pre_mlp is not None and len(hparams.pre_mlp) > 0:
            self.pre_mlp = MLP(hparams.pre_mlp, final_activation=True)
        self.avg_accuracy = None

        # torch lightning specific
        self.automatic_optimization = False
        self.debug = False

    def forward(self, x, edges):
        """
        Forward propagation of BPGNN

        :param x: node features,
        :param edges: graph structure.
        :return: predicted probabilities and probability of existing edges.
        """

        if self.hparams.pre_mlp is not None and len(self.hparams.pre_mlp) > 0:
            x = self.pre_mlp(x)

        graph_x = x.detach() 
        lprobslist = []
        A_init = edges.detach()

        for f, g in zip(self.graph_f, self.gnn):
            # construct the latent graph 
            graph_x, edges, lprobs, _ = f(graph_x, edges, None, A_init)
            b, n, d = x.shape

            # GNN module
            x = torch.nn.functional.relu(
                g(
                    torch.dropout(
                        x.view(-1, d), self.hparams.dropout, train=self.training
                    ),
                    edges
                )
            ).view(b, n, -1)
            graph_x = torch.cat([graph_x, x.detach()], -1)
            if lprobs is not None:
                lprobslist.append(lprobs)

        return self.fc(x), torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()

        X, y, mask, edges = train_batch
        edges = edges[0]

        assert X.shape[0] == 1  # only works in transductive setting
        mask = mask[0]

        pred, logprobs = self(X, edges)

        train_pred = pred[:, mask.to(torch.bool), :]
        train_lab = y[:, mask.to(torch.bool), :]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            train_pred, train_lab
        )
        loss.backward()

        correct_t = (
            (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()
        )

        # GRAPH LOSS
        if logprobs is not None:
            corr_pred = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().detach()

            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred) * 0.5

            point_w = (
                self.avg_accuracy - corr_pred
            )  # *(1*corr_pred + self.k*(1-corr_pred))
            graph_loss = point_w * logprobs[:, mask.to(torch.bool), :].exp().mean(
                [-1, -2]
            )


            graph_loss = graph_loss.mean()  # + self.graph_f[0].Pr.abs().sum()*1e-3

            graph_loss.backward()

            self.log("train_graph_loss", graph_loss.detach().cpu())
            if self.debug:
                self.point_w = point_w.detach().cpu()

            self.avg_accuracy = (
                self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred
            )

        optimizer.step()

        self.log("train_acc", correct_t)
        self.log("train_loss", loss.detach().cpu())

    def test_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]

        assert X.shape[0] == 1  # only works in transductive setting
        mask = mask[0]
        pred, _ = self(X, edges)
        pred = pred.softmax(-1)
        for i in range(1, self.hparams.test_eval):
            pred_, _ = self(X, edges)
            pred += pred_.softmax(-1)
        test_pred = pred[:, mask.to(torch.bool), :]
        test_lab = y[:, mask.to(torch.bool), :]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred, test_lab)
        self.log("test_loss", loss.detach().cpu())
        self.log("test_acc", 100 * correct_t)

    def validation_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]

        assert X.shape[0] == 1  # only works in transductive setting
        mask = mask[0]

        pred, _ = self(X, edges)
        pred = pred.softmax(-1)
        for i in range(1, self.hparams.test_eval):
            pred_, _ = self(X, edges)
            pred += pred_.softmax(-1)

        test_pred = pred[:, mask.to(torch.bool), :]
        test_lab = y[:, mask.to(torch.bool), :]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred, test_lab)

        self.log("val_loss", loss.detach())
        self.log("val_acc", 100 * correct_t)

