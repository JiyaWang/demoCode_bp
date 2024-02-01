'''This file contains the latent graph inference module, MLP module and other utility functions'''

import torch
import math
import numpy as np
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_adj
from torch.nn import Module, ModuleList, Sequential
from torch import nn

#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
    return dist

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1 
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)
    
    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x

class boolean_product(nn.Module):
    """
    Graph residual connection block based on Boolean Product.
    This class includes generating probabilty graph,computing Boolean product between probability graph and the original graph and sampling based on gumble-top-k.
    """
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, method = 'sparse', sparse=True):
        super(boolean_product, self).__init__()
        self.sparse = sparse
        self.temperature = nn.Parameter(torch.tensor(1. if distance == "hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k
        self.method = method
        self.debug = False

        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances

    def forward(self, x, A, fixedges=None, A_init=None):
        x = self.embed_f(x, A)
        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1] // self.k, self.k,
                                                dtype=torch.float, device=x.device)
            # compute the distance of features between pairs of nodes
            D = self.distance(x)
            
            # to contruct the latent graph 
            edges_hat, logprobs = self.booleanProduct_and_sample(D, A_init)


        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1] // self.k, self.k,
                                                    dtype=torch.float, device=x.device)
                D = self.distance(x)

                edges_hat, logprobs = self.booleanProduct_and_sample(D, A_init)


        if self.debug:
            self.D = D
            self.edges_hat = edges_hat
            self.logprobs = logprobs
            self.x = x

        return x, edges_hat, logprobs, A_init

    def booleanProduct_and_sample(self, logits, A_init):
        """
        perfom Boolean product and sample from probability graph.

        :param logits: distance of pairs of features. 
        :param A_init: the original graph structure.
        :return: - edges: the latent graph (edge_index),
                 - lprobs: the log probability of pairs of exsiting nodes on the latent graph.
        """
        b, n, _ = logits.shape

        log_prob = logits * torch.exp(torch.clamp(self.temperature, -5, 5))  # dim:b*n*n
        
        if A_init is not None:
            if self.method == 'sparse':
                probs = self.prob_boolean_product_sparse(log_prob, A_init)
            if self.method == 'dense':
                probs = self.prob_boolean_product_dense(log_prob, A_init)
        else:
            probs = log_prob

        q = torch.rand_like(logits) + 1e-8  # dim: b*n*n
        lq = (probs - torch.log(-torch.log(q)))  # dim: b*n*n

        lprobs, indices = torch.topk(-lq, self.k)  # dim:b*n*k

        rows = torch.arange(n).view(1, n, 1).to(logits.device).repeat(b, 1, self.k)
        edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)

        if self.sparse:
            return (edges + (torch.arange(b).to(logits.device) * n)[:, None, None]).transpose(0, 1).reshape(2,
                                                                                                            -1), lprobs
        return edges, lprobs


    def prob_boolean_product_dense(self, log_prob, graph_init):
        """
        Boolean product between probability graph and the original graph (dense version).

        :param log_prob: log probability of feature graph. 
        :param A_init: the original graph structure.
        :return: a merge graph.
        """
        b, n, _ = log_prob.shape

        prob = (-log_prob).exp()

        graph_init = to_dense_adj(graph_init).view(b, n, -1)
        count = graph_init.sum(dim=-2, keepdim=True) 


        merge_matrix_1 = (torch.matmul(prob, graph_init).clamp_(min=1e-24)).log() - (count.clamp_(min=1e-8)).log()
        merge_matrix_2 = torch.transpose(merge_matrix_1, dim0=-1, dim1=-2)

        # Guarantee symmetry (the latent graph is undirected graph)
        merge_matrix = torch.logsumexp(torch.cat([merge_matrix_1.unsqueeze(-1), merge_matrix_2.unsqueeze(-1)], dim=-1),
                                       dim=-1) - math.log(2.) 
        
        return -merge_matrix



    def prob_boolean_product_sparse(self, log_prob, graph_init):
        """
        Boolean product between probability graph and the original graph (sparse version).

        :param log_prob: log probability of feature graph. 
        :param A_init: the original graph structure.
        :return: a merge graph.
        """
        b, n, _ = log_prob.shape

        log_prob=log_prob.squeeze()

        prob= (-log_prob).exp()
        
        graph_init = to_dense_adj(graph_init.detach()).squeeze()

        count = torch.sum(graph_init,dim=-1,keepdim=True) 

        graph_init = graph_init.to_sparse()

        merge_matrix_1 = (torch.sparse.mm(graph_init,prob).clamp_(min = 1e-24)).log()-(count.clamp_(min=1e-8)).log()
        merge_matrix_2 = torch.transpose(merge_matrix_1,dim0=-1,dim1=-2)
        
        merge_matrix=  torch.logsumexp(torch.cat([merge_matrix_1.unsqueeze(-1), merge_matrix_2.unsqueeze(-1)], dim=-1),dim=-1) - math.log(2.)

        return -merge_matrix.unsqueeze(0)
 
 
 # MLP module
class MLP(nn.Module): 
    def __init__(self, layers_size,final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for li in range(1,len(layers_size)):
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li-1],layers_size[li]))
            if li==len(layers_size)-1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))
            
            
        self.MLP = nn.Sequential(*layers)
        
    def forward(self, x, e=None):
        x = self.MLP(x)
        return x
    
# identity mapping
class Identity(nn.Module):
    def __init__(self,retparam=None):
        self.retparam=retparam
        super(Identity, self).__init__()
        
    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params
    