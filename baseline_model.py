#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/27 17:18
# @Author : LYX-夜光

import torch
from torch import nn

from initData import FEATURE_DIR, FEATURE_LIST
from utils import read_json
from utils.pytorchModel import DLClassifier

size_dict = read_json(FEATURE_DIR + "/size.dict")

class BaselineNet(DLClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super(BaselineNet, self).__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "baseline"

    def create_model(self):
        self.hidden_layers_list = nn.ModuleList([])
        for feat in FEATURE_LIST:
            hidden_layer = nn.Sequential(
                nn.Embedding(num_embeddings=size_dict[feat], embedding_dim=100),
                nn.Linear(in_features=100, out_features=16)
            )
            self.hidden_layers_list.append(hidden_layer)
        self.out_layer = nn.Linear(in_features=len(FEATURE_LIST)*16, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        H = []
        for x, hidden_layer in zip(X.T, self.hidden_layers_list):
            H.append(hidden_layer(x))
        y = torch.cat(H, dim=1)
        y = self.out_layer(y)
        y = self.softmax(y)
        return y