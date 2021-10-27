#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/27 11:00
# @Author : LYX-夜光

import pandas as pd

from baseline_model import BaselineNet
from initData import NEW_TRAIN_PATH, FEATURE_LIST
from utils import yaml_config
from utils.dataUtil import stratified_shuffle_split

if __name__ == "__main__":
    trainData = pd.read_csv(NEW_TRAIN_PATH, index_col=0)

    fold, seed = yaml_config['cv_param']['fold'], yaml_config['cus_param']['seed']

    yaml_config['cus_param']['feature_list'] = FEATURE_LIST

    X, y = trainData[FEATURE_LIST].values, trainData['label'].values
    X, y = stratified_shuffle_split(X, y, n_splits=fold, random_state=seed)  # 随机分层排列

    model = BaselineNet(learning_rate=0.001, batch_size=500, epochs=100, random_state=seed)
    model.param_search = False
    model.save_model = True
    train_point = int(len(X) / fold)
    model.fit(X[train_point:], y[train_point:], X[:train_point], y[:train_point])