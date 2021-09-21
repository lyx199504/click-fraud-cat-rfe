#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/5/18 11:49
# @Author : LYX-夜光
import pandas as pd

from initData import NEW_TRAIN_PATH, FEATURE_LIST
from utils import yaml_config
from utils.dataUtil import stratified_shuffle_split
from utils.trainUtil import cv_train
from utils.modelUtil import model_selection

if __name__ == "__main__":


    trainData = pd.read_csv(NEW_TRAIN_PATH, index_col=0)

    fold, seed = yaml_config['cv_param']['fold'], yaml_config['cus_param']['seed']

    model_name_list = ['knn', 'lr', 'dt', 'rf', 'xgboost', 'lightgbm', 'catboost']

    for model_name in model_name_list:
        # 训练单个模型
        feature_list = FEATURE_LIST.copy()
        yaml_config['cus_param']['feature_list'] = feature_list
        yaml_config['cus_param']['model_name'] = model_name
        X, y = trainData[feature_list].values, trainData['label'].values
        X, y = stratified_shuffle_split(X, y, n_splits=fold, random_state=seed)  # 随机分层排列

        model = model_selection(model_name)
        if model_name == "catboost":
            model.set_params(**{
                'eval_metric': 'Accuracy',
                'iterations': 200,
                'cat_features': list(range(len(feature_list)))
            })
        cv_train(X, y, model_name='raw_ml', model=model)
