#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/5/18 11:49
# @Author : LYX-夜光
import pandas as pd

from initData import NEW_TRAIN_PATH, FEATURE_LIST
from utils import yaml_config
from utils.dataUtil import stratified_shuffle_split
from utils.trainUtil import bayes_search_train
from voting_model import Voting, voting_train
from utils.logUtil import get_rank_param
from utils.modelUtil import model_selection

if __name__ == "__main__":

    trainData = pd.read_csv(NEW_TRAIN_PATH, index_col=0)

    fold, seed = yaml_config['bys_param']['fold'], yaml_config['cus_param']['seed']

    model_name, model_param = yaml_config['model'][0]

    # 训练单个模型+RFE
    feature_list = FEATURE_LIST.copy()
    t_rounds = 22
    for _ in range(t_rounds):
        yaml_config['cus_param']['feature_list'] = feature_list
        X, y = trainData[feature_list].values, trainData['label'].values
        X, y = stratified_shuffle_split(X, y, n_splits=fold, random_state=seed)  # 随机分层排列

        model = None
        if model_name == "catboost":
            model = model_selection(model_name, **{'cat_features': list(range(len(feature_list)))})
        model = bayes_search_train(X, y, model_name, model_param, model=model)

        remove_feat = sorted(zip(feature_list, model.feature_importances_), key=lambda x: x[-1])[0][0]
        feature_list.remove(remove_feat)
        print("移除特征[%s]，剩下%d个特征..." % (remove_feat, len(feature_list)))

    # Voting集成
    X, y = trainData[FEATURE_LIST].values, trainData['label'].values
    X, y = stratified_shuffle_split(X, y, n_splits=fold, random_state=seed)  # 随机分层排列

    for model_num in [2, 3, 4, 5]:
        model_path_list, base_models, data_cols = [], [], []
        paramList = get_rank_param(model_name)[:model_num]
        for param in paramList:
            model_path_list.append(param['model_path'])
            data_col = [FEATURE_LIST.index(x) for x in param['cus_param']['feature_list']]
            data_cols.append(data_col)
            base_model = model_selection(model_name, **param['best_param_'])
            if model_name == "catboost":
                base_model.set_params(**{'cat_features': list(range(len(data_col)))})
            base_models.append(base_model)
        print("模型选择：", model_path_list)
        yaml_config['cus_param']['base_models'] = model_path_list
        ensemble_model = Voting(base_models=base_models, data_cols=data_cols, max_voting=200)

        voting_train(
            ensemble_model, X, y,
            cv=fold,
            n_jobs=yaml_config['cv_param']['workers'],
            verbose=4,
            random_state=seed,
        )
