#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/29 7:41
# @Author : LYX-夜光

import time

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import joblib
from joblib import Parallel, delayed

import itertools

from utils import set_seed, yaml_config, make_dir
from utils.logUtil import logging_config


def get_score_list(model, X, y, train_index, test_index, weights_list):
    model.fit(X[train_index], y[train_index])
    score_list = model.score_list(X[test_index], y[test_index], weights_list)
    return score_list

def find_best_voting_model(model, X, y, cv, n_jobs, verbose, max_voting, random_state):
    model_num = len(model.base_models)
    weights_list = [list(weights) for weights in itertools.product([0., 1.], repeat=model_num)][1:]
    random_num = max_voting - len(weights_list) if max_voting - len(weights_list) > 0 else 0
    if random_state:
        set_seed(random_state)
    weights_list += [np.random.rand(model_num).tolist() for _ in range(random_num)]
    # weights_list.append(None)

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    k_fold = KFold(n_splits=cv)
    score_lists = parallel(
        delayed(get_score_list)(model, X, y, train, test, weights_list) for train, test in k_fold.split(X, y))
    score_list = np.mean(score_lists, axis=0)
    best_index = score_list.argmax()
    for scores in score_lists:
        print('score:', scores[best_index])
    best_score = score_list[best_index]
    best_weights = weights_list[best_index]

    model.fit(X, y)
    model.weights = best_weights

    return model, best_score


def voting_train(model, X, y, cv, n_jobs=None, verbose=0, random_state=0, max_voting=200):
    model_dir, log_dir = yaml_config['dir']['model_dir'], yaml_config['dir']['log_dir']
    cus_param = yaml_config['cus_param']

    model, best_score = find_best_voting_model(model, X, y, cv, n_jobs, verbose, max_voting, random_state)

    make_dir(model_dir)
    model_path = model_dir + '/%s-%s.model' % (model.model_name, int(time.time()))
    joblib.dump(model, model_path)

    # 配置日志文件
    make_dir(log_dir)
    logger = logging_config(model.model_name, log_dir + '/%s.log' % model.model_name)
    log_message = {
        "cus_param": cus_param,
        "best_param_": {"weights": model.weights},
        "best_score_": best_score,
        "train_score": model.score(X, y),
        "model_path": model_path,
    }
    logger.info(log_message)

    return model


class Voting(BaseEstimator):
    def __init__(self, base_models, data_cols=None, max_voting=100):
        self.model_name = "voting"

        self.base_models = base_models
        self.data_cols = data_cols
        self.max_voting = max_voting
        self.model_num = len(self.base_models)
        self.weights = None

    def fit(self, X, y):
        if self.data_cols is None:
            self.data_cols = [list(range(X.shape[1]))] * self.model_num
        # 训练base_models
        for i in range(self.model_num):
            self.base_models[i].fit(X[:, self.data_cols[i]], y)

    def predict_proba(self, X):
        # base_models预测概率
        y_prob_list = []
        for i in range(self.model_num):
            y_prob_list.append(self.base_models[i].predict_proba(X[:, self.data_cols[i]]))
        y_prob_list = np.array(y_prob_list)
        # voting集成预测概率
        return self.voting(y_prob_list, self.weights)

    def predict(self, X, y_prob=None):
        if y_prob is None:
            y_prob = self.predict_proba(X)
        return y_prob.argmax(1)

    def score(self, X, y, y_prob=None):
        y_pred = self.predict(X, y_prob)
        return accuracy_score(y, y_pred)

    def score_list(self, X, y, weights_list):
        # base_models预测概率
        y_prob_list = []
        for i in range(self.model_num):
            y_prob_list.append(self.base_models[i].predict_proba(X[:, self.data_cols[i]]))
        y_prob_list = np.array(y_prob_list)
        # 计算score
        score_list = []
        for weights in weights_list:
            y_prob = self.voting(y_prob_list, weights)
            score_list.append(self.score(X, y, y_prob))
        return score_list

    def voting(self, y_pred_proba_list, weights):
        y_prob = sum([prob * w for prob, w in zip(y_pred_proba_list, weights)]) / sum(weights)
        return np.array(y_prob)