#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/2 15:39
# @Author : LYX-夜光

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# 选择模型
def model_selection(model_name, **params):
    if model_name == 'xgboost':
        return XGBClassifier(**params)
    if model_name == 'lightgbm':
        return LGBMClassifier(**params)
    if model_name == 'catboost':
        return CatBoostClassifier(**params)
    if model_name == 'voting':
        return VotingClassifier(**params)
    if model_name == 'bagging':
        return BaggingClassifier(**params)
    if model_name == 'stacking':
        return StackingClassifier(**params)
    if model_name == 'dt':
        return DecisionTreeClassifier(**params)
    if model_name == 'lr':
        return LogisticRegression(**params)
    if model_name == 'svm':
        return SVC(**params)
    if model_name == 'rf':
        return RandomForestClassifier(**params)
    if model_name == 'knn':
        return KNeighborsClassifier(**params)
    return None
