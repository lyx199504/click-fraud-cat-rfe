#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/5/20 18:29
# @Author : LYX-夜光

import joblib
import numpy as np
import pandas as pd

from initData import NEW_TEST_PATH, FEATURE_LIST
from utils import make_dir
from utils.logUtil import get_param_from_log

if __name__ == "__main__":

    testData = pd.read_csv(NEW_TEST_PATH, index_col=0)

    model_key = "raw_ml-1626094153"
    clf = joblib.load('./model/%s.model' % model_key)

    feature_list = FEATURE_LIST
    log_param = get_param_from_log(model_key.split('-')[0], model_key)
    if 'feature_list' in log_param['cus_param']:
        feature_list = log_param['cus_param']['feature_list']
    X = testData[feature_list].values
    labelList = clf.predict(X)

    # 保存csv文件
    RESULT_DIR = "./results"
    make_dir(RESULT_DIR)
    result_path = RESULT_DIR + '/results_%s.csv' % model_key
    results = testData.loc[:, "sid"]
    results = pd.DataFrame({"sid": np.array(results, dtype="int64"), "label": labelList})
    results.to_csv(result_path, index=False)
    print("结果文件保存至：", result_path)
