#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/3/31 15:28
# @Author : LYX-夜光
import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from utils import make_dir, yaml_config, write_json

RAWDATA_DIR = "./raw_data"
TRAIN_PATH = RAWDATA_DIR + "/train.csv"
TEST_PATH = RAWDATA_DIR + "/test.csv"

NEWDATA_DIR = "./new_data"
NEW_TRAIN_PATH = NEWDATA_DIR + "/new_train.csv"
NEW_TEST_PATH = NEWDATA_DIR + "/new_test.csv"

FEATURE_DIR = "./feature_dict"

# 构建特征，特征名称后缀
class CF:
    origin = ''
    count_filter = '_count_filter'
    count_bi = '_count_bi'
    minute = '_minute'
    minute_count = '_minute_count'
    count_by_minute = '_count_by_minute'

# None为舍弃特征
FEATURE_PATTERN = {
    # 'sid': None,

    'android_id': [(CF.origin, []), (CF.count_filter, [20]), (CF.count_bi, [1])],
    'media_id': [(CF.origin, []), (CF.count_bi, [1000])],
    'apptype': [(CF.origin, [])],
    'package': [(CF.origin, []), (CF.count_filter, [20]), (CF.count_bi, [150])],
    'version': [(CF.origin, []), (CF.count_bi, [13000])],

    'ntt': [(CF.origin, [])],
    'carrier': [(CF.origin, [])],

    'os': [(CF.origin, [])],
    'osv': [(CF.origin, []), (CF.count_bi, [2000])],
    'dev_height': [(CF.origin, []), (CF.count_bi, [360])],
    'dev_ppi': [(CF.origin, [])],
    'dev_width': [(CF.origin, []), (CF.count_bi, [850])],
    'lan': [(CF.origin, [])],

    'location': [(CF.origin, [])],
    'fea_hash': [(CF.origin, []), (CF.count_filter, [20]), (CF.count_bi, [1])],
    'fea1_hash': [(CF.origin, []), (CF.count_filter, [20]), (CF.count_bi, [60])],
    'cus_type': [(CF.origin, [])],

    'timestamp': [(CF.minute, [5, 10, 30, 60, 120, 360]), (CF.minute_count, [1, 2, 5, 10, 30])],
}

# # None为舍弃特征
# FEATURE_PATTERN = {
#     # 'sid': None,
#
#     'android_id': [(CF.origin, [])],
#     'media_id': [(CF.origin, [])],
#     'apptype': [(CF.origin, [])],
#     'package': [(CF.origin, [])],
#     'version': [(CF.origin, [])],
#
#     'ntt': [(CF.origin, [])],
#     'carrier': [(CF.origin, [])],
#
#     'os': [(CF.origin, [])],
#     'osv': [(CF.origin, [])],
#     'dev_height': [(CF.origin, [])],
#     'dev_ppi': [(CF.origin, [])],
#     'dev_width': [(CF.origin, [])],
#     'lan': [(CF.origin, [])],
#
#     'location': [(CF.origin, [])],
#     'fea_hash': [(CF.origin, [])],
#     'fea1_hash': [(CF.origin, [])],
#     'cus_type': [(CF.origin, [])],
# }

FEATURE_LIST = []
for feat in FEATURE_PATTERN:
    for suffix, valueList in FEATURE_PATTERN[feat]:
        if suffix == CF.origin:
            FEATURE_LIST.append(feat)
        else:
            for value in valueList:
                FEATURE_LIST.append(feat + suffix + '_%s' % value)

MISSING_VALUE = {
    'android_id': 0,
    'package': 0,
    'version': '0',
    'carrier': -1.0,
    'osv': np.nan,
    'dev_height': 0.0,
    'dev_ppi': 0.0,
    'dev_width': 0.0,
    'lan': np.nan,
    'location': -1,
    'fea_hash': '0',
}

# # 统计特征值数量
# def featCount(trainData, testData, feat):
#     data = trainData.groupby(feat)[feat].count()
#     trainData[feat + '_count'] = trainData[feat].map(data.to_dict()).fillna(1)
#     testData[feat + '_count'] = testData[feat].map(data.to_dict()).fillna(1)
#     return trainData, testData
#
# # 时间戳转换
# def createTime(dataset, feat):
#     timeList = []
#     for value in dataset[feat]:
#         time_array = datetime.datetime.fromtimestamp(value / 1000)
#         timeList.append(time_array.day*24*60*60 + time_array.hour*60*60 + time_array.minute*60 + time_array.second)
#     dataset[feat] = timeList
#     return dataset
#
# # 分桶计数
# def bucketing_count(dataset, trainData, testData, feat, bucket_count, newFeatName):
#     trainData, testData = bucketing(dataset, trainData, testData, feat, bucket_count, newFeatName)
#     data = trainData[newFeatName].append(testData[newFeatName])
#     data = data.groupby(data).count().to_dict()
#     trainData[newFeatName] = trainData[newFeatName].map(data).fillna(1)
#     testData[newFeatName] = testData[newFeatName].map(data).fillna(1)
#     return trainData, testData
#
# # 分桶
# def bucketing(dataset, trainData, testData, feat, bucket, newFeatName):
#     value_min, value_max = dataset[feat].min(), dataset[feat].max()
#     bucket_len = (value_max - value_min) / bucket
#     bucket_list = [value_min + i*bucket_len for i in range(bucket+1)]
#     trainData[newFeatName] = pd.cut(trainData[feat], bins=bucket_list, include_lowest=True, labels=False)
#     testData[newFeatName] = pd.cut(testData[feat], bins=bucket_list, include_lowest=True, labels=False)
#     return trainData, testData
#
# # 按照特征值占label的比例分桶
# def featValueProb(trainData, testData, feat, split):
#     featValue = trainData['label'].groupby(trainData[feat]).apply(lambda x: sum(x == 1)/len(x))
#     featValue = pd.cut(featValue, bins=[0, split, 1-split, 1], include_lowest=True, labels=False).to_dict()
#     trainData[feat + '_prob'] = trainData[feat].map(featValue).fillna(1)
#     testData[feat + '_prob'] = testData[feat].map(featValue).fillna(1)
#     return trainData, testData

# 按照一段时间计数
def countByMintue(trainData, testData, feat, minutes, newFeatName):
    trainData, testData = timeSegment(trainData, testData, 'timestamp', minutes, newFeatName)
    data = trainData.append(testData).groupby([newFeatName, feat])[newFeatName].transform('count')
    trainData[newFeatName] = data.iloc[:len(trainData)]
    testData[newFeatName] = data.iloc[len(trainData):]
    return trainData, testData

# 时间按照分钟分段
def timeSegment(trainData, testData, feat, minutes, newFeatName):
    trainData[newFeatName] = list(map(lambda x: int(x / 1000 / 60 / minutes), trainData[feat]))
    testData[newFeatName] = list(map(lambda x: int(x / 1000 / 60 / minutes), testData[feat]))
    return trainData, testData

# 按分钟分段再计数
def timeSegmentCount(trainData, testData, feat, minutes, newFeatName):
    trainData, testData = timeSegment(trainData, testData, feat, minutes, newFeatName)
    data = trainData[newFeatName].append(testData[newFeatName])
    data = data.groupby(data).count().to_dict()
    trainData[newFeatName] = trainData[newFeatName].map(data).fillna(1)
    testData[newFeatName] = testData[newFeatName].map(data).fillna(1)
    return trainData, testData

# 计数编码
def countEncoder(trainData, testData, feat, count_bi, newFeatName):
    data = trainData.append(testData).groupby(feat)[feat].count()
    data[data <= count_bi] = 1
    data[data > count_bi] = 2
    valueDict = data.to_dict()
    if feat in MISSING_VALUE and MISSING_VALUE[feat] is not np.nan:
        valueDict.update({MISSING_VALUE[feat]: 0})
    trainData[newFeatName] = trainData[feat].map(valueDict)
    trainData[newFeatName] = trainData[newFeatName].fillna(0)
    testData[newFeatName] = testData[feat].map(valueDict)
    testData[newFeatName] = testData[newFeatName].fillna(0)
    return trainData, testData

# 转换大离散特征
def transDiscrete(trainData, testData, feat, count_filter, newFeatName):
    valueDict = {value: str(value) for value in set(trainData[feat][~trainData[feat].isna()])}
    data = trainData.groupby(feat)[feat].count()
    removeValue = -2
    data[data <= count_filter] = removeValue
    valueDict.update(data.loc[data == removeValue].to_dict())
    trainData[newFeatName] = trainData[feat].map(valueDict)
    trainData[newFeatName] = trainData[newFeatName].fillna(removeValue)
    testData[newFeatName] = testData[feat].map(valueDict)
    testData[newFeatName] = testData[newFeatName].fillna(removeValue)
    return trainData, testData

# 构建新的数据集
def createNewDataset(trainData, testData):
    # 转换特征
    for feat in FEATURE_PATTERN:
        del_origin = True
        for suffix, valueList in FEATURE_PATTERN[feat]:
            if suffix == CF.origin:
                del_origin = False
                continue
            for value in valueList:
                newFeatName = feat + suffix + '_%s' % value
                if suffix == CF.count_filter:
                    trainData, testData = transDiscrete(trainData, testData, feat, value, newFeatName)
                elif suffix == CF.count_bi:
                    trainData, testData = countEncoder(trainData, testData, feat, value, newFeatName)
                elif suffix == CF.minute:
                    trainData, testData = timeSegment(trainData, testData, feat, value, newFeatName)
                elif suffix == CF.minute_count:
                    trainData, testData = timeSegmentCount(trainData, testData, feat, value, newFeatName)
                elif suffix == CF.count_by_minute:
                    trainData, testData = countByMintue(trainData, testData, feat, value, newFeatName)
        if del_origin:
            del trainData[feat]
            del testData[feat]

    # 预处理特征
    dataset = trainData.append(testData)
    encoder = LabelEncoder()
    sizeDict = {}
    for feat in FEATURE_LIST:
        # 离散特征编码
        encoder.fit(list(set(dataset[feat])))
        trainData[feat] = encoder.transform(list(trainData[feat]))
        testData[feat] = encoder.transform(list(testData[feat]))

        print("转换离散特征[%s]：共有%d个值" % (feat, len(set(dataset[feat]))))
        sizeDict[feat] = len(set(dataset[feat]))

    # make_dir(FEATURE_DIR)
    # write_json(FEATURE_DIR + "/size.dict", sizeDict)

    return trainData, testData


if __name__ == "__main__":
    # 创建新数据集文件夹
    make_dir(NEWDATA_DIR)
    # 预处理特征，创建新数据集
    trainData = pd.read_csv(TRAIN_PATH, index_col=0)
    testData = pd.read_csv(TEST_PATH, index_col=0)
    trainData, testData = createNewDataset(trainData, testData)

    # 保存csv文件
    csv = pd.DataFrame(trainData)
    csv.to_csv(NEW_TRAIN_PATH)
    print("新训练集创建完成！")

    csv = pd.DataFrame(testData)
    csv.to_csv(NEW_TEST_PATH)
    print("新测试集创建完成！")

    pass