import pandas as pd
import numpy as np
import math


def get_input_label_split(train_data, label_name=None):
    if label_name is None:
        train = train_data.iloc[:, :].to_numpy()
        return train
    y = train_data[label_name].to_numpy()
    train = train_data.drop(columns=[label_name])
    train = train.iloc[:, :].to_numpy()
    return train, y


def get_accuracy(pred, y, thres=0.5):
    if len(pred) != len(y):
        raise Exception(f"size of pred is inconsistent with y. Expected pred \
            to have size {len(y)} but got {len(pred)}")

    total = len(pred)
    acc_cnt = 0

    for i in range(total):
        cur_pred = 1 if pred[i] > thres else 0
        if cur_pred == y[i]:
            acc_cnt += 1

    return acc_cnt / total


def get_precision(pred, y, thres=0.5):
    if len(pred) != len(y):
        raise Exception(f"size of pred is inconsistent with y. Expected pred \
            to have size {len(y)} but got {len(pred)}")

    total = 0
    acc_cnt = 0

    for i in range(len(pred)):
        if y[i] == 0:
            continue
        total += 1
        cur_pred = 1 if pred[i] > thres else 0
        if cur_pred == y[i]:
            acc_cnt += 1

    return acc_cnt / total
