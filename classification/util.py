import pandas as pd
import numpy as np
import math

def get_input_label_split(train_data, label_name=None):
    if label_name == None:
        train = train_data.iloc[:,:].to_numpy()
        return train
    y = train_data[label_name].to_numpy()
    train = train_data.drop(columns=[label_name])
    train = train.iloc[:,:].to_numpy()
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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def calculate_entropy(df):
    instances = df.shape[0]
    columns = df.shape[1]

    decisions = df['Decision'].value_counts().keys().tolist()
    entropy = 0

    for i in range(0, len(dicisions)):
        decision = decisions[i]
        num_of_decisons = df['Decision'].value_counts().tolist()[i]

        class_probability = num_of_decisions / instances
        entropy = entropy - class_probability * math.log(class_probability, 2)
    return entropy


def discretize(df_col):
    '''
    Discretize a column if it contains more than 10 distinct values

    Returns:
        None if the column needs to be discretized
        the discretized column in a numpy array
    '''

    distinct = np.unique(df_col.to_numpy())

    if len(distinct) < 7:
        # if number of distinct elements is less than 7
        # do nothing and return false
        return None
    
    else:
        # get the mean, std, min and max of the df column
        mean = df_col.mean()
        std = df_col.std()
        minm = df_col.min()
        maxm = df_col.max()

        # sort values into 7 buckets
        scaler = [-3, -2, -1, 0, 1, 2, 3]
        values = []

        for i, scale in enumerate(scaler):
            if i == 0:
                values.append((float('-inf'), scale * std + mean))
            if i == len(scaler)-1:
                values.append((scale * std + mean, float('inf')))
            else:
                next_scale = scaler[i+1]
                values.append((scale * std + mean, next_scale * std + mean))
        
        # assign the values to the discretized intervals
        to_replace = np.zeros(len(df_col), dtype=tuple)
        for i in range(len(df_col)):
            cur_val = df_col.iloc[i]
            for v in values:
                if cur_val >= v[0] and cur_val <= v[1]:
                    to_replace[i] = v
                    break
        return to_replace
