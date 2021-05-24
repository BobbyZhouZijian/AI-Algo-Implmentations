'''
Naive Bayesian Classifier builds on the assumption that among
the different features of a sample data point x, each feature xi
are independent of each other. In another word,
P(x | t) = P(x1 | t) * P(x2 | t) * ... * P(xn | t)

Hence we can easily compute P(t | d) = P(d | t) * P(t) / P(d) = πP(xi | t) * P(t) / P(d)

Hence expected t = argmax P(t | d) = πP(xi | t) * P(t)
'''

import math
import numpy as np
import pandas as pd
import argparse
from util import get_input_label_split, get_accuracy, get_precision, discretize


class NBC:
    def __init__(self):
        self.train_data = None
        self.label = ''
        self.classes = set()
        self.prob = {}
        self.is_discrete = set()


    def train(self, data, label_name):
        self.train_data = data
        self.label = label_name
        size = self.train_data.shape[0]
        train_y = self.train_data[self.label].to_numpy()
        for y in train_y:
            self.classes.add(y)
        for y in self.classes:
            self.prob[y] = train_y[train_y == y].shape[0] / size

        # discretize columns for more accurate inference
        columns = self.train_data.columns
        for col in columns:
            if col == label_name:
                continue
            
            discretized_col = discretize(self.train_data[col])
            if discretized_col is not None:
                self.is_discrete.add(col)
                self.train_data[col] = discretized_col


    def infer(self, test_data):
        size = test_data.shape[0]
        pred = np.zeros(size)

        columns = test_data.columns

        for i in range(size):
            best_y = None
            best_prob = -1

            for y in self.classes:
                prob = 1.
                temp = self.train_data[self.train_data[self.label] == y]
                temp_size = len(temp)
                for col in columns:
                    cur_col = temp[col].to_numpy()
                    
                    test_val = test_data[col].iloc[i]
                    if col in self.is_discrete:
                        vals = np.unique(cur_col)
                        found = False
                        for v in vals:
                            if test_val >= v[0] and test_val <= v[1]:
                                test_val = v
                                found = True
                                break
                        if not found:
                            # dummy range that spans the entire R (or strictly speaking the entire FLOAT)
                            test_val = (float('-inf'), float('inf'))
                        cnt = 0
                        for val in cur_col:
                            if val == test_val:
                                cnt += 1
                        prob *= cnt / temp_size
                    else:
                        prob *= len(cur_col[cur_col == test_val]) / temp_size
                prob *= self.prob[y]

                if best_prob < prob:
                    best_y = y
                    best_prob = prob
            pred[i] = best_y
        
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True, help='training data file path')
    parser.add_argument('--label_name', type=str, default='label', help='label column name for the input file')
    parser.add_argument('--eval_mode', action='store_true', help='run this in evaluation mode')
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    if args.eval_mode:
        train_sz = int(len(df) * 0.8)
        df_train = df[:train_sz]
        df_test = df[train_sz:]
        nbc = NBC()
        nbc.train(df_train, args.label_name)

        test_x = df_test.drop(columns=[args.label_name])
        test_y = df_test[args.label_name].to_numpy()
        pred = nbc.infer(test_x)

        print(f"accuracy score: {get_accuracy(pred, test_y)}")
        print(f"precision score: {get_precision(pred, test_y)}")

    else:
        pass
    