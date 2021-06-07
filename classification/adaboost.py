"""
Adaboost builds on the idea that a series of weak classifiers
can be aggregated and form a strong classifier.

More specifically, let each weak classifier be H_k,
let the weightage assigned to each weak classifier be alpha_k,
then the strong classifier is just H = sum of (H_k * alpha_k)

To calculate the alphas, we assign each training data a weight, and
update the weights based on how well each weak classifier classifies the samples.

The formula for the kth weak classifier:
    error = sum of (w_i * 1[incorrect prediction])
    alpha_k = 0.5 * log((1-error)/error)

    updated weight = weight * exp{-yi * alpha_k * H_k(x)} / Z
    where Z is a normalizer: Z = sum of (updated weight)
"""

import math
import numpy as np
import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier
from util import get_input_label_split, get_accuracy, get_precision


class Adaboost:
    def __init__(self, n_estimators=200, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.train_x = None
        self.train_y = None
        self.trees = []
        self.alphas = []

    def create_tree(self, random_state=2021):
        return DecisionTreeClassifier(max_depth=self.max_depth, random_state=random_state)

    def calc_ek(self, weights, diff, method='linear'):
        # maxm of the difference
        abs_diff = abs(diff)
        E = abs_diff.max()
        if method == 'linear':
            return abs_diff / E
        if method == 'square':
            return (abs_diff / E) ** 2
        if method == 'exp':
            return 1. - np.exp(-abs_diff / E)
        raise Exception('method is not in {linear, square, exp}')

    def train(self, data, label_name, loss='linear'):
        self.train_x, self.train_y = get_input_label_split(data, label_name)

        size = self.train_x.shape[0]
        # initialize
        weights = np.ones(size) / size
        self.trees = [self.create_tree() for _ in range(self.n_estimators)]
        self.alphas = [0 for _ in range(self.n_estimators)]

        for i in range(self.n_estimators):
            self.trees[i].fit(self.train_x, self.train_y, sample_weight=weights)
            pred = self.trees[i].predict(self.train_x)
            ek = self.calc_ek(weights, self.train_y - pred, method=loss)
            error = np.dot(weights, ek)
            alpha = np.log((1. - error) / error)

            # update weights and normalize it
            temp = weights * np.exp(-alpha * (1. - ek))
            weights = temp / sum(temp)
            self.alphas[i] = alpha

    def infer(self, test_data):
        size = test_data.shape[0]
        pred = np.zeros(size)

        for i in range(self.n_estimators):
            cur_pred = self.trees[i].predict(test_data) * self.alphas[i]
            pred += cur_pred
        return np.sign(pred)


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
        ada = Adaboost()
        ada.train(df_train, args.label_name)

        test_x, test_y = get_input_label_split(df_test, args.label_name)
        pred = ada.infer(test_x)

        print(f"accuracy score: {get_accuracy(pred, test_y)}")
        print(f"precision score: {get_precision(pred, test_y)}")

    else:
        pass
