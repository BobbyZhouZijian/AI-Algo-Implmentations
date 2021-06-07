'''
Implmentation of Gradient Boosting Decision Tree involves using
many decision trees as weak learners. To achieve high efficiency,
we use the sklearn implementation of decision trees.

Given a set of training data xi, yi, i=1,2,3,...,n, the resultant model at
the (m-1)th round is Fm-1(x). With the mth weak learner h(x), we have

Fm = Fm-1 + argmin Loss(yi, Fm-1(xi)+h(xi))

gm = - (Loss(y,Fm-1)/Fm-1)' i.e. the negative gradients

'''

import numpy as np
import pandas as pd
import argparse
from sklearn.tree import DecisionTreeRegressor
from util import get_input_label_split, get_accuracy, get_precision, sigmoid


class GBDT:
    def __init__(self, n_estimators=300, max_depth=5, lr=0.1):
        self.estimator_list = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.train_x = None
        self.train_y = None
        self.F = None

    def train(self, data, label_name):
        self.train_x, self.train_y = get_input_label_split(data, label_name)
        self.F = np.zeros_like(self.train_y, dtype=float)

        # boosting
        for _ in range(self.n_estimators):
            # get negative grads
            neg_grads = self.train_y - sigmoid(self.F)
            base = DecisionTreeRegressor(max_depth=self.max_depth)
            base.fit(self.train_x, neg_grads)
            train_preds = base.predict(self.train_x)
            self.estimator_list.append(base)

            if _ == 0:
                self.F = train_preds
            else:
                self.F += self.lr * train_preds

    def infer(self, data, thres=0.5):
        size = data.shape[0]
        pred = np.zeros(size, dtype=float)
        for i, est in enumerate(self.estimator_list):
            if i == 0:
                pred += est.predict(data)
            else:
                pred += self.lr * est.predict(data)
        # broadcasting
        pred = pred > thres
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
        gbdt = GBDT()
        gbdt.train(df_train, args.label_name)

        test_x, test_y = get_input_label_split(df_test, args.label_name)
        pred = gbdt.infer(test_x)

        print(f"accuracy score: {get_accuracy(pred, test_y)}")
        print(f"precision score: {get_precision(pred, test_y)}")

    else:
        pass
