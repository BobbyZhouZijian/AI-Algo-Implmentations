import numpy as np
import pandas as pd
from util import get_input_label_split, get_accuracy, get_precision, sigmoid
import argparse
import math


class LogisticRegression:
    def __init__(self, lr=1e-5, num_epochs=100):
        self.lr = lr
        self.num_epochs = num_epochs
        self.train_x = None
        self.train_y = None
        self.weights = None

        # for reproduction
        np.random.seed(seed=2021)

    def train(self, data, label_name):
        self.train_x, self.train_y = get_input_label_split(data, label_name)
        data_size = self.train_x.shape[0]
        weight_size = self.train_x.shape[1] + 1

        # initialize weights using Xavier initialization
        self.weights = np.random.randn(weight_size, 1) * math.sqrt(1.0 / weight_size)
        bias = np.ones((data_size, 1))
        train_x = np.mat(np.hstack((self.train_x, bias)))
        train_y = self.train_y.reshape((data_size, 1))

        # train for num_epochs epochs
        for i in range(self.num_epochs):
            a = train_x * self.weights
            o = sigmoid(a)
            diff = (train_y - o)
            self.weights = self.weights + self.lr * train_x.T * diff

    def infer(self, test_data, thres=0.5):
        data_size = test_data.shape[0]
        bias = np.ones((data_size, 1))
        test_data = np.hstack((test_data, bias))
        a = test_data * self.weights
        o = sigmoid(a)
        o = o > thres
        return o


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
        logisticReg = LogisticRegression()
        logisticReg.train(df_train, args.label_name)

        test_x, test_y = get_input_label_split(df_test, args.label_name)
        pred = logisticReg.infer(test_x)

        print(f"accuracy score: {get_accuracy(pred, test_y)}")
        print(f"precision score: {get_precision(pred, test_y)}")

    else:
        pass
