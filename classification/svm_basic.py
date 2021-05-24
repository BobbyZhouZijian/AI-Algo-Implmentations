'''
SVM builds on the idea of maximising the width
of the street between the positive sets of points 
and the negative sets of points.

This maximisation can be achieved by minimising the following loss function:

L = 1/2 ||w||^2 + C / N * sum (max(0, 1 - yi(w * xi + b)))

To minimise this L, we only need to adjust the value of dot(xi, xj)
'''

import numpy as np
import pandas as pd
import argparse
from util import get_input_label_split, get_accuracy, get_precision


class SVM:
    def __init__(self, C=1.0, lr=5e-3):
        self.C = C
        self.lr = lr
        self.train_x = None
        self.train_y = None
        self.weights = None
        self.bias = None


    def grad_descent(self, x, y):

        '''
        Gradient of the loss function 
        grad = 1/N * sum(w if 1-y*f(x) < 0, w - C*yi*xi otherwise)
        
        if yi * f(x) >= 1
        dL/dw = 2 * w
        dL/db = 0

        else
        dL/dw = 2 * w - C * yi * xi
        dL/db = yi
        '''
        
        is_classified = y * (np.dot(x, self.weights) - self.bias) >= 1.0
        if is_classified:
            self.weights -= self.lr * 2.0 * self.weights
        else:
            self.weights -= self.lr * (2.0 * self.weights - self.C * y * x)
            self.bias -= self.lr * self.C * y



    def train(self, data, label_name, num_epochs=5000):
        self.train_x, self.train_y = get_input_label_split(data, label_name)
        data_size, num_features = self.train_x.shape

        # initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # convert y labels to (1,-1) instead of (1,0)
        train_y = np.where(self.train_y==0, -1.0, 1.0)
        
        for _ in range(num_epochs):
            for i in range(data_size):
                self.grad_descent(self.train_x[i], train_y[i])


    def infer(self, data):
        size = data.shape[0]
        res = np.zeros(size)
        for i in range(size):
            res[i] = np.dot(data[i], self.weights) - self.bias
        return np.sign(res)


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
        svm = SVM()
        svm.train(df_train, args.label_name)

        test_x, test_y = get_input_label_split(df_test, args.label_name)
        pred = svm.infer(test_x)

        print(f"accuracy score: {get_accuracy(pred, test_y)}")
        print(f"precision score: {get_precision(pred, test_y)}")

    else:
        pass
    



