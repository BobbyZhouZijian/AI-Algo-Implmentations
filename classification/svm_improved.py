"""
The basic version of SVM uses gradietn ascent
to find the local maxima of the loss function:

L = 1/2 ||w||^2 + C / N * sum (max(0, 1 - yi(w * xi + b)))

While it works, it takes much time to finetune the value of
the learning rate in order to for the loss function to converge.

SMO (Sequential Minimal Optimization) can solve the Loss function without
the need of introduing extra hyperparameters. We shall use it for the improved
version of SVM.

SMO optimize the following Loss which is equivalent to the one above:

L = sum alpha - 0.5 * sum sum alphai * alphaj * yi * yj * xiT * xj
"""


import random
import numpy as np
import pandas as pd
import argparse
from util import get_input_label_split, get_accuracy, get_precision


class SVM:
    def __init__(self, C=1., gamma=0.01, kernel='rbf'):
        self.C = C
        self.kernel = kernel
        self.gamma=gamma
        self.train_x = None
        self.train_y = None
        self.weights = None
        self.bias = None

    
    def kernel_func(self, x, y):
        '''
        dot x and y based on the specified kernel

        K(x, y) = dot(phi(x), phi(y))
        '''
        if self.kernel == 'linear':
            return x.dot(y.T)
        elif self.kernel == 'poly':
            # for now, default it to degree 3, scale=1, bias=1
            scale = 1.
            bias = 1.
            deg = 3
            return (scale * x.dot(y.T) + bias)**deg
        elif self.kernel == 'rbf':
            # for now, default sigma = 1.
            m = x.shape[0]
            return np.exp(-self.gamma * np.linalg.norm(x-y.T)**2)
        else:
            raise Exception('Kernel not defined')
    

    def train_SMO(self, num_epochs):

        X = self.train_x
        y = self.train_y
        m, n = X.shape

        alpha = np.zeros(m)
        self.bias = 0

        for _ in range(num_epochs):
            for j in range(0, m):
                i = self.select_rand(0, m-1, j)
                xi, xj, yi, yj = X[i,:], X[j,:], y[i], y[j]
                kij = self.kernel_func(xi,xi) + self.kernel_func(xj,xj) - 2*self.kernel_func(xi,xj)
                if kij == 0:
                    continue
                
                ai, aj = alpha[i], alpha[j]
                L, H = self.compute_L_H(ai, aj, yi, yj)

                # compute w and b
                self.weights = self.calc_w(alpha, y, X)
                self.bias = self.calc_b(y, X)

                # compute Ei, Ej
                Ei = self.E(xi, yi)
                Ej = self.E(xj, yj)

                # update alpha
                alpha[j] = aj + float(yj * (Ei - Ej)) / kij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = ai + yi * yj * (aj - alpha[j])


    def build_ker_mat(self, data1, data2):
        m1 = data1.shape[0]
        m2 = data2.shape[0]
        ker_mat = np.mat(np.zeros((m1, m2)))

        for i in range(m1):
            for j in range(m2):
                ker_mat[i,j] = self.kernel_func(data1[i],data2[j])

        ker_mat = torch.tensor(ker_mat)
        return ker_mat


    def E(self, x, y):
        return self.infer(x) - y


    def calc_w(self, alpha, y, x):
        return np.dot(x.T, np.multiply(alpha, y)) 
    

    def calc_b(self, y, x):
        b_sum = y - np.dot(self.weights.T, x.T)
        return np.mean(b_sum)


    def select_rand(self, a, b, i):
        j = i
        while j == i:
            j = random.randint(a, b)
        return j


    def compute_L_H(self, ai, aj, yi, yj):
        if yi != yj:
            return max(0, aj-ai), min(self.C, self.C-ai+aj)
        else:
            return max(0, ai+aj-self.C), min(self.C, ai+aj)


    def train(self, data, label_name, num_epochs=30):
        self.train_x, self.train_y = get_input_label_split(data, label_name)
        self.train_SMO(num_epochs)


    def infer(self, data):
        return np.sign(np.dot(self.weights.T, data.T) + self.bias).astype(int)


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
    




