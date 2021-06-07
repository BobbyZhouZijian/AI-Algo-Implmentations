import math
import numpy as np
import pandas as pd
import argparse
from util import get_input_label_split, get_accuracy, get_precision
import faiss


class kNN:
    def __init__(self, k=5):
        self.k = k
        self.index = None
        self.train_x = None
        self.train_y = None

    def train(self, data, label_name):
        self.train_x, self.train_y = get_input_label_split(data, label_name)
        self.index = faiss.IndexFlatL2(self.train_x.shape[1])
        self.index.add(np.ascontiguousarray(self.train_x.astype('float32')))

        self.k = math.floor(math.sqrt(len(self.train_x)))

    def slow_infer(self, test_data):
        pred = []
        for d in test_data:
            dist = np.array([[np.linalg.norm(d - t), label] for t, label in zip(self.train_x, self.train_y)])
            dist = dist[dist[:, 0].argsort()]
            labels = np.array([d[1] for d in dist[:self.k]]).astype('int')
            l = np.argmax(np.bincount(labels))
            pred.append(l)

        return pred

    def infer(self, test_data):
        distances, indices = self.index.search(np.ascontiguousarray(test_data.astype('float32')), k=self.k)
        votes = self.train_y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


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
        knn = kNN()
        knn.train(df_train, args.label_name)

        test_x, test_y = get_input_label_split(df_test, args.label_name)
        pred = knn.infer(test_x)
        # pred = knn.slow_infer(test_x)

        print(f"accuracy score: {get_accuracy(pred, test_y)}")
        print(f"precision score: {get_precision(pred, test_y)}")

    else:
        pass
