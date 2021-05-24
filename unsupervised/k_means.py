import math
import numpy as np
import pandas as pd
import argparse
from util import get_input_label_split


class kMeans:

    '''
    Implement K-Means using K-Means++ initialization of centers.
    Then use the elkan algorithm to speed up the E step. Specifically:
        Let x be a point and let b and c be centers, then:
        1. if d(b,c) >= 2d(x,b) then d(x,c) >= d(x,b)
        2. d(x,c) >= max{0, d(x,b) - d(b,c)}
    '''
    def __init__(self, k=5):
        self.k = k
        self.index = None
        self.train_x = None
        self.classification = []
        self.centers = []

        # for testing
        np.random.seed(2021)
    

    def init_centers(self):

        '''initialize the centers using the k-Means++ way'''
        size = self.train_x.shape[0]

        # calculate a center first
        rand = np.random.randint(size)
        self.centers.append(self.train_x[rand])

        # use the heuristic: D(x) = arg min||xi - ur||^2 where r = 1,2,...,k_selected
        # choose the x with larger D(x) as the next center

        for _ in range(self.k-1):
            dist = [float('inf') for _ in range(size)]
            for i in range(size):
                pt = self.train_x[i]
                for c in self.centers:
                    dist[i] = min(dist[i], np.linalg.norm(pt - c))
            best_idx = dist.index(max(dist))
            self.centers.append(self.train_x[best_idx])


    def elkan(self, iterations):

        size = self.train_x.shape[0]

        # initialize center_dists and sc for elkan algorithm
        center_dists = [[0 for _ in range(self.k)] for _ in range(self.k)]
        sc = []
        l = [[0 for _ in range(self.k)] for _ in range(size)]
        u = [0 for _ in range(size)]
        r = [False for _ in range(size)]

        for i in range(size):
            pt = self.train_x[i]
            dist = np.linalg.norm(pt - self.centers[0]); closest_j = 0
            l[i][0] = dist
            for j,c in enumerate(self.centers):
                if center_dists[closest_j][j] >= 2 * dist:
                    continue
                cur_dist = np.linalg.norm(pt - c)
                l[i][j] = cur_dist
                if cur_dist < dist:
                    dist = cur_dist
                    closest_j = j
            u[i] = dist
            self.classification[i] = closest_j


        for _ in range(iterations):
            for i in range(self.k):
                min_dist = float('inf')
                for j in range(self.k):
                    if i == j:
                        center_dists[i][j] = 0.0
                        continue
                    center_dists[i][j] = np.linalg.norm(self.centers[i] - self.centers[j])
                    min_dist = min(min_dist, center_dists[i][j])
                sc.append(0.5 * min_dist)

            selected_points = []
            for i in range(size):
                if u[i] > sc[self.classification[i]]:
                    selected_points.append(i)

            for i in selected_points:
                c = self.classification[i]
                dist = 0
                if r[i]:
                    dist = np.linalg.norm(self.centers[c] - self.train_x[i])
                    r[i] = False
                else:
                    dist = u[i]
                
                for j in range(self.k):
                    if j == c or u[i] <= l[i][j]:
                        continue
                    if dist > l[i][j] or \
                        dist > 0.5 * center_dists[c][j]:
                        dist_j = np.linalg.norm(self.train_x[i] - self.centers[j])
                        if dist_j < dist:
                            self.classification[i] = j
            
            # recalculate means
            means = [np.zeros_like(self.train_x[0]) for _ in range(self.k)]
            counts = [0 for _ in range(self.k)]
            for i in range(size):
                c = self.classification[i]
                means[c] += self.train_x[i]
                counts[c] += 1
            for i in range(self.k):
                means[i] /= counts[i]
            
            for i in range(size):
                for j in range(self.k):
                    l[i][j] = max(l[i][j] - np.linalg.norm(self.centers[j] - means[j]), 0.0)
                
                c = self.classification[i]
                u[i] = u[i] + np.linalg.norm(self.centers[c] - means[c]) 
                r[i] = True
            
            self.centers = means




    def classify(self):

        '''similar to the E Step in the EM algorithm'''
        size = self.train_x.shape[0]
        for i in range(size):
            pt = self.train_x[i]
            final_c = -1; dist = float('inf')
            for j, c_pt in enumerate(self.centers):
                cur_dist = np.linalg.norm(pt  - c_pt)
                if cur_dist < dist:
                    dist = cur_dist
                    final_c = j
            self.classification[i] = final_c



    def update_centers(self):

        '''similar to the M Step in the EM algorithm'''
        counts = [0 for _ in range(self.k)]
        weights = [np.zeros_like(self.train_x[0]) for _ in range(self.k)]
        size = self.train_x.shape[0]

        for i in range(size):
            pt = self.train_x[i]
            c = self.classification[i]
            weights[c] += pt
            counts[c] += 1
        for i in range(self.k):
            self.centers[i] = weights[i] / counts[i]


    
    def train(self, data, iterations=500, algorithm='normal'):
        self.train_x = get_input_label_split(data, label_name=None)
        size = self.train_x.shape[0]
        self.init_centers()
        self.classification = [-1 for _ in range(size)]

        if algorithm == 'normal':
            for _ in range(iterations):
                    self.classify()
                    self.update_centers()
        elif algorithm == 'elkan':
            self.elkan(iterations=iterations)


    
    def get_centroids(self):
        return self.centers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True, help='training data file path')
    parser.add_argument('--algorithm', default='normal', help='K-Means algorithm to run')
    parser.add_argument('--eval_mode', action='store_true', help='run this in evaluation mode')
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    if args.eval_mode:
        kmeans = kMeans()
        kmeans.train(df, algorithm=args.algorithm)
        centroids = kmeans.get_centroids()
        
        print(f"The {kmeans.k} centroids are {centroids}")
    else:
        pass
    