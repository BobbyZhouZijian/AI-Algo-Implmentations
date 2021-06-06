'''
Gaussian Process Regressor builds on the property
that the conditional probability distribution of a
subset of multivariate normal is still a multibariate normal
distribution.

As the population mean of any distribution approaches a normal
distribution, when we make observations about a distribution, we
can assume each point follows a normal distribution. Hence, if we
make n observations in toatl, we can treat this n-observation as a
vector of n elements following a n-dim multivariate normal distribution.

If we want information on points that are not been observed yet,
we can add the observed points and the unobserved points to form a 
larger dimension multivariate normal distribution.

A multivariate normal distribution contains two parameters, the mean vector
which we typically assume to be 0 as a prior, and the covariance matrix, which
we calculate using a kernel function i.e. cov(xi, xj) = k(xi, xj).

By amalgamating the observed and the unobserved, we can derive the 
distribution of the unoserved points: the conditional distribution of 
the multivariate normal given that the observed points.

To improve the performance of the algorithm, we can do a preprocessing
to find a good value for the hyperparameters of the kernel function by minimizing
the negative loglikelihood loss.
'''

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


class GaussianProcessRegressor:
    def __init__(self, sigma=0.2, l=0.5, kernel='rbf'):
        self.kernel = kernel
        self.sigma = sigma
        self.l = l
    
    def gaussian_kernel(self, x1, x2):
        if self.kernel == 'rbf':
            dist_matrix = cdist(x1.reshape(-1,1), x2.reshape(-1,1), 'sqeuclidean')
            return self.sigma**2 * np.exp(-0.5 / self.l**2 * dist_matrix)
        else:
            raise Exception('kernel not recognized')
        
    def update(self, x_star):
        x = self.x
        x_star = np.asarray(x_star)
        k_yy = self.gaussian_kernel(x, x)
        k_ff = self.gaussian_kernel(x_star, x_star)
        k_yf = self.gaussian_kernel(x, x_star)
        k_fy = k_yf.T
        # add 1e-8 for numerical stability
        k_yy_inv = np.linalg.inv(k_yy + 1e-8 * np.eye(len(x)))

        mu_star = k_fy.dot(k_yy_inv).dot(self.y)
        cov_star = k_ff - k_fy.dot(k_yy_inv).dot(k_yf)

        return mu_star, cov_star
    
    def neg_log_likelihood_loss(self, params):
        # tune hyperparameter sigma nad l
        self.sigma = params[0]
        self.l = params[1]
        k_yy = self.gaussian_kernel(self.x, self.x)
        k_yy_inv = np.linalg.inv(k_yy + 1e-8 * np.eye(len(self.x)))
        loss = 0.5 * self.y.T.dot(k_yy_inv).dot(self.y) + 0.5 * np.linalg.slogdet(k_yy)[1] + 0.5 * len(self.x) * np.log(2*np.pi)
        return loss.reshape(-1)

    def fit(self, x, y):
        # set up prior belief
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        # optimize
        res = minimize(self.neg_log_likelihood_loss,
            [self.sigma, self.l], bounds=((1e-4, 1e4), (1e-4, 1e4)),
            method='L-BFGS-B')
        
        # self.sigma = res.x[0]
        # self.l = res.x[1]
        self.sigma = 0.2
        self.l = 0.5
    
    def predict(self, x_test):
        test = np.asarray(x_test)
        return self.update(test)


if __name__ == '__main__':
    # for testing purpose
    X = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1, 1)

    def f(x):
        return np.sin(X)*0.4 + np.random.normal(0, 0.05, size=X.shape)

    y = f(X)
    print(f"testing GP using X:{X}, and y:{y}")
    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)

    X_samples = np.array([1.5, 2.5, 3.5, 5.5, 7.5]).reshape(-1, 1)
    print(f"predicting using samples:{X_samples}")
    mu, cov = gpr.predict(X_samples)
    print(f"the mu of the samples are {mu}. The covariance matrix is {cov}")