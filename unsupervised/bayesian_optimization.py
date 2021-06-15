'''
Bayesian Optimization uses Bayesian probability to increase
the chance of finding a global extrema efficiently as Bayesian
probability helps utilising prior information.

More specifically, we maintain a initial samples points
and fit a gaussian process regressor on the samples (the _surrogate function)

Then, for each iteration, we generate a set of samples and ask the _surrogate function
to predict the mean and std of the samples, then
we use an acquisition function to produce a set of probabilities that each sample is the extrema we are looking for.

There are 3 most common choices of acquisitions:
    1. Probability of Improvement (POI)
    2. Expectation Improvement (EI)
    3. Upper Confidence Bound (UCB) 

Then we append the sample with the largest probability in the initial samples
and continue the optimization process.

Reference: https://machinelearningmastery.com/what-is-bayesian-optimization/#:~:text=Bayesian%20Optimization%20is%20an%20approach,and%2For%20expensive%20to%20evaluate.
'''

import argparse
import warnings
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


class BayesianOpitmizer:
    def __init__(self, objective=None, rang=(0,1), is_max=True, acq='ei', kappa=2.576):
        # model used to perform gaussian process
        self.model = GaussianProcessRegressor()
        # range to be optimised
        self.rang = rang

        # decide if we are looking for maxima or minima
        self.is_max = is_max

        # use Expectation Improvement by default
        # kappa will be ignored if mode is not ucb
        self.acq = acq

        # the objective function to be approached
        if objective is None:
            self.objective = self._default_objective()
        else:
            self.objective = objective
            self.rang = (0, 1)

    def _default_objective(self):
        def objective(x, noise=0.5):
            noise = np.random.normal(loc=0, scale=noise)
            return x**2 * math.sin(5 * math.pi * x) + noise
        print('using x**2 * math.sin(5 * math.pi * x) + noise as sample objective funciton.')
        return objective

    def _surrogate(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.model.predict(X, return_std=True)
    
    def plot(self, X, y, title='Plot'):
        plt.scatter(X, y, alpha=0.4)
        step = (self.rang[1] - self.rang[0]) / 1000.0
        X_samples = np.asarray(np.arange(self.rang[0], self.rang[1], step)).reshape(-1, 1)
        y_samples, _ = self._surrogate(X_samples)

        plt.plot(X_samples, y_samples)
        plt.show()

    def _acquisition(self, X, X_samples):
        y_hat, _ = self._surrogate(X)
        if self.is_max:
            best = max(y_hat)
        else:
            best = min(y_hat)
        mu, std = self._surrogate(X_samples)
        mu = mu[:, 0]

        a = mu - best
        z = a / (std + 1e-9)

        if self.acq == 'poi':
            # add 1e-9 to std to avoid divison by 0
            probs = norm.cdf(z)
        elif self.acq == 'ei':
            probs = a * norm.cdf(z) + std * norm.pdf(z)
        elif self.acq == 'ucb':
            probs = mu + self.kappa * std
        else:
            raise ValueError('acquisition not recognized')
        return probs
    
    def generate_samples(self, num):
        return np.random.random(num) * (self.rang[1] - self.rang[0]) + self.rang[0]
    
    def opt_acquisition(self, X):
        X_samples = self.generate_samples(100).reshape(-1, 1)
        scores = self._acquisition(X, X_samples)
        if self.is_max:
            ix = np.argmax(scores)
        else:
            ix = np.argmin(scores)
        return X_samples[ix, 0]
    
    def optimize(self, X, y, num_epochs=500):
        X = np.copy(X)
        y = np.copy(y)
        for i in range(num_epochs):
            self.model.fit(X, y)

            x = self.opt_acquisition(X)
            actual = self.objective(x)
            est, _ = self._surrogate([[x]])
            if (i+1) % 100 == 0:
                print(f"epoch: {i+1}/{num_epochs}, x = {x}, f = {est}, actual = {actual}")
            
            X = np.vstack((X, [[x]]))
            y = np.vstack((y, [[actual]]))

        self.plot(X, y, title='optimized X and y')
        return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_samples', default=50, help='number of samples to start with')
    parser.add_argument('--objective', help='the objective function, \
        can be recovered to python function using eval()')
    parser.add_argument('--range', default='(0,1)', help='a tuple containing the left and right bound')
    parser.add_argument('--acquisition', default='ei', help='choose from ei/poi/ucb')
    parser.add_argument('--min', action='store_true', help='whether we are looking for minima of the objective')
    parser.add_argument('--eval_mode', action='store_true', help='run this in evaluation mode')
    args = parser.parse_args()

    is_max = True
    if args.min:
        is_max = False
    
    acq = args.acquisition
    if acq not in ['ei', 'poi', 'ucb']:
        raise ValueError('Acquisition function not recognized')

    if args.eval_mode:
        optimizer = BayesianOpitmizer(is_max=is_max, acq=acq)
    else:
        fn = eval(args.objective)
        rang = eval(args.range)
        optimizer = BayesianOpitmizer(fn, rang, is_max=is_max, acq=acq)
    
    # get samples
    init_samples = int(args.init_samples)
    X = optimizer.generate_samples(init_samples)
    X.sort()
    y = [optimizer.objective(x) for x in X]
    y_actual =[optimizer.objective(x, 0) for x in X]

    print(f"Generated {init_samples} samples X and corresponding y")

    if is_max:
        best_idx_actual = np.argmax(y_actual)
        best_idx_samples = np.argmax(y)
    else:
        best_idx_actual = np.argmin(y_actual)
        best_idx_samples = np.argmin(y)
    print(f"Actual: Best X = {X[best_idx_actual]}; Best y = {y_actual[best_idx_actual]}")
    print(f"Samples: Best X = {X[best_idx_samples]}; Best y = {y_actual[best_idx_samples]}")

    # show graph before optimization
    plt.scatter(X, y)
    plt.plot(X, y_actual)
    plt.title('X and y before optimization')
    plt.show()

    X = np.asarray(X).reshape(-1, 1)
    y = np.asarray(X).reshape(-1, 1)

    opt_X, opt_y = optimizer.optimize(X, y, num_epochs=500)

    print("Optimization done!")
    if is_max:
        best_idx = np.argmax(opt_y)
    else:
        best_idx = np.argmin(opt_y)
    print(f"Best X = {opt_X[best_idx]}; Best y = {opt_y[best_idx]}")
