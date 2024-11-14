"""
Code for Gaussian Processes for Machine Learning
available at: https://gaussianprocess.org/gpml/chapters/RW.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class GP(object):
    def __init__(
        self,
        X,
        y,
        cov_f="se",
        l=1,
        signal_var=1,
        noise_var=0,
        n_star=100,
        x_range=[-5, 5],
    ):
        """
        Algorithm 2.1, page 19 && equation 4.31, page 119

        `cov_f` can be "se" or "periodic"
        """
        self.X = X
        self.y = y
        self.n = X.shape[1]  # number of observations
        self.cov_f = cov_f  # covariance function, either 'se' or 'periodic'
        self.l = l  # length scale
        self.signal_var = signal_var  # signal/kernel variance, used in 'se' kernel
        self.noise_var = noise_var  # observations noise
        self.n_star = n_star  # n of test points to predict
        self.X_star = np.linspace(x_range[0], x_range[1], n_star).reshape(
            n_star, 1
        )  # test points
        self.calc_Ks()

    def cov_matrix(self, xp, xq):
        if self.cov_f == "se":
            dists = cdist(xp, xq, metric="sqeuclidean")
            K = self.signal_var * np.exp(-1 / (2 * pow(self.l, 2)) * dists)
        elif self.cov_f == "periodic":
            dists = cdist(xp, xq, metric="euclidean")
            K = np.exp(-2 * pow(np.sin(dists / 2), 2) / pow(self.l, 2))
        return K

    def calc_Ks(self):
        self.K = self.cov_matrix(self.X, self.X) + self.noise_var * np.eye(self.n)
        self.Kstar = self.cov_matrix(self.X, self.X_star)

        self.Kss = self.cov_matrix(self.X_star, self.X_star)

        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(
            self.L.transpose(), np.linalg.solve(self.L, self.y)
        )
        self.v = np.linalg.solve(self.L, self.Kstar)

    def plot_samples(self, n, mean, sigma):
        for _ in range(n):
            f = np.random.multivariate_normal(mean, sigma)
            plt.plot(np.squeeze(self.X_star), f, c="teal", alpha=0.25)

    def plot_prior(self, n):
        self.plot_samples(n, np.zeros(self.n_star), self.Kss)
        plt.show()

    def predict(self):
        self.predictive_mean = np.matmul(self.Kstar.transpose(), self.alpha)
        self.predictive_variance = self.Kss - np.matmul(self.v.transpose(), self.v)

    def plot_posterior(self, n):
        self.plot_samples(n, self.predictive_mean.T[0], self.predictive_variance)
        plt.scatter(self.X, self.y, c="maroon")
        plt.show()

    def log_marginal_likelihood(self):
        return (
            0.5 * np.matmul(self.y.transpose(), self.alpha)
            - sum((np.log(self.L[i, i]) for i in range(self.L.shape[0])))
            - self.n / 2 * np.log(2 * np.pi)
        )


# EXAMPLE USE

# Some observations (interpretation of plot on page 15)
X = np.array([[-4.0], [-2.7], [-1.0], [0.0], [2.0]])
y = np.array([[-2], [0], [1], [2], [-1]])

seGP = GP(X, y, l=1)
seGP.plot_prior(10)

seGP.predict()
seGP.plot_posterior(10)

# seGP.log_marginal_likelihood()

# periodicGP = GP(X, y, cov_f="periodic", l=0.5)
# periodicGP.plot_prior(10)

# periodicGP.predict()
# periodicGP.plot_posterior(10)
