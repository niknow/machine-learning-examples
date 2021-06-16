from copy import deepcopy
from unittest import TestCase
import numpy as np
import GPy
import matplotlib.pyplot as plt

from dcke import DCKE, DCKEGridIndependent, DCKEGridRecursive
from locreg import LocalRegression
from models.black_scholes import BlackScholes


class TestDCKE(TestCase):

    def setUp(self):
        self.locreg = LocalRegression(degree=0)
        self.gpr_kernel = GPy.kern.RBF(input_dim=1)

    def test_black_scholes(self):
        self.r = 0.01
        self.sigma = 0.3
        self.bs = BlackScholes(r=self.r, sigma=self.sigma)
        self.time_grid = np.array([0, 0.5, 1.])
        self.num_sims = 1000
        np.random.seed(1)
        self.X = self.bs.paths(s0=100, time_grid=self.time_grid, num_sims=self.num_sims)
        self.T = self.time_grid[-1]
        self.t = self.time_grid[-2]
        self.K = 95
        self.df = np.exp(-(self.T-self.t) * self.r)
        self.y = self.df * np.maximum(self.X[-1] - self.K, 0)
        self.h = (4 / (3 * self.num_sims)) ** (1 / 5) * np.std(self.y)
        self.eps = 1 / (2 * self.h **2)
        self.num_quantiles = 100
        self.quantile_grid = np.linspace(0.1, 99.0, num=self.num_quantiles)
        self.x_mesh = np.percentile(self.X[1], self.quantile_grid)
        self.beta = np.zeros(self.num_quantiles)
        self.mz = np.zeros(self.num_quantiles)
        self.my = np.zeros(self.num_quantiles)
        self.var = np.zeros(self.num_quantiles)
        self.cov = np.zeros(self.num_quantiles)
        for i in range(self.x_mesh.shape[0]):
            x = self.x_mesh[i]
            k = np.exp(-self.eps * (self.X[1] - x)**2)
            self.mz[i] = np.sum(self.df * self.X[2] * k) / np.sum(k)
            self.my[i] = np.sum(self.y * k) / np.sum(k)
            cov = (self.y - self.my[i]) * (self.df * self.X[2] - self.mz[i])
            self.cov[i] = np.sum(cov * k) / np.sum(k)
            var = (self.df * self.X[2] - self.mz[i])**2
            self.var[i] = np.sum(var * k) / np.sum(k)
            self.beta[i] = - self.cov[i] / self.var[i]
        self.y_mesh = self.my + self.beta * (self.mz - self.x_mesh)
        self.gpr = GPy.models.GPRegression(np.atleast_2d(self.x_mesh).T,
                                           np.atleast_2d(self.y_mesh).T,
                                           deepcopy(self.gpr_kernel))
        self.gpr.optimize(messages=False)
        y_pred = self.gpr.predict(np.atleast_2d(self.x_mesh).T)[0].squeeze()
        self.dcke = DCKE(locreg=deepcopy(self.locreg), gpr_kernel=deepcopy(self.gpr_kernel))
        self.dcke.fit(X=np.atleast_2d(self.X[1]).T,
                      y=self.y,
                      Z=self.df * self.X[2],
                      x_mesh=np.atleast_2d(self.x_mesh).T,
                      mz=self.x_mesh)
        y_pred_dcke = self.dcke.predict(np.atleast_2d(self.x_mesh).T)
        y_true = np.array([self.bs.option_price(s, self.T - self.t, self.K) for s in self.x_mesh])
        # plt.plot(self.x_mesh, y_true, label="truth")
        # plt.plot(self.x_mesh, y_pred, label="pred")
        # plt.plot(self.x_mesh, y_pred_dcke, label="pred dcke")
        # plt.legend()
        # plt.show()
        np.testing.assert_array_almost_equal(self.y_mesh.squeeze(), self.dcke.y_mesh_.squeeze())
        np.testing.assert_array_almost_equal(self.x_mesh.squeeze(), self.dcke.x_mesh_.squeeze())
        np.testing.assert_array_almost_equal(y_pred, y_pred_dcke)
        self.assertTrue(np.all(np.abs(y_pred -y_true)<1))


class TestDCKEGridIndependent(TestCase):

    def setUp(self):
        self.mu = np.array([1, 2, 0])
        self.Sigma = np.array([[3, 0, 0],
                               [0, 4, 0],
                               [0, 0, 5]])
        quantile_levels = np.linspace(0.1, 99, 10)
        N = 100
        np.random.seed(1)
        W = np.random.multivariate_normal(self.mu, self.Sigma, N)
        self.X = W[:, 0]
        self.Y = W[:, 1]
        self.Z = W[:, 2]
        self.x_mesh = np.percentile(self.X, quantile_levels)
        self.mz = np.zeros_like(self.x_mesh)
        self.locreg = LocalRegression(degree=0)
        self.gpr_kernel = GPy.kern.RBF(input_dim=1)

    def test_singleton(self):
        self.dcke = DCKE(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.dcke.fit(np.atleast_2d(self.X).T, self.Y, np.atleast_2d(self.x_mesh).T, self.Z, self.mz)
        y_pred = self.dcke.predict(np.atleast_2d(self.x_mesh).T)
        self.dcke_grid = DCKEGridIndependent(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.dcke_grid.fit(self.X[np.newaxis, : , np.newaxis],
                           self.Y[np.newaxis, :],
                           self.x_mesh[np.newaxis, :, np.newaxis],
                           self.Z[np.newaxis, :],
                           self.mz[np.newaxis, :])
        y_pred_grid = self.dcke_grid.predict(self.x_mesh[np.newaxis, :, np.newaxis])
        np.testing.assert_array_almost_equal(y_pred, y_pred_grid.squeeze())

    def test_grid_components(self):
        self.dcke1 = DCKE(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.dcke2 = DCKE(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.dcke1.fit(np.atleast_2d(self.X).T,
                       self.Y,
                       np.atleast_2d(self.x_mesh).T,
                       self.Z,
                       self.mz)
        bandwidth = self.dcke1.locreg.bandwidth
        self.dcke2.fit(2 * np.atleast_2d(self.X).T,
                       2 * self.Y,
                       2 * np.atleast_2d(self.x_mesh).T,
                       2 * self.Z,
                       2 * self.mz,
                       bandwidth)
        y_pred1 = self.dcke1.predict(np.atleast_2d(self.x_mesh).T)
        y_pred2 = self.dcke2.predict(2 * np.atleast_2d(self.x_mesh).T)
        self.dcke_grid = DCKEGridIndependent(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        X = np.concatenate((np.atleast_2d(self.X).T, 2 * np.atleast_2d(self.X).T), axis=1).T[:, :, np.newaxis]
        y = np.concatenate((np.atleast_2d(self.Y).T, 2 * np.atleast_2d(self.Y).T), axis=1).T
        Z = np.concatenate((np.atleast_2d(self.Z).T, 2 * np.atleast_2d(self.Z).T), axis=1).T
        mz = np.concatenate((np.atleast_2d(self.mz).T, 2 * np.atleast_2d(self.mz).T), axis=1).T
        x_mesh = np.concatenate((np.atleast_2d(self.x_mesh).T, 2 * np.atleast_2d(self.x_mesh).T), axis=1).T[:, :, np.newaxis]
        self.dcke_grid.fit(X, y, x_mesh, Z, mz, bandwidth)
        y_pred_grid = self.dcke_grid.predict(x_mesh)
        y_pred = {0: y_pred1, 1: y_pred2}
        dcke = {0: self.dcke1, 1: self.dcke2}
        for i in range(2):
            np.testing.assert_array_almost_equal(y_pred[i], y_pred_grid[i].squeeze())
            np.testing.assert_array_almost_equal(self.dcke_grid.X_[i], dcke[i].X_train_)
            np.testing.assert_array_almost_equal(self.dcke_grid.y_[i], dcke[i].y_train_)
            np.testing.assert_array_almost_equal(self.dcke_grid.x_mesh_[i], dcke[i].x_mesh_)
            np.testing.assert_array_almost_equal(self.dcke_grid.Z_[i], dcke[i].Z_)
            np.testing.assert_array_almost_equal(self.dcke_grid.mz_[i], dcke[i].mz_)
            np.testing.assert_array_almost_equal(self.dcke_grid.cov_[i], dcke[i].cov_)
            np.testing.assert_array_almost_equal(self.dcke_grid.var_[i], dcke[i].var_)
            np.testing.assert_array_almost_equal(self.dcke_grid.beta_[i], dcke[i].beta_)

    def test_bandwidths(self):
        self.dcke = DCKEGridIndependent(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.assertListEqual(self.dcke._get_bandwidths(None, 3), [None, None, None])
        np.testing.assert_array_almost_equal(self.dcke._get_bandwidths(2.7, 3),
                                             np.array([2.7, 2.7, 2.7]))
        np.testing.assert_array_almost_equal(self.dcke._get_bandwidths(np.array([1., 2., 3.]), 3),
                                             np.array([1., 2., 3.]))


class TestDCKEGridRecursive(TestCase):

    def setUp(self):
        self.mu = np.array([1, 2, 0])
        self.Sigma = np.array([[3, 0, 0],
                               [0, 4, 0],
                               [0, 0, 5]])
        quantile_levels = np.linspace(0.1, 99, 10)
        N = 100
        np.random.seed(1)
        W = np.random.multivariate_normal(self.mu, self.Sigma, N)
        self.X = W[:, 0]
        self.Y = W[:, 1]
        self.Z = W[:, 2]
        self.x_mesh = np.percentile(self.X, quantile_levels)
        self.mz = np.zeros_like(self.x_mesh)
        self.locreg = LocalRegression(degree=0)
        self.gpr_kernel = GPy.kern.RBF(input_dim=1)

    def test_singleton(self):
        self.dcke = DCKE(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.dcke.fit(np.atleast_2d(self.X).T, self.Y, np.atleast_2d(self.x_mesh).T, self.Z, self.mz)
        y_pred = self.dcke.predict(np.atleast_2d(self.x_mesh).T)
        y_pred = self.dcke.predict(np.atleast_2d(self.x_mesh).T)
        h = self.dcke.locreg.bandwidth
        self.dcke_grid = DCKEGridRecursive(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.dcke_grid.fit(self.X[np.newaxis, : , np.newaxis],
                           self.Y[np.newaxis, :],
                           self.x_mesh[np.newaxis, :, np.newaxis],
                           self.Z[np.newaxis, :],
                           self.mz[np.newaxis, :],
                           h)
        y_pred_grid = self.dcke_grid.predict(self.x_mesh[np.newaxis, :, np.newaxis])
        np.testing.assert_array_almost_equal(y_pred, y_pred_grid.squeeze())

    def test_grid_components(self):
        self.dcke2 = DCKE(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.dcke1 = DCKE(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        self.dcke2.fit(2 * np.atleast_2d(self.X).T,
                       2 * self.Y,
                       2 * np.atleast_2d(self.x_mesh).T,
                       2 * self.Z,
                       2 * self.mz)
        y_pred2 = self.dcke2.predict(2 * np.atleast_2d(self.X).T)
        df = np.exp(-0.5)
        self.dcke1.fit(np.atleast_2d(self.X).T,
                       df * y_pred2,
                       np.atleast_2d(self.x_mesh).T,
                       self.Z,
                       self.mz)
        y_pred1 = self.dcke1.predict(np.atleast_2d(self.X).T)
        self.dcke_grid = DCKEGridRecursive(deepcopy(self.locreg), deepcopy(self.gpr_kernel))
        X = np.concatenate((np.atleast_2d(self.X).T, 2 * np.atleast_2d(self.X).T), axis=1).T[:, :, np.newaxis]
        Z = np.concatenate((np.atleast_2d(self.Z).T, 2 * np.atleast_2d(self.Z).T), axis=1).T
        mz = np.concatenate((np.atleast_2d(self.mz).T, 2 * np.atleast_2d(self.mz).T), axis=1).T
        x_mesh = np.concatenate((np.atleast_2d(self.x_mesh).T, 2 * np.atleast_2d(self.x_mesh).T), axis=1).T[:, :, np.newaxis]
        bandwidths = np.array([self.dcke1.locreg.bandwidth, self.dcke2.locreg.bandwidth])
        self.dcke_grid.fit(X, 2 * self.Y, x_mesh, Z, mz, bandwidths, recursion_functions=[lambda x: x * df])
        y_pred_grid = self.dcke_grid.predict()
        y_pred = {0: y_pred1, 1: y_pred2}
        dcke = {0: self.dcke1, 1: self.dcke2}
        for i in range(2):
            np.testing.assert_array_almost_equal(self.dcke_grid[i].X_train_, dcke[i].X_train_)
            np.testing.assert_array_almost_equal(self.dcke_grid[i].y_train_, dcke[i].y_train_)
            np.testing.assert_array_almost_equal(self.dcke_grid[i].x_mesh_, dcke[i].x_mesh_)
            np.testing.assert_array_almost_equal(self.dcke_grid[i].Z_, dcke[i].Z_)
            np.testing.assert_array_almost_equal(self.dcke_grid[i].mz_, dcke[i].mz_)
            np.testing.assert_array_almost_equal(self.dcke_grid[i].var_, dcke[i].var_)
            np.testing.assert_array_almost_equal(self.dcke_grid[i].cov_, dcke[i].cov_)
            np.testing.assert_array_almost_equal(self.dcke_grid[i].beta_, dcke[i].beta_)
            np.testing.assert_array_almost_equal(self.dcke_grid[i].y_mesh_, dcke[i].y_mesh_)
            np.testing.assert_array_almost_equal(y_pred[i], y_pred_grid[i].squeeze())
