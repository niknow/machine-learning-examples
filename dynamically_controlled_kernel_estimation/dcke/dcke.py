from abc import ABC, abstractmethod
from inspect import isfunction
import numpy as np
from sklearn.base import RegressorMixin
from copy import deepcopy
from GPy.models import GPRegression


class DCKE(RegressorMixin):
    """ Dynamically Controlled Kernel Estimation
        Computes the conditional expectation $E[Y \mid X=x]$ from
        a training set $X_i$, $y_i$, $i=1, \ldots, N$ of joint
        realizations of $X$ and $Y$ for an arbitrary prediction
        set of $x$'s. The DCKE regressor first uses local regression
        on a mesh grid to solve the problem on the mesh grid and then
        uses GPR to evaluate in between the points on the mesh grid.
        Optionally, a control variate $Z$ can be supplied together
        with $\mu_Z = E[Z \mid X=x_k]$ for the points $x_k$ on the
        mesh grid. In that case, the expectation
        $E[Y +\beta (Z-\mu_Z) \mid X=x_k]$ is computed on the
        mesh grid with variance reduced by the correlation between
        $Y$ and $Z$.
    """

    def __init__(self, locreg, gpr_kernel):
        """
        Initializes the DCKE object.
        :param locreg: an instance of LocalRegression
        :param gpr_kernel: an instance of GPy.kern
        """
        self.locreg = locreg
        self.gpr_kernel = gpr_kernel
        self.gpr_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.x_mesh_ = None
        self.y_mesh_ = None
        self.Z_ = None
        self.mz_ = None
        self.cov_ = None
        self.var_ = None
        self.beta_ = None

    def fit(self, X, y, x_mesh, Z=None, mz=None, bandwidth=None):
        """
        Fits the DCKE to training data.

        :param X: a numpy array of shape (num_samples, num_dimensions)
        :param y: a numpy array of shape (num_samples,)
        :param Z: a numpy array of shape (num_samples,)
        :param x_mesh: a numpy array of shape (num_meshes, num_dimensions)
        :param mz: a numpy array of shape (num_meshes,) any any mz[k]
                   satisties mz[k] = E[Z \mid X=x_k]$ where x_k are the
                   points in x_mesh
        :param bandwidth: bandwidth parameter for the local regression
        :return:
        """
        self.X_train_ = X
        self.y_train_ = y
        self.x_mesh_ = x_mesh
        if Z is None and mz is None:
            self.Z_ = np.zeros_like(self.y_train_)
            self.mz_ = np.zeros(self.x_mesh_.shape[0])
        elif (Z is None and mz is not None) or (Z is not None and mz is None):
            raise ValueError('Parameter Z and mz have to be either both None or both not None.')
        else:
            self.Z_ = Z
            self.mz_ = mz
        self.locreg.warm_start = True
        self.locreg.fit(X, y, bandwidth)

    def _calculate_locregs(self):
        """
        Uses the approximate conditional expectation operator
        $\tilde E[_ \mid X=x]$ defined by the local regression in self.locreg
        to compute the approximate optimal beta for the control variate $Z$
        defined by $\beta_x = - \tfrac{\Cov[Y, Z \mid X=x]}{\Var[Z \mid X=x]}$
        for all $x$ in self.x_mesh.

        :return: beta, a numpy array of shape (num_mesh_points, )
        """
        h = self.locreg.bandwidth
        n = self.x_mesh_.shape[0]
        self.cov_ = np.zeros(n)
        self.var_ = np.zeros(n)
        self.y_mesh_ = np.zeros(n)
        self.beta_ = np.zeros(n)
        m_y = np.zeros(n)
        m_z = np.zeros(n)
        for i in range(n):
            m_y[i] = self.locreg.predict(np.atleast_2d(self.x_mesh_[i]).T).squeeze()
            self.locreg.fit_partial(np.atleast_2d(self.Z_).T, h)
            m_z[i] = self.locreg.predict_partial().squeeze()
            self.locreg.fit_partial((self.y_train_ - m_y[i]) * (self.Z_ - m_z[i]), h)
            self.cov_[i] = self.locreg.predict_partial().squeeze()
            self.locreg.fit_partial((self.Z_ - m_z[i]) ** 2, h)
            self.var_[i] = self.locreg.predict_partial().squeeze()
            self.beta_[i] = - self.cov_[i] / self.var_[i]
            self.locreg.fit_partial(self.y_train_ + self.beta_[i] * (self.Z_ - self.mz_[i]), h)
            self.y_mesh_[i] = self.locreg.predict_partial()

    def predict(self, X):
        """
        Predicts the conditional expectation $E[Y \mid X=x]$ for all x in $X$.

        :param X: a numpy array of shape (num_predictions, num_dimensions)
        :return: a numpy array of shape (num_predictions,)
        """

        self._calculate_locregs()
        self.gpr_ = GPRegression(self.x_mesh_,
                                 np.atleast_2d(self.y_mesh_).T,
                                 self.gpr_kernel)
        self.gpr_.optimize(messages=False)
        #self.gpr_.optimize_restarts(num_restarts = 10)
        y_pred, self.gp_var_ = self.gpr_.predict(X)
        self.gp_var_ = self.gp_var_.squeeze()
        return y_pred.squeeze()


class DCKEGrid(ABC):

    def __init__(self, locreg, gpr):
        self.locreg = locreg
        self.gpr = gpr
        self.dckes = []

    @abstractmethod
    def fit(self, X, y, x_mesh, Z=None, mz=None, bandwidth=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def __getitem__(self, key):
        return self.dckes[key]

    @property
    def cov_(self):
        return np.array([dcke.cov_ for dcke in self.dckes])

    @property
    def var_(self):
        return np.array([dcke.var_ for dcke in self.dckes])

    @property
    def beta_(self):
        return np.array([dcke.beta_ for dcke in self.dckes])


class DCKEGridIndependent(DCKEGrid):
    """
    Provides a wrapper for consistently estimating conditional expectations
    via DCKE on a grid of random variables, e.g. from a stochastic process.
    """

    def _get_bandwidths(self, bandwidth, m):
        if bandwidth is None:
            return [None for _ in range(m)]
        elif isinstance(bandwidth, (list, tuple, np.ndarray)):
            return bandwidth
        else:
            return np.array([bandwidth for _ in range(m)])

    def fit(self, X, y, x_mesh, Z=None, mz=None, bandwidth=None):
        """
        Fits the DCKE to training data.

        :param X: a numpy array of shape (num_grid_points, num_samples, num_dimensions)
        :param y: a numpy array of shape (num_grid_points, num_samples,)
        :param Z: a numpy array of shape (num_grid_points, num_samples,)
        :param x_mesh: a numpy array of shape (num_grid_points, num_meshes, num_dimensions)
        :param mz: a numpy array of shape (num_meshes,) any any mz[k]
                   satisfies mz[k] = E[Z \mid X=x_k]$ where x_k are the
                   points in x_mesh
        :param bandwidth: bandwidth parameters for the local regression
                          if None, then bandwidth will be selected automatically
                          if scalar, then bandwith will be the same for all
                          if array, then each DCKE uses its own bandwidth
        :return:
        """
        m = X.shape[0]
        self.dckes = [DCKE(deepcopy(self.locreg), deepcopy(self.gpr)) for _ in range(m)]
        bandwidths = self._get_bandwidths(bandwidth, m)
        for i in range(m):
            self.dckes[i].fit(
                np.atleast_2d(X[i]),
                y[i],
                x_mesh[i],
                Z[i],
                mz[i],
                bandwidths[i])

    def predict(self, X):
        """
        Predicts the conditional expectation $E[Y \mid X=x]$ for all x in $X$.

        :param X: a numpy array of shape (num_grid_points, num_predictions, num_dimensions)
        :return: a numpy array of shape (num_grid_points, num_predictions,)
        """
        m = X.shape[0]
        return np.array([self.dckes[i].predict(np.atleast_2d(X[i])) for i in range(m)])

    @property
    def X_(self):
        return np.array([dcke.X_train_ for dcke in self.dckes])

    @property
    def y_(self):
        return np.array([dcke.y_train_ for dcke in self.dckes])

    @property
    def x_mesh_(self):
        return np.array([dcke.x_mesh_ for dcke in self.dckes])

    @property
    def Z_(self):
        return np.array([dcke.Z_ for dcke in self.dckes])

    @property
    def mz_(self):
        return np.array([dcke.mz_ for dcke in self.dckes])


class DCKEGridRecursive(DCKEGrid):

    def __init__(self, locreg, gpr):
        super().__init__(locreg, gpr)
        self.X_train_ = None
        self.y_train_ = None
        self.x_mesh_ = None
        self.Z_ = None
        self.mz_ = None
        self.bandwidths_ = None
        self.recursion_functions_ = None
        self.y_rec_ = None

    def fit(self, X, y, x_mesh, Z=None, mz=None, bandwidth=None, recursion_functions=None):
        """
        Fits the DCKE to training data.

        :param X: a numpy array of shape (num_grid_points, num_samples, num_dimensions)
        :param y: a numpy array of shape (num_samples,)
                             or of shape (num_samples,)
        :param Z: a numpy array of shape (num_grid_points, num_samples,)
        :param x_mesh: a numpy array of shape (num_grid_points, num_meshes, num_dimensions)
        :param mz: a numpy array of shape (num_meshes,) any any mz[k]
                   satisfies mz[k] = E[Z \mid X=x_k]$ where x_k are the
                   points in x_mesh
        :param bandwidth: bandwidth parameters for the local regression
                          if None, then bandwidth will be selected automatically
                          if scalar, then bandwith will be the same for all
                          if array of scalars, then each DCKE uses its own bandwidth
                          if array of functions, then each DCKE computes its own bandwidth
                          by evaluating the function on y_train_
        :return:
        """
        self.X_train_ = X
        self.y_train_ = y
        self.x_mesh_ = x_mesh
        self.Z_ = Z
        self.mz_ = mz
        self.bandwidths_ = self._get_bandwidths(bandwidth)
        m = X.shape[0]
        self.dckes = [DCKE(deepcopy(self.locreg), deepcopy(self.gpr)) for _ in range(m)]
        self.bandwidths_ = self._get_bandwidths(bandwidth)
        self.recursion_functions_ = self._get_recursion_functions(recursion_functions)

    def _get_bandwidths(self, bandwidth):
        m = self.X_train_.shape[0]
        if bandwidth is None:
            bw = [lambda x: None for _ in range(m)]
        elif np.isscalar(bandwidth):
            bw = [lambda x: bandwidth for _ in range(m)]
        elif isinstance(bandwidth, (list, tuple, np.ndarray)):
            if np.isscalar(bandwidth[0]):
                bw = [lambda x, b=b: b for b in bandwidth]
            elif isfunction(bandwidth[0]):
                bw = bandwidth
            else:
                raise ValueError("Bandwidths not recognized.")
        else:
            raise ValueError("Bandwidths not recognized..")
        return bw

    def _get_recursion_functions(self, recursion_functions):
        m = self.X_train_.shape[0]
        if recursion_functions is None:
            rf = [lambda x: x for _ in range(m)]
        elif isinstance(recursion_functions, (list, tuple, np.ndarray)):
            if isfunction(recursion_functions[0]):
                rf = recursion_functions
            else:
                raise ValueError("Recursion functions not recognized.")
        else:
            raise ValueError("Recursion functions not recognized..")
        return rf

    def predict(self, X=None):
        """
        Predicts the conditional expectation $E[Y \mid X=x]$ for all x in $X$.

        :param X: a numpy array of shape (num_grid_points, num_predictions, num_dimensions)
        :param recursion_functions: If not None, then only self[-1] uses y_train_ for 
                                    the prediction. Traversing the list of DCKEs backwards,
                                    in step i, self[i] uses f(self[i+1].predict(X[i+1])) 
                                    instead of self[i].y_train_, where 
                                    f = recursion_functions[i].
        :return: a numpy array of shape (num_grid_points, num_predictions,)
        """

        num_grid_points = self.X_train_.shape[0]
        num_samples = self.X_train_.shape[1]
        self.y_rec_ = np.zeros((num_grid_points, num_samples))
        if X is not None:
            num_predictions = X.shape[1]
            y_pred = np.zeros((num_grid_points, num_predictions))
        self[-1].fit(
            self.X_train_[-1],
            self.y_train_,
            self.x_mesh_[-1],
            self.Z_[-1],
            self.mz_[-1],
            self.bandwidths_[-1](self.y_train_))
        self.y_rec_[-1, :] = self[-1].predict(self.X_train_[-1])
        if X is not None:
            y_pred[-1, :] = self[-1].predict(X[-1])
        for i in range(num_grid_points-2, -1, -1):
            y = self.recursion_functions_[i](self.y_rec_[i+1])
            self[i].fit(
                self.X_train_[i],
                y,
                self.x_mesh_[i],
                self.Z_[i],
                self.mz_[i],
                self.bandwidths_[i](y))
            self.y_rec_[i, :] = self[i].predict(self.X_train_[i])
            if X is not None:
                y_pred[i, :] = self[i].predict(X[i])
        if X is not None:
            return y_pred
        else:
            return self.y_rec_
