import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, newton
from py_vollib.black_scholes.implied_volatility import implied_volatility


class BlackScholes:
    """
    Implements the Black-Scholes model $dS_t = r S_t dt + \sigma S_t dW_t$.
    """

    def __init__(self, sigma, r):
        self.sigma = sigma
        self.r = r

    @staticmethod
    def _d1(sigma, r, s0, tm, sk):
        """
        Implements the d1 in the Black-Scholes option price formula.
        """
        d1 = np.log(s0 / sk) + (r + sigma ** 2 / 2) * tm
        return d1 / (sigma * np.sqrt(tm))

    @staticmethod
    def _option_price(sigma, r, s0, tm, sk, call):
        """
        Implements Black-Scholes option price formula.
        :param sigma: instantaneous volatility
        :param r: risk-free rate
        :param s0: value of underlying stock price at t=0
        :param tm: time to maturity of the option
        :param sk: strike of the option
        :param call: True if call option, False if put
        :return: option price
        """
        d1 = BlackScholes._d1(sigma, r, s0, tm, sk)
        d2 = d1 - sigma * np.sqrt(tm)
        pvk = sk * np.exp(-r * tm)
        phi = norm.cdf
        if call:
            return phi(d1) * s0 - phi(d2) * pvk
        else:
            return phi(-d2) * pvk - phi(-d1) * s0

    @staticmethod
    def _paths(sigma, r, s0, time_grid, num_sims, seed=1):
        """
        Create random paths of the underlying.

        :param sigma: instantaneous volatility
        :param r: risk-free rate
        :param time_grid: time grid of shape (num_time_steps) on which to simulate
        :param s0: initial value of stock at time_grid[0]
        :param num_sims: number of paths to generate
        :param seed: seed value of random number generator

        returns: a tensor `paths´ of shape (num_time_steps, num_sims) where S[i,j] is the j-th
                 realization of the underlying at time_grid[i]
        """
        delta = time_grid[1:] - time_grid[:-1]
        num_steps = delta.shape[0]
        np.random.seed(seed)
        dw = np.random.randn(num_sims, num_steps)
        paths = s0 * np.cumprod(np.exp((r - sigma ** 2 / 2) * delta + sigma * np.sqrt(delta) * dw), axis=1)
        return np.transpose(np.c_[np.ones(num_sims) * s0, paths])

    @staticmethod
    def _delta(sigma, r, s0, tm, sk, call=True):
        """
        Computes the Delta of a European call/put option.

        :param sigma: instantaneous volatility
        :param r: risk-free rate
        :param s0: value of underlying stock price at t=0
        :param tm: time to maturity of the option
        :param sk: strike of the option
        :param call: True if call option, False if put
        """
        phi = norm.cdf
        delta = phi(BlackScholes._d1(sigma, r, s0, tm, sk))
        if call:
            return delta
        else:
            return delta - 1

    @staticmethod
    def _vega(sigma, r, s0, tm, sk):
        """
        Computes the Vega of a European call/put option.

        :param sigma: instantaneous volatility
        :param r: risk-free rate
        :param s0: value of underlying stock price at t=0
        :param tm: time to maturity of the option
        :param sk: strike of the option
        """
        d1 = BlackScholes._d1(sigma, r, s0, tm, sk)
        return s0 * norm.pdf(d1) * np.sqrt(tm)

    @staticmethod
    def _implied_volatility(r, s0, tm, sk, call, price):
        """
        Computes the implied volatility of a European option.

        :param r: risk-free rate
        :param s0: value of underlying stock price at t=0
        :param tm: time to maturity of the option
        :param sk: strike of the option
        :param call: True if call option, False if put
        """

        return implied_volatility(price, s0, sk, tm, r, 'c' if call else 'p')

    @staticmethod
    def calibrate(vol_quotes):

        def cost(sigma):
            num_quotes = vol_quotes.shape[0]
            c = np.zeros(num_quotes)
            for i in range(num_quotes):
                tm, sk, iv = vol_quotes[i]
                c[i] = (iv - sigma) ** 2
            return np.sum(c) / 2

        return minimize_scalar(cost, bounds=(0, 1),  method='bounded', options={'xatol': 1e-8}).x

    def option_price(self, s0, tm, sk, call=True):
        return BlackScholes._option_price(self.sigma, self.r, s0, tm, sk, call)

    def paths(self, s0, time_grid, num_sims, seed=1):
        return BlackScholes._paths(self.sigma, self.r, s0, time_grid, num_sims, seed)

    def delta(self, s0, tm, sk, call=True):
        return BlackScholes._delta(self.sigma, self.r, s0, tm, sk, call)

    def vega(self, s0, tm, sk):
        return BlackScholes._vega(self.sigma, self.r, s0, tm, sk)

    def implied_volatility(self, s0, tm, sk, call, price):
        return BlackScholes._implied_volatility(self.r, s0, tm, sk, call, price)
