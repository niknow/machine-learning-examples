import itertools
from unittest import TestCase
import numpy as np
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility

from models.black_scholes import BlackScholes


class TestBlackScholes(TestCase):

    def setUp(self):
        self.sigma = 0.2
        self.r = 0.03
        self.bs = BlackScholes(self.sigma, self.r)
        self.maturities = np.array([3/12, 9/12, 1., 5., 10.])
        self.strikes = np.array([80., 95., 105., 120.3])
        self.rates = np.array([-0.03, -0.01, 0, 0.01, 0.03])
        self.sigmas = np.array([0.01, 0.05, 0.2, 0.5, 1.5])
        self.spots = np.array([80., 90., 100., 110., 120])
        self.call = [True, False]
        self.s0 = 100.
        self.sk = 103.
        self.tm = 2.4

    def test_vs_py_vollib(self):
        for sigma, r, s0, tm, sk, call in itertools.product(self.sigmas, self.rates, self.spots, self.maturities, self.strikes, self.call):
            with self.subTest():
                np.testing.assert_almost_equal(BlackScholes._option_price(sigma, r, s0, tm, sk, call),
                                               black_scholes(flag='c' if call else 'p', S=s0, K=sk, t=tm, r=r, sigma=sigma))

    def test_put_call_parity(self):
        call = self.bs.option_price(self.s0, self.tm, self.sk, call=True)
        put = self.bs.option_price(self.s0, self.tm, self.sk, call=False)
        df = np.exp(-self.r * self.tm)
        np.testing.assert_almost_equal(call - put, self.s0 - self.sk * df, decimal=6)

    def test_path_distribution(self):
        self.time_grid = np.array([0., 1., 5.])
        self.num_sims = 10000
        self.seed = 1
        paths = self.bs.paths(self.s0, self.time_grid, self.num_sims, self.seed)
        #np.testing.assert_array_almost_equal(paths.mean(axis=1), np.exp(self.time_grid * self.r) * self.s0)
        #print(paths.mean(axis=1), np.exp(self.time_grid * self.r) * self.s0)
        #print(paths.std(axis=1)**2, self.s0**2 * np.exp(2 * self.time_grid * self.r) * (np.exp(self.sigma**2 * self.time_grid) - 1))

    def test_vega(self):
        bump = 1. / 10000
        for tm, sk in itertools.product(self.maturities, self.strikes):
            with self.subTest():
                price = BlackScholes._option_price(self.sigma, self.r, self.s0, tm, sk, True)
                price_bumped = BlackScholes._option_price(self.sigma + bump, self.r, self.s0, tm, sk, True)
                vega_df = (price_bumped - price) / bump
                vega = BlackScholes._vega(self.sigma, self.r, self.s0, tm, sk)
                np.testing.assert_almost_equal(vega, vega_df, decimal=2)

    def test_delta(self):
        bump = 1. / 10000
        for tm, sk, call in itertools.product(self.maturities, self.strikes, self.call):
            with self.subTest():
                price = BlackScholes._option_price(self.sigma, self.r, self.s0, tm, sk, True)
                price_bumped = BlackScholes._option_price(self.sigma, self.r, self.s0 + bump, tm, sk, True)
                delta_fd = (price_bumped - price) / bump
                delta = BlackScholes._delta(self.sigma, self.r, self.s0, tm, sk)
                np.testing.assert_almost_equal(delta, delta_fd, decimal=5)

    def test_implied_volatility(self):
        for sigma, r, s0, tm, sk, call in itertools.product(self.sigmas, self.rates, self.spots, self.maturities, self.strikes, self.call):
            with self.subTest():
                price = BlackScholes._option_price(sigma, r, s0, tm, sk, call)
                iv = BlackScholes._implied_volatility(r, s0, tm, sk, call, price)
                np.testing.assert_almost_equal(sigma, iv)

    def test_implied_volatility_vs_py_vollib(self):
        for sigma, r, s0, tm, sk, call in itertools.product(self.sigmas, self.rates, self.spots, self.maturities, self.strikes, self.call):
            with self.subTest():
                price = BlackScholes._option_price(sigma, r, s0, tm, sk, True)
                np.testing.assert_almost_equal(BlackScholes._implied_volatility(r, s0, tm, sk, call, price),
                                               implied_volatility(price=price, S=s0, K=sk, t=tm, r=r, flag='c' if call else 'p'))

    def test_calibrate(self):
        self.bs = BlackScholes(self.r, self.sigma)
        self.vol_quotes = np.array([[1., 100., 0.23]])
        sigma = self.bs.calibrate(self.vol_quotes)
        np.testing.assert_almost_equal(sigma, 0.23)

    def test_calibrate_multiple(self):
        self.vol_quotes = np.array([[1., 100., 0.23], [2., 100., 0.27]])
        sigma = self.bs.calibrate(self.vol_quotes)
        self.assertTrue(sigma <= 0.27)
        self.assertTrue(sigma >= 0.23)
