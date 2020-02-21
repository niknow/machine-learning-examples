import numpy as np
import scipy.stats as stats


def BS_d1(S, dt, r, sigma, K):
    """
    Computes the auxilliary quantity d1 in the Black/Scholes forumla
    
    param S:  the current spot price of the stock
    param dt: the remaining time to maturity of the option
    param r:  the assumed risk-free rate
    param sigma: the volatility of the stock
    param K: the strike of the option
    
    returns: d1 as per Black/scholes formula (scalar)
    """
    return (np.log(S/K) + (r+sigma**2/2)*dt) / (sigma*np.sqrt(dt))


def BlackScholesCallPrice(S, r, sigma, T, K, t=0):
    """
    Computes the price of a call option in the Black/Scholes model.
    
    param S: the current spot price of the stock
    param r: the assumed risk-free rate
    param r: the assumed volatility of the stock
    param T: the maturity of the option    
    param K: the strike of the option
    param t: the current time
    
    returns: price of call option maturing at T as of t (scalar)
    """
    
    dt = T-t
    Phi = stats.norm(loc=0, scale=1).cdf
    d1 = BS_d1(S, dt, r, sigma, K)
    d2 = d1 - sigma * np.sqrt(dt)
    return S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
