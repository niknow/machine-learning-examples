import numpy as np
from scipy import  pi, exp, real, log
from scipy.integrate import quad,quadrature, trapz
from scipy.optimize import least_squares,fminbound
from scipy.stats import norm


#Heston Characteristic Function
def heston_char_fkt(S,T,r,q,u,v0,vLong,kappa,sigma,rho):
        gamma = kappa - 1j*rho*sigma*u
        d = np.sqrt( gamma**2 + (sigma**2)*u*(u+1j) )
        g = (gamma - d)/(gamma + d)
        C = (kappa*vLong)/(sigma**2)*((gamma-d)*T-2*np.log((1 - g*exp(-d*T))/( 1 - g ) ))
        D = (gamma - d)/(sigma**2)*((1 - np.exp(-d*T))/
          (1 - g*np.exp(-d*T)))
        F = S*exp((r-q)*T)
        return exp(1j*u*np.log(F) + C + D*v0)

#Heston Fundamental Transform
def heston_trafo(S,T,r,q,u,v0,vLong,kappa,sigma,rho):
        F = S*np.exp((r-q)*T)
        return np.exp(-1j*u*log(F))* heston_char_fkt(S,T,r,q,u,v0,vLong,kappa,sigma,rho)
    
#Heston Integrand(self,k,K,T)
def hestonintegrand(S,T,K,r,q,k,v0,vLong,kappa,sigma,rho):
        F = S*np.exp((r-q)*T)
        x = np.log(F/K)
        return real(np.exp(1j*k*x)/(k**2 + 1.0/4.0) * heston_trafo(S,T,r,q,k - 0.5*1j,v0,vLong,kappa,sigma,rho))

def heston_f1(u,logeps,v0,T):
        return abs(-0.5* v0* T * u**2 - np.log(u) - logeps)
        
def heston_f2(u,logeps,v0,vLong,kappa,sigma,rho,T):
        Cinf = (v0+kappa*vLong*T)/sigma*np.sqrt(1-rho**2)
        return abs(-Cinf*u - np.log(u) - logeps) 
    
def HestonCallPrice(S,T,K,r,q,v0,vLong,kappa,sigma,rho):
    """
    Computes the price of a call option in the Black/Scholes model.
    
    param S: the current spot price of the stock
    param r: the assumed risk-free rate
    param q: the assumed dividend yield
    param v0: spot variance
    param vLong: long term variance
    param kappa: mean reversion of variance
    param sigma: volatility of variance
    param rho: correlation of the driving BMs
    param T: the maturity of the option    
    param K: the strike of the option
    param t: the current time
    
    returns: price of call option maturing at T as of t (scalar)
    """

    val = 0
    a = (v0 * T)**0.5
    d1 = (log(S /K) + ((r-q) + v0 / 2) * T) / a
    d2 = d1 - a
    BSCall = S * np.exp(-q*T) * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
    logeps = log(0.00001)
    F = S*exp((r-q)*T)
    x = log(K/F)

    umax1 = fminbound(heston_f1,0,1000,args=(logeps,v0,T,))
    umax2 = fminbound(heston_f2,0,1000,args=(logeps,v0,vLong,kappa,sigma,rho,T,))
    umax = max(umax1,umax2)
    X = np.linspace(0,umax,1000)
    integrand = lambda k: real(np.exp(-1j*k*x)/(k**2 + 0.25) *(np.exp(-0.5*T*v0*(k**2 + 0.25))-
                     heston_trafo(S,T,r,q,k - 0.5*1j,v0,vLong,kappa,sigma,rho)))
    integral = trapz(integrand(X),x=X)
    val = (BSCall + np.sqrt(F*K)/pi * np.exp(-r*T) * integral)
    return val
