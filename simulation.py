# -*- coding: utf-8 -*-
"""
This script will create simulations of our variables
"""

import numpy as np

def multiBrownian(n, dim, T):
    '''
    A multidimensional independent Brownian motion 
    '''
    dt = T/n
    Z = np.sqrt(dt)*np.random.randn(dim, n)
    return np.cumsum(Z, axis = 1)
    
def reshape(array, dim):
    if type(array) == float or type(array) == int:
        array = array*np.ones(dim).reshape((dim,1))
    else:
        array = np.array(array).reshape((dim,1))
        
    return array
    
def geometricBM (n, dim, T, S0, r, delta, sigma, pho):
    '''
    This function will simulate a geometric BM
    
    S0 -> initial Value
    r -> risk free interest rate
    delta -> dividends yields
    sigma -> volatility
    pho -> Brownian Motion correlations
    '''
    delta = reshape(delta, dim)
    sigma = reshape(sigma, dim)
    S0 = reshape(S0, dim)
        
    Tp = np.arange(0, T, T/n)*np.ones((dim,n))
    
    mt = r - delta - sigma**2/2
    
    ms = sigma*sigma.transpose()
    Pho = pho*np.ones((dim, dim)) - (pho-1)*np.diag(np.ones(dim))
    ms = ms*Pho
    
    ms = np.linalg.cholesky(ms)
    
    Z = multiBrownian(n, dim, T)
    
    return S0 * np.exp(mt * Tp + np.matmul(ms, Z))
    
def simulate_x (M, n, dim, T, S0, r, delta, sigma, pho):
    '''
    This function will generate M copies of the geometric BM of dimension dim 
    and time step T/n
    
    S0 -> initial Value
    r -> risk free interest rate
    delta -> dividends yields
    sigma -> volatility
    pho -> Brownian Motion correlations
    '''
    return np.array([geometricBM (n, dim, T, S0, r, delta, sigma, pho) \
                     for i in range(M)])
    

