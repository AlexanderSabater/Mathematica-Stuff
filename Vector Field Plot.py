# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:29:00 2022

@author: Alexa
"""

import numpy as np
import matplotlib.pyplot as plt

# vector field
def P(f,L,N):
    # square window of side length of 2L, for simplicity
    # N sampling points
    x1 = np.linspace(-L,L,N)
    x2 = np.linspace(-L,L,N)
    
    X1, X2 = np.meshgrid(x1,x2)
    
    u, v = np.zeros(X1.shape), np.zeros(X2.shape)
    
    NI, NJ = X1.shape
    
    for i in range(NI):
        for j in range(NJ):
            x = X1[i, j]
            y = X2[i, j]
            xprime = f(x,y)
            u[i,j] = xprime[0]
            v[i,j] = xprime[1]
        # end of for loop
    # end of for loop
         
    Q = plt.streamplot(X1, X2, u, v, density = 1)
    
    plt.xlabel('$y$')
    plt.ylabel('$y\'$') # p for momentum!
    plt.xlim([-L,L])
    plt.ylim([-L,L])
    
    return Q
# end of P

# harmonic oscillator
def f(y1,y2):
    return np.array([y2,y1*(1-y1)])
# end of f

L = 2
N = 50
P(f,L,N)