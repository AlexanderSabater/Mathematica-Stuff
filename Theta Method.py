# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 21:31:12 2022

@author: Alexa
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# tolerance for Newton's method
NTOL = 10**(-6)
# dimensionality of system
d = 1

# f = function
# J = Jacobian of f
# theta = theta
# t0 = initial time
# tf = final time
# h = step size
# y0 = initial value
# M = maximum number of iterations for Newton's method
def ThetaMethod(f,J,theta,t0,tf,h,y0,M):
    N = int((tf - t0) / h) 
    y = np.zeros((N+1,d))
    # tally for the max number of iterations Newton's method took
    nutally = np.zeros(N)
    y[0,:] = y0 
    
    for n in range(1,N):
        # Newton's iteration
        solutionfound = False 
        # initialize iteration at previous step
        Y = y[n-1,:]
        
        nu = 0
        A = -(np.identity(d) - h*(1-theta)*J(t0+(n)*h,y[n,:]))
        for m in range(M): # hope that it converges in M iterations
            b = Y - y[n-1,:] - h*( theta*f(t0+(n-1)*h,y[n-1,:]) + (1-theta)*f(t0+n*h,Y) )
            delta = np.linalg.solve(A,b)
            Y = Y + delta
            nu = nu + 1
            
            # break if delta is small enough
            if np.linalg.norm(delta)<=NTOL:
                solutionfound = True
                # this tallies how many iterations it took for Newton's method to converge
                nutally[n] = nu 
                break
            # end of if statement
        # end of while loop for Newton's iteration
        
        # if we ran all the way up to M, Newton's iteration did not converge in M iterations
        if solutionfound == False:
            print("Newton's iteration did not converge in " + str(M) + " iterations.")
            break
        # end of if statement
        
        y[n,:] = Y
    # end of for loop in n
    
    if solutionfound == False:
        numax = M
    else:
        numax = np.amax(nutally)
    
    return y, numax
# end of ThetaMethod

# functions and jacobians
def f(t,y):
    return ((y**2) - (y**4)*(1 + np.cos(t)**2))
# end of f

def J(t,y):
    return (2*y - 4*(y**3)*(np.sin(t)**2))
# end of J

def f1(t,y):
    return (y**2) - 2*(y**4)
# end of f1

def J1(t,y):
    return 2*y - 8*(y**3)
# end of J1

def f2(t,y):
    return (y**2) - (y**4)
# end of f1

def J2(t,y):
    return 2*y - 4*(y**3)
# end of J1

# maximum number of iterations for Newton's method
M = 20

# set up
y0 = 1/2
h = -0.01
t0 = 0
tf = -100
theta = 1/2

start = timer()
y, numax = ThetaMethod(f,J,theta,t0,tf,h,y0,M)
end = timer()
time = end - start
print("Run time = " + str(time))
print("Max number of iterations for Newton = " + str(numax))

# plotting
T = np.linspace(t0,tf,int((tf - t0) / h)+1)

plt.figure()
plt.title("$y_0$ = " + str(y0) + ", h = " + str(h) + ", theta = " + str(theta))
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.plot(T,y)