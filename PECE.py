# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:11:08 2022

@author: Alexa
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# PECE Method w/ AB2 and AM2

d = 1 # dimensionality of system
ETOL = 10**(-6) # error tolerance
alpha = 0.9 # fraction used to determine next step size
p = 2 # order of the methods used

# pick False for fixed step size
variable_stepsize = True

# choose to interpolate at the final time or not
interpolate = True

# short hand for AB2
def AB2(y,t1,t2,t3,f1,f2):
    hn = t1 - t2
    hn_1 = t2 - t3
    return y + hn*f1 + ((f1-f2)/(hn_1))*((hn**2)/2)
# end of AB2

# short hand for AB2 (constant stepsize)
def AB2C(y,h,f1,f2):
    return y + h*f1 + ((f1-f2)/(h))*((h**2)/2)
# end of AB2

# short hand for AM2
def AM2(y,h,f1,f2):
    return y + h*(f1 + f2)/2
# end of AM2

# interpolant for AM2
def phi(t0,tn_1,tn,fn_1,fn):
    interpolant = fn + ( ( fn - fn_1 ) / ( tn - tn_1 ) ) * ( t0 - tn )
    return interpolant
# end of phi

# fixed step size
def PECE1(t0,tf,h,f,y0):
    N = int((tf - t0) / h)
    
    y = np.zeros((N+1,d))
    y[0] = y0 
    
    t = np.zeros(N+1)
    t[0] = t0
    t[1] = t0 + h
    
    F = np.zeros((N+1,d))
    F[0] = f(t0,y0) 
    
    # must manually compute y1, use Backward Euler 
    yn0 = AB2C(y0,h,f(t[0],y0),y0-h*f(t[0],y0))
    fn0 = f(t[1],yn0)
    y[1] = AM2(y[1-1],h,f(t[1-1],y[1-1]),fn0)
    F[1] = f(t[1],y[1])
    
    for n in range(2,N+1):
        t[n] = t[n-1] + h
        
        yn0 = AB2C(y[n-1],h,F[n-1],F[n-2])
        fn0 = f(t[n],yn0)
        y[n] = AM2(y[n-1],h,F[n-1],fn0)     
        F[n] = f(t[n],y[n])
    # end of for loop in n
    
    # return y values, t values, and number of steps taken
    return y, t, N
# end of PECE1

# variable step size
def PECE2(t0,tf,h,f,y0):
    # for storing the old values
    # since the number of steps to be taken is not pre-determined, 
    # the following arrays must be appended with the new values
    y = np.zeros((2,d)) # store y(t) values
    y[0] = y0 # initialize y
    t = np.array([t0,t0+h]) # store t values
    F = np.array([f(t0,y0),0]) # store f(t,y) values

    
    # must manually compute y1, use Backward Euler 
    yn0 = AB2C(y0,h,f(t[0],y0),y0-h*f(t[0],y0))
    fn0 = f(t[1],yn0)
    y[1] = AM2(y[1-1],h,f(t[1-1],y[1-1]),fn0)
    F[1] = f(t[1],y[1])
    
    n = 2 
    tn = t[n-1] + h
    
    while tn < tf:
        # carry out the PEC step once
        yn0 = AB2(y[n-1],tn,t[n-1],t[n-2],F[n-1],F[n-2])
        fn0 = f(tn,yn0)
        yn = AM2(y[n-1],h,F[n-1],fn0)
        
        LTE = (5/4)*np.linalg.norm(yn-yn0) # local truncation error
        r = ((alpha*ETOL)/(h*LTE))**(1/(p+1)) # r needed to get under ETOL
        
        # repeat PEC step until we get desired error estimate
        while h*LTE > ETOL:
            h = r*h # update h
            yn0 = AB2(y[n-1],t[n-1]+h,t[n-1],t[n-2],F[n-1],F[n-2])
            fn0 = f(tn,yn0)
            yn = AM2(y[n-1],h,F[n-1],fn0)
            
            LTE = (5/4)*np.linalg.norm(yn-yn0)
            r = ((alpha*ETOL)/(h*LTE))**(1/(p+1))
        # end of while loop
        
        # update h, y, t, and F for next step
        h = r*h
        y = np.append(y,yn)
        t = np.append(t,tn)
        F = np.append(F,f(tn,yn))
    
        n = n + 1
        tn = t[n-1] + h
    # end of while loop 
    
    # manually put in value at tf via interpolant 
    if interpolate == True:
    # since we need the value of f at t_n+1, we perform the PECE computation one last time
        yn0 = AB2(y[n-1],tn,t[n-1],t[n-2],F[n-1],F[n-2])
        fn0 = f(tn,yn0)
        yn = AM2(y[n-1],h,F[n-1],fn0)
        
        LTE = (5/4)*np.linalg.norm(yn-yn0) # local truncation error
        r = ((alpha*ETOL)/(h*LTE))**(1/(p+1)) # r needed to get under ETOL
        
        # repeat PEC step until we get desired error estimate
        while h*LTE > ETOL:
            h = r*h # update h
            yn0 = AB2(y[n-1],t[n-1]+h,t[n-1],t[n-2],F[n-1],F[n-2])
            fn0 = f(tn,yn0)
            yn = AM2(y[n-1],h,F[n-1],fn0)
            
            LTE = (5/4)*np.linalg.norm(yn-yn0)
            r = ((alpha*ETOL)/(h*LTE))**(1/(p+1))
        # end of while loop
        
        # interpolated value
        yf = phi(tf,t[n-1],tn,y[n-1],yn)
        t = np.append(t,tf)
        y = np.append(y,yf)
        n = n + 1
    
    # return y values, t values, and number of steps taken
    return y, t, n-1
# end of PECE2


# Examples
t0 = 0
tf = 100
y0 = 1/2
h = 0.01

def f(t,y):
    return (y**2) - (1+np.cos(t)**2)*(y**4)
# end of f

def f1(t,y):
    return (y**2) - 2*(y**4)
# end of f1

def f2(t,y):
    return (y**2) - (y**4)
# end of f2

# Plotting
if variable_stepsize == True:
    name = str("Variable Step Size for ETOL = " + str(ETOL))
    start = timer()
    y, t, N = PECE2(t0,tf,h,f,y0)
    end = timer()
    time = end - start
else:
    name = str("Fixed Step Size")
    start = timer()
    y, t, N = PECE1(t0,tf,h,f,y0)
    end = timer()
    time = end - start
# end of if-else statement for choosing between fixed and variable stepsize

print("Run time for " + name + ": " + str(time))
print("Number of steps taken: " + str(N))
print("Final time: " + str(t[N]))
print("Value at final time: " + str(y[N]))

plt.figure()
plt.title(name)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.plot(t,y)

# plot step size vs. time
# specifically, this is plotting h_n vs. t_{n-1}, where h_n = t_n - t_{n-1}
# of course this is only relevant for the variable step size methods
# that said, this does work with the fixed stepsize PECE, just in case!
step_sizes = np.zeros(len(t)-1)
for n in range(len(step_sizes)):
    step_sizes[n] = t[n+1] - t[n]
# end of for loop in n

plt.figure()
plt.xlabel('$t$')
plt.xlabel('$h$')
plt.title("Step Size vs. Time for ETOL = " + str(ETOL))
plt.plot(t[:-1],step_sizes)
print("Maximum step size: " + str(np.amax(step_sizes)))
print("Minimum step size: " + str(np.amin(step_sizes)))
print("Average step size : " + str(np.average(step_sizes)))