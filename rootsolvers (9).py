"""
MATH2019 CW1 rootsolvers module

@author: Charlie White
"""

import numpy as np

def bisection(f,a,b,Nmax):
    
    """
    Bisection Method: Returns a numpy array of the 
    sequence of approximations obtained by the bisection method.
    
    Parameters
    ----------
    f : function
        Input function for which the zero is to be found.
    a : real number
        Left side of interval.
    b : real number
        Right side of interval.
    Nmax : integer
        Number of iterations to be performed.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)

    for i in range(Nmax):
        
        p = (a+b)/2
        p_array[i]=p
        
        if f(p) * f(a) >0:
            a=p
            
        else:
            b=p

    return p_array

#%% Question 2

import numpy as np

def bisection(f,a,b,Nmax):
    
    p_array=np.zeros(Nmax)
    
    for i in range(Nmax):
        
        p = (a+b)/2
        p_array[i]=p
        
        if f(p) * f(a) >0:
            a=p
            
        else:
            b=p

    return p_array


#%% Question 3

import numpy as np

def fp_iteration(g,p0,Nmax):
    
    """
    Fixed-point iteration method: returns an array of approximations for a root
    of g, given p0 and Nmax.
    
    Parameters
    ----------
    g : function
        Input function.
    p0 :real number
        Initial approximation to start the fixed poit iteration.
    Nmax : integer
        Number of iterations performed.
    Returns
    -------
    p_array : array
        Array of the resulting approximations.
    """
    
    p_array=np.zeros(Nmax)
    
    for i in np.arange(Nmax):
        
        p0 = g(p0)
        p_array[i] = p0
        
    return p_array

#%% Question 4

import numpy as np

def fp_iteration_stop(g,p0,Nmax,TOL):
    
    """
    The fixed-point iteration method with an added stopping criterion. 
    The iterations cease once the error is within a set parameter TOL, or 
    until Nmax iterations have been performed.
    
    Parameters
    ----------
    g : function
         Input function.
    p0 :real number
        Initial approximation to start the fixed poit iteration.
    Nmax : integer
        Number of iterationa performed.
    TOL : real number
        The iterations stop if the error is less than this value TOL.
    Returns
    -------
    p_array : array
        Array of the resulting approximations.
    pdiff_array : array
        Array of values of the differences between each approximation and the
        true value of p.
    """
    
    p_array = np.zeros(Nmax)
    pdiff_array = np.zeros(Nmax)
    
    for i in np.arange(Nmax):
        
        #set up p_array
        p=g(p0)
        p_array[i]=p  
        #set up pdiff_array
        pd = (p-p0)
        pdiff_array[i]=pd
    
        if np.abs(p-p0)<= TOL:
            break
    
        p0=p
        
    return p_array[:i+1], pdiff_array[:i+1]

#%% Question 5

import numpy as np

def newton_stop(f,dfdx,p0,Nmax,TOL):
    
    """
    Newton's Method: returns an array of approximations of the root of some 
    function f for Nmax iterations or until the resulting value has an error 
    less than or equal to some other value TOL.'
    
    Parameters
    ----------
    f : function
        Input function.
    dfdx : function
        Computes the derivative of a function f.
    p0 : real
        Initial approximation at the start of the Newton's Method iterative 
        process.
    Nmax : integer
        Number of itegers performed.
    TOL : real number
        The iterations stop if the error is less than this value TOL.
    Returns
    -------
    p_array : array
        Array of the resulting approximations.
    """
        
    p_array = np.zeros(Nmax)
    
    i=0  
    for i in np.arange(Nmax):
        
        p=p0-f(p0)/dfdx(p0)
        p_array[i]=p       
        p0=p

        if abs(f(p)) < TOL:         
            break
        
        if dfdx(p) ==0:
            break

    return p_array[:i+1]
        
#%% Question 6

import matplotlib.pyplot as plt

def plot_convergence(p,f,dfdx,g,p0,Nmax):
# Fixed - point iteration
    p_array = fp_iteration(g,p0,Nmax)
    e_array = np.abs(p - p_array)
    n_array = 1+np.arange(np.shape(p_array)[0])
    
# Newton method
    p1_array = newton_stop(f,dfdx,p0,Nmax,10**(-14))
    e1_array = np.abs(p-p1_array)
    n1_array = 1+np.arange(np.shape(p1_array)[0])
    
# Preparing figure  
    fig,ax=plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("|p-p_n|")
    ax.set_title(" Convergence behaviour ")
    ax.grid (True)
# Plot
    ax.plot(n_array,e_array,"o",label ="FP iteration ",linestyle="--",color="r")
    ax.plot(n1_array,e1_array,"h",label="Newton's Method", linestyle="--",color="b")
# Add legend
    ax.legend() ;
    return fig,ax

#%% Question 7

import numpy as np

def steffensen_stop(f,p0,Nmax,TOL):
    
    p_array=np.zeros(Nmax)
    
    q=lambda x:(f(x+f(x))/f(x))-1
    
    i=0
    for i in np.arange(Nmax):
        
        p = p0 - f(p0)/q(p0)
        p_array[i]=p
        p0=p
        
        if np.abs(f(p))<TOL:
            break
    
    return p_array[:i+1]
 
#%% Question 8

import matplotlib . pyplot as plt

def plot_convergence3(p,f,dfdx,g,p0,Nmax):
    
    # Fixed - point iteration
        p_array = fp_iteration(g,p0,Nmax)
        e_array = np.abs(p - p_array)
        n_array = 1+np.arange(np.shape(p_array)[0])
        
    # Newton method
        p1_array = newton_stop(f,dfdx,p0,Nmax,10**(-14))
        e1_array = np.abs(p-p1_array)
        n1_array = 1+np.arange(np.shape(p1_array)[0])
        
    # Steffensen method
        p2_array = steffensen_stop(f,p0,Nmax,10**(-14))
        e2_array = np.abs(p-p2_array)
        n2_array = 1+np.arange(np.shape(p2_array)[0])
    
    # Preparing figure
        fig,ax=plt.subplots()
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("|p-p_n|")
        ax.set_title(" Convergence behaviour ")
        ax.grid (True)
    # Plot
        ax.plot(n_array,e_array,"o",label ="FP iteration ",linestyle="--",color="r")
        ax.plot(n1_array,e1_array,"h",label="Newton's Method",linestyle="--",color="b")
        ax.plot(n2_array,e2_array,"*",label="Steffenson Method", linestyle="--",color="g")
        
    # Add legend
        ax.legend() ;
    
    
        return fig , ax
    
    