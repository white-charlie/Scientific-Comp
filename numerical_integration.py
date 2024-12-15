#################################################################
## Functions to carry out numerical integration
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
#################################################################
import numpy as np
import matplotlib.pyplot as plt

#%% Question 1
def composite_trapezium(a,b,n,f):

    #Set width 'h' of each subinterval using the given formula
    h = (b-a) / n
    
    #Initialise size of approximation
    integral_approx_CTR = (f(a)+f(b)) / 2

    #Create loop and sum over the function evaluated at each subinterval
    for i in range(1,n):
        integral_approx_CTR += f(a+i*h)

    #Multiply by h
    integral_approx_CTR *= h
    
    return integral_approx_CTR

#%% Question 2 (a)
def composite_simpson(a,b,n,f):
    
    '''
    Parameters
    ----------
    a, b : floats
        endpoints of the interval of integration
    n : int
        number subintervals/strips
    f : function of one variable
        function to approximate the integral of

   Returns
   -------
   integral_approx_CSR : floating point number
       the integral approximation using the composite simpson's rule
    '''

    #Set width 'h' of each subinterval using the given formula
    h = (b-a)/(2*n)
    
    integral_approx_CSR = f(a) + f(b)
    
    #Create loop and calculate the 'odd sum' part of the approximation
    for j in range(1,n):
        
        integral_approx_CSR += 2*f(a+ (2*j*h))
        
    #Create loop and calculate the 'even sum' part of the approximation   
    for j in range(1,n+1):
        
        integral_approx_CSR += 4*f(a + (2*j-1)*h)
    
    #Calculate the Composite Simpson's Rule integral approximation by multiplying through by h/3
    integral_approx_CSR *= h/3
        
    return integral_approx_CSR
#%% Question 2 (b)
def compute_errors_integral(N,f,a,b,true_val):

    '''
    Parameters
    ----------
    a, b : floats
        endpoints of the interval of integration
    N : list
        list of different numbers of subintervals/strips
    f : function of one variable
        function to approximate the integral of
    true_val : float
        the true numerical value of the integral
        
   Returns
   -------
   errors_trapezium : array
        the absolute values of the errors for the composite trapezium rule
    errors_simpson : array
        the absolute values of the errors for the composite simpson's rule
    fig : plot
        a plot of the errors for both the composite trapezium rule and composite simpson's rule against the number of subintervals n
        
    Results
    -------
    The composite simpson's rule generally yields more accurate approximations than the composite trapezium rule.
    This is because it uses quadratic polynomials to approximate between each subinterval , giving a more accurate represention of the curve.
    This is consistent with the resultsin each array.
    The composite simpson's rule error is smaller for each n, and diminishes at a quicker rate than the composite trapezium rule errors for the same values of n.'
    '''
    
    #Initialise zeros matrices of the same length as N
    errors_trapezium = np.zeros(len(N))
    errors_simpson = np.zeros(len(N))
    
    #Loop over enumerated data
    for i,n in enumerate(N):
        
        #Call the prior functions
        approx_trapezium = composite_trapezium(a, b, n, f)
        approx_simpson = composite_simpson(a, b, n, f)

        #Compute errors for the Composite Trapezium Rule
        CTR_errors = np.abs(true_val - approx_trapezium)
        errors_trapezium[i] = CTR_errors       
        
        #Compute errors for the Composite Simpson's Rule     
        CSR_errors = np.abs(true_val - approx_simpson)
        errors_simpson[i] = CSR_errors
    
    #Create a new figure
    fig = plt.figure() 
    
    #Plot errors against number of subintervals with axes and legends
    plt.loglog(N, errors_trapezium, label ='Composite Trapezium rule errors', color='c')
    plt.loglog(N, errors_simpson, label ='Composite Simpsons rule errors', color='m')  
    plt.xlabel('Number of subintervals $n$')
    plt.ylabel('Errors')
    plt.title('Errors against number of subintervals')
    plt.legend()
    plt.show()

    return errors_trapezium,  errors_simpson, fig

#%% Question 3 (a)
def double_integral_trapezium(f,x_range,y_range,m,n,true_val):
    
    '''
    Parameters
    ----------
    x_range, y_range b : lists
        lists containing the lower and upper limits of integration for the x and y variable respectively
    f : function of two variables
        function to approximate the double integral of
    true_val : float
        the true numerical value of the double integral
    m : int
        number of subintervals into which the range of x is divided
    n : int
        number of subintervals into which the range of y is divided
        
   Returns
   -------
   double_CTR : float
        approximation for the double integral using the composite trapezium rule
    errors_CTR : float
        the absolute value of the errors for the composite trapezium rule
    '''

    #Define a new function, setting y constant, to calculate the inner integral
    def integral_inner_T(y):
       
        #Call the omposite_trapezium function setting the limits of integration accordingly
        return composite_trapezium(x_range[0], x_range[1], m, lambda x: f(x,y))

    #Calculate the approximation by calling composite_trapezium again and setting the function as the inner integral that was just calculated
    double_CTR = composite_trapezium(y_range[0], y_range[1], n, integral_inner_T)   
    
    #Define the error as the absolute value of the difference between the true value and the approximation
    error_CTR = np.abs(true_val - double_CTR)
    
    return double_CTR, error_CTR


#%% Question 3 (b)
def double_integral_simpson(f,x_range,y_range,m,n,true_val):
    
    '''
    Parameters
    ----------
    x_range, y_range b : lists
        lists containing the lower and upper limits of integration for the x and y variable respectively
    f : function of two variables
        function to approximate the double integral of
    true_val : float
        the true numerical value of the double integral
    m : int
        number of subintervals into which the range of x is divided
    n : int
        number of subintervals into which the range of y is divided
        
   Returns
   -------
   double_CSR : float
        approximation for the double integral using the composite simpson's rule
    errors_CSR : float
        the absolute value of the errors for the composite simpson's rule
    '''

    #Define a new function, this time to calulate the inner integral of the Simpson's Rule approximation
    def integral_inner_S(y):
        
        #Call the composite_simpson function setting limits accordingly and ensuring that f is only a function of x (y constant)
        return composite_simpson(x_range[0], x_range[1], m, lambda x: f(x,y))
    
    #Calculate the approximation by calling composite_simpson again and setting the function as the inner integral
    double_CSR = composite_simpson(y_range[0], y_range[1], n, integral_inner_S)
    
    #Define the error as the absolute value of the difference between the true value and the approximation   
    error_CSR = np.abs(true_val - double_CSR)
    
    return double_CSR, error_CSR


#%% Question 4
def runge_kutta(a,b,f,N,y0,m,method):
    
    '''
    Parameters
    ----------
    a, b : floats
        endpoints of the interval of integration
    N : int
        number of time steps to take
    f : function
        m-vector function describing the system of ODE's to be solved
    y0 : array
        contains the initial conditions for the dependent variables
    m : int
        the number of dependent variables
    method : int
        determines which of the Runge Kutta Methods to use
        1) the forward Euler method
        2) the modified Euler method
        4) the RK4 method
        
    Returns
    -------
    t : array
        an array of the time steps that the system is evaluated at
    y : array
        an array of the solutions to the system of ODE's
    '''
    
    #Set width 'h' of each step
    h = (b-a)/N      
    
    #Create N+1 equally spaced points over the interval [a,b]
    t = np.linspace(a,b,N+1)
    
    #Initialise zeros matrix for y values, ensuring it has the correct shape
    y = np.zeros((m,N+1))

    #Establish initial conditions
    y[:,0]=y0
    t[0]=a

    #Loop over the N time-steps
    for i in range(N):
        
        #Calculate t and y for method 1 - the forward Euler method
        if method ==1:
            
            y[:,i+1] = y[:,i] + h*f(t[i],y[:,i])        
            t[i+1] = t[i] + h
            
        #Calculate t and y for method 2 - the modified Euler method    
        elif method ==2:
            
            Q = f(t[i+1], y[:,i] + h*f(t[i],y[:,i]))
            y[:,i+1] = y[:,i] + (h/2) * (f(t[i],y[:,i])+Q)
    
        #Calculate t and y for method 4 - the RK4 method
        elif method ==4:
            
            k1 = h*f(t[i], y[:,i])
            k2 = h*f(t[i]+h/2, y[:,i]+k1/2)
            k3 = h*f(t[i]+h/2, y[:,i]+k2/2)
            k4 = h*f(t[i]+h, y[:,i]+k3)
            
            y[:,i+1] = y[:,i] + ((k1/6) + (k2/3) + (k3/3) + (k4/6))
                  
    return t, y
