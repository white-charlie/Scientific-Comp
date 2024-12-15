### Import modules ###
import numpy as np
import matplotlib.pyplot as plt
### No further imports should be required ###

#%% Question 1
def lagrange_poly(p, xhat, n, x, tol):
    
    # Initialize zero matrix
    lagrange_matrix = np.zeros((p+1, n))
    
    #Loop over nodal points
    for i in range(p+1):
        
        #Define initial Li_xj
        Li_xj = 1
        
        for k in range(p+1):
            
            if k != i:
                
                #Calculate ijth entry of the Lagrange matrix
                Li_xj *= (x-xhat[k])/(xhat[i]-xhat[k])
                    
        lagrange_matrix[i] = Li_xj
            
    #Ensure that our nodal points are distinct and set the error flags accordingly
    if len(set(xhat)) != len(xhat):
        error_flag = 1    

    else:
        error_flag = 0
    
    return lagrange_matrix, error_flag


#%% Question 2
def uniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    
    fig = None
    
    #Generate evenly spaced numbers over the interval [a,b]
    xhat = np.linspace(a,b,p+1)
    
    #Call lagrange_poly function and specify tol
    lagrange_matrix, error_flag = lagrange_poly(p,xhat,n,x,tol=1.0e-10)

        #Calculate interpolant
    interpolant = np.dot(f(xhat), lagrange_matrix)   

    #Plot interpolant and original function
    if produce_fig:
        fig=plt.figure()
        plt.plot(x, f(x), label='Original function $f(x)$', color='g')
        plt.plot(x, interpolant, label ='Interpolant $pp_x$', color ='r')
        
        #Add labels, title and legend
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Uniform Polynomial Interpolation')
        plt.legend()
        plt.show()

    return interpolant, fig
    
#%% Question 3

def nonuniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    
    fig = None
    
    #Initialise zeros matrix with the same size as x
    interpolant = np.zeros_like(x)
    
    #Set xhat so that it is llinearly scaled, adherring to the mapping M
    xhat = (a+b)/2 + (b-a)/2 * np.cos(np.pi*((2*np.arange(p+1)+1)/ (2*(p+1))))
    
    lagrange_matrix, error_flag = lagrange_poly(p, xhat, n, x, tol= 1.0e-10)
    

    #Calculate interpolant
    interpolant = np.dot(f(xhat), lagrange_matrix)
        
    #Plot interpolant and original function
    if produce_fig:
            
        fig = plt.figure()  
        plt.plot(x, f(x), label ='Original function $f(x)$', color = 'g')
        plt.plot(x, interpolant, label ='Interpolant $pp_x$', color='r')
        
        #Add labels, title and legend
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Non-Uniform Polynomial Interpolation')
        plt.legend()
        plt.show()
    
    return interpolant, fig
    
    
#%% Question 4

def compute_errors(a,b,n_p,P,f):
    
    """
    a) When compute_errors is run with f(x) = e^x
    
    . Initially, the uniform error decreases rapidly, however, after reaching the polynomialdegree of around 13, this error begins to increase
    . However, this increase in the error still remains relatively low when compared to approximations
    of lower degrees
    
    . The non-uniform error also starts by declining rapidly as the degree increases, almost matching that of the uniform error
    . Around P=13, the behaviour of the error also changes, but instead of increasing, it plateaus and remains under 10^-14
    
    . These results are generally consistent with the notion that higher degree approximations are generally more accurate, with smaller errors
    
    b) When compute_errors is run with f(x) = sin(πx) + 1/10 cos(8πx)
    
    . The uniform error stays relatively constant to begin with, at around 1
    . It then starts to increase erratically at around P=10, before slowly deceasing from the 25th to 40th degree
    
    . The non-uniform error also remains relatively constant, up until around P=20 but then begins to decrease up until the 40th poynomial degree
    
    . This suggests that the non-uniform approximations are generally more accurate than the uniform approximations
    """
    n = 2500
    x =np.linspace(a,b,2500) 
    
    #Initialise zeros error matrix of size (2,n_p)
    error_matrix = np.zeros((2,n_p))
    
    #Loop over enumerated data
    for i, p in enumerate(P):
           
        #Calculate error matrix for uniform and non-uniform interpolants
        interpolant_u, _ = uniform_poly_interpolation(a, b, p, n, x, f, False)
        error_matrix[0,i] = np.max(np.abs(interpolant_u - f(x)))
    
        interpolant_nu, _ = nonuniform_poly_interpolation(a, b, p, n, x, f, False)
        error_matrix[1,i] = np.max(np.abs(interpolant_nu - f(x)))

    #Plot figure
    fig = plt.figure() 
    plt.semilogy(P, error_matrix[0], label ='Uniform error', color = 'c')
    plt.semilogy(P, error_matrix[1], label = 'Non-Uniform Error', color = 'm')
    plt.xlabel('P')
    plt.ylabel('Error')
    plt.title('Error Approximations')
    plt.legend()
    plt.show()
    
    return error_matrix, fig

#%% Question 5

def piecewise_interpolation(a,b,p,m,n,x,f,produce_fig):
    
    #Set width w
    w = (b-a)/m
    
    #Initialise zeros matrix
    p_u_interpolant = np.zeros(n)
    p_nu_interpolant = np.zeros(n)
    solution = []
    
    #Loop over subintervals
    for i in range(m):
        
        #Define endpoints of subinterval
        subinterval_a = a+(i*w)
        subinterval_b = a+((i+1)*w)
        
        #Initialise list
        solution = np.where(np.logical_and(subinterval_a <= x, x <= subinterval_b))[0]
        
        #Calculate interpolants
        interpolant_u, _= uniform_poly_interpolation(subinterval_a, subinterval_b, p, n, x, f, False)

        interpolant_nu, _= nonuniform_poly_interpolation(subinterval_a, subinterval_b, p, n, x, f, False)
        
        # Assign interpolant values to appropriate indices
        p_u_interpolant[solution] = interpolant_u[solution]
        p_nu_interpolant[solution] = interpolant_nu[solution]     
        
   #Plot figures
    if produce_fig:
      
        fig, ax = plt.subplots()
        ax.plot(x, f(x), label='Original function $f(x)$', color='g')
        ax.plot(x, p_u_interpolant, label='Uniform interpolant $S_m^p(x)$', linestyle='dashed', color='b', marker ='s')
        ax.plot(x, p_nu_interpolant, label='Non-uniform interpolant $S_m^p(x)$', linestyle='dashed', color='r', marker ='^')       
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Piecewise Polynomial Interpolation')
        plt.legend()
        plt.show()
        
    else:
        
        fig = None
    
    return p_u_interpolant, p_nu_interpolant, fig


#%% Question 6

def compute_piecewise_errors(a, b, n_m, M, n_p, P, f):
    
    """
    a) When compute_piecewise_errors is run with f(x) = sin(πx) + 1/10 cos(8πx)
    
    . Initially, the uniform errors for all values of p in P stay relatively level for the first 10^1 subintervals m
    . At this point, the errors begin to fall, with higher values of p showing smaller errors than the smaller values of p
    . This is consistent with the notion that increasing the polynomial degree p provides more accurate approximations of functions within Piecewise Polynomial Interpolation, especially for higher numbers of subintervals m
    
    . The non-uniform errors of f(x) display an almost identical set of errors to the uniform errors
    . This is likely due to the nonuniform_poly_interpolation function utilising a trigonometric function to approximate the function
    
    b) When compute_piecewise_errors is run with f(x) = np.sqrt(|x-1/3|)
    
    . The uniform errors when approximating this function begin at around 10^0 and decrease gradually as the number of subintervals increases
    . The errors for p=3 and p=6 end up abandoning this trend and decrease further than the rest, with the error for p=6 decreasing moreso
    
    . The non-uniform errors for this function however, show a linear decline on the logarithmic scale, with p=3 displaying the smallest error out of the 6 polynomial degrees 
    
    """
    
    #Initialise zeros matrices
    u_error_matrix = np.zeros((n_m, n_p))
    nu_error_matrix = np.zeros((n_m, n_p))
    
    #Create figures
    u_fig = plt.figure()
    nu_fig = plt.figure()
    
    #Set 2500 linearly distributed nodal points that can be called upon in piecewise_interpolation
    x = np.linspace(a,b,2500)
 
    #Set loops over enumerated data
    for l, m in enumerate(M):
     
        for k, p in enumerate(P):
            
            # Compute piecewise interpolants
            p_u_interpolant, _, _ = piecewise_interpolation(a, b, p, m, len(x), x, f, False)
            _, p_nu_interpolant, _ = piecewise_interpolation(a, b, p, m, len(x), x, f, False)
            
            # Compute errors for uniform and nonuniform interpolants
            u_error = np.max(np.abs(p_u_interpolant - f(x)))
            nu_error = np.max(np.abs(p_nu_interpolant - f(x)))
            
            # Store errors in error matrices
            u_error_matrix[l, k] = u_error
            nu_error_matrix[l, k] = nu_error
    
    # Plot uniform errors
    for k,p in enumerate(P):
        plt.figure(u_fig.number)
        plt.loglog(M, u_error_matrix[:, k], label=f'P={p}')
        
    # Plot nonuniform errors
        plt.figure(nu_fig.number)
        plt.loglog(M, nu_error_matrix[:, k], label=f'P={p}')        
        
    #Add labels, titles and legends  
    plt.figure(u_fig.number)
    plt.xlabel('Number of Subintervals (M)')
    plt.ylabel('Error')
    plt.title('Uniform Errors')
    plt.legend()
    
    plt.figure(nu_fig.number)
    plt.xlabel('Number of Subintervals (M)')
    plt.ylabel('Error')
    plt.title('Non-uniform Errors')
    plt.legend()
    
    return u_error_matrix, nu_error_matrix, u_fig, nu_fig
