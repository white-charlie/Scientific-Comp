"""
MATH2019 CW2 systemsolvers module

@author: Charlie White
"""
#%% Question 1

import numpy as np
import matplotlib.pyplot as plt
import warmup_solution as ws


def find_max(M,n,i):
    
    m=0
    p=i
    q=i
    
    for k in range(i,n):
        
        for j in range(i,n):
            
            if abs(M[j,k])>m:
                
                m = abs(M[j,k])
                p=j
                q=k
    
    return p, q

#%% Question 2

def complete_pivoting(A,b,n,c):
    
    M=np.hstack((A,b))
    s=np.arange(n)

    for i in range(c):
        
        p, q = find_max(M,n,i)

        M[[i,p],:] = M[[p,i],:]
        M[:,[i,q]] = M[:,[q,i]]
        s[[i,q]] = s[[q,i]]
        
        if M[i,i]!=0:            
            for j in range(i+1,n):
                m = M[j,i]/M[i,i]
                M[j,i:]-= m * M[i,i:]

    return M, s

#%% Question 3

def complete_solve(A,b,n):
    
    M,s=complete_pivoting(A,b,n,n-1)
    y=ws.backward_substitution(M,n)
    x=np.zeros((n,1))
    
    for i in range(n):
        x[s[i]]=y[i]
    
    return x

#%% Question 4

def weighted_Jacobi(A,b,n,x0,w,N):
    
    x_approx=np.zeros([n,N+1])

    L = -np.tril(A,-1)
    U = -np.triu(A, 1)
    D = np.diag(np.diag(A))
    T = np.linalg.inv(D)@ (L+U)
    c = np.linalg.inv(D)@ b
    x = x0
    
    for k in range(1,N+1):
        x = w*(T@x + c) + (1-w)*x
        x_approx[:, k ] = x.flatten()

    return x_approx

#%% Question 5

def weighted_Jacobi_plot(A,b,n,x0,weights,M,N):

    k=np.arange(N+1)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$||x-x^{(k)}||_\infty$')
    ax.grid(True)
    
    for i in range(M):
        
        e=[]
        w=weights[i]
        
        for j in range(N+1):

            x_approx = weighted_Jacobi(A, b, n, x0, w, j)
            x_real = ws.no_pivoting_solve(A, b, n)
            u = np.subtract(x_real, x_approx)
            x_absolute = np.linalg.norm(u,np.inf)
        
            e.append(x_absolute)
        
        ax.plot(k ,e, label = f'Weight {w}', linestyle = '--')
            
    ax.legend()
        
    return fig

