#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:00:48 2025

@author: louisebertelsen
"""

# import packages and set directory 

import pandas as pd # to import eg. excel
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D



home = str(Path.home())
Path = home + "/Library/CloudStorage/OneDrive-Personligt/Dokumenter/Universitetet/8. Semester/Micro and Macro Models of the Labour Market/Exam Paper"
datapath = Path + "/Python"

#%% Distributions (no longer needed)

n_types = 50

# income_grid = np.linspace(1/n_types/2,1-1/n_types/2,n_types)

# Creating distributions for male and female types as alternative to data from class
# male_density = np.exp(-0.5 * (income_grid - 5) ** 2)
# female_density = np.exp(-0.5 * (income_grid - 4.5) ** 2)

# Random normal distribution with mean zero for love shocks
#z_distribution = np.zeros((n_types,n_types))
#np.random.normal(0, 1, size=(n_types, n_types))

#%% import data (distribution of income for male and female)

import_male = pd.read_csv(datapath+"/income_distribution_male.csv").to_numpy(copy=True)
# 50x2 - 50x[1] is income in currency and 50x[2] is the density 
import_female = pd.read_csv(datapath+"/income_distribution_female.csv").to_numpy(copy=True)
# same just for women meaning that 50x[1] is exactly the same as for male 


#%% Define functions 

# Checking whether we integrate to 1
def integrate_uni(values, xstep):
    "integrates the grid with the specified values with stepsize xstep"
    #spacing = x1/values.size
    copy = np.copy(values)
    # make a copy of the array 
    copy.shape = (1, values.size)
    # reshape the copied array as a 1x array size 
    return integrate.simpson(copy, dx=xstep)
    # note that this simpson rule calculate the area under the curve using parabolic slices
    # this is why it makes sence that we need the stepsize
    # se paper to see how it is calculated dx/3 (f0+2*f1+4*f2+2*f3+??)

def integrate_red(matrix, result, xstep): #integrating in 2 dimensions
    n = matrix.shape[0]
    # n = the first dimention in the matrix 
    if result == 'male':
    # if the second input is male then do teh below
        inner = np.zeros((1,n))
        # set inner equal to a vector (1,n) of zeroes
        for i in range(0,n):
        # for each entrance 0,n
            inner[0,i] = np.squeeze(integrate_uni(matrix[:,i],xstep))
            # np.squueze 
    elif result == 'female':
        inner = np.zeros((n,1))
        for i in range(0,n):
            inner[i,0] = np.squeeze(integrate_uni(matrix[i,:],xstep))
    return inner
    # note that this does not return a matrix but a array (vector) 

# production function (simple)
def production_function(x,y):
    return x*y
# we use a simple production function simply x*y meaning that the two imputs we insert will be multiplied 

# Equation (16) and (17) from Jaquemet & Robin 
def flow_update(dens_e, dens_u_o, alphas, c_delta, result, c_lambda_m, xstep): 
    int_u_o = integrate_red(dens_u_o * alphas, result, xstep) 
    # this is only the integral of u times alpha 
    # note this is always done for the opisite sex so for men we integrate u_f(y) * alpha over dy 
    # xstep is the same for both sex so we can simply use xstep as dy as well 
    # u is 1,50 (male) (opisite) and alpha is a matrix of 50x50 --> becomes a 50x50 and then interated --> 1x50 (male (opisite))
    int_u_o.shape = dens_e.shape
    # we make sure int_u_o is the same shape as e_m(x) or e_f(y) they have to be multiplied in the next line 
    # this might be because male runs the integral for women returning a array with the oppisite simentions
    return c_delta*dens_e / (c_delta + c_lambda_m * int_u_o)
    # returns the whole formula with the constants as well 
    
def chebychev_grid(n, z_mean, z_sd):
    """Generates Chebyshev-Lobatto grid points mapped to [a, b].

    Args:
        n (int): Number of intervals (will return n+1 points)
        a (float): Lower bound of the interval
        b (float): Upper bound of the interval

    Returns:
        np.ndarray: Chebyshev grid of shape (n+1,)
    """
    z_min = stats.norm.ppf(0.0001, loc=z_mean, scale=z_sd)
    z_max = -z_min
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)       # Chebyshev nodes in [-1, 1]
    return 0.5 * (z_min + z_max) + 0.5 * (z_max - z_min) * x 

def cc_weights(n):
    """Clenshaw-Curtis Quadrature nodes and weights.

    Args:
        n (int): Number of nodes (n + 1 nodes).

    Returns:
        nodes (np.ndarray): Chebyshev nodes (mapped to [-1, 1]).
        weights (np.ndarray): Clenshaw-Curtis weights.
    """
    # Chebyshev nodes: cos(pi * k / n)
    k = np.arange(n + 1)
    nodes = np.cos(np.pi * k / n)

    # Clenshaw-Curtis weights
    weights = np.pi / n * np.ones(n + 1)
    weights[0] = weights[0] / 2  # Adjust weight for the first node
    weights[-1] = weights[-1] / 2  # Adjust weight for the last node

    return weights

def integr_z(c_xy, s, s_o, gridsize, n_types, z_mean, z_sd):

    #Integrates over z, given the joint home production function C_xy,

    #the value of singlehood of the specified sex (s), the value of singlehood

    #of the other sex (s_o) and the gridsize, which should be used in approximating

    #the integral. Note that this function always uses Chebychev Grids.

    # 50x50x50
    ones = np.ones((n_types, n_types, gridsize))
    
    
    z_min = stats.norm.ppf(0.0001, loc=z_mean, scale=z_sd)
    # -3,719
    z_max = -z_min
    # 3,719

    z_grid = chebychev_grid(gridsize - 1, z_mean, z_sd)
    # (50,) from 3,719 to -3,719

    z_grid.shape = (1, 1, gridsize)
    # 1x1x50

    z_vals = ones * z_grid
    # 50x50x50

    # print(np.shape(z_vals))

    new_order = (1, 0, 2)
    # tuple 

    z_vals_prime = np.transpose(z_vals,axes=new_order)
    # 50x50x50
    # change the order so axis 1 is swaped with axis 0

    # print(np.shape(z_vals_prime))

    z_weights = cc_weights(gridsize - 1)
    # (50,) 0< weights <1
    

    #we need the 2d version of c_xy transposed

    C_xy_prime = np.transpose(c_xy)
    # 50 x 50 

    # Making the joint home production function temporarily three dimensional,
    # so I can integrate over the third dimension.

    tmp_shape_cxy = c_xy.shape
    # tuple (50,50)

    tmp_shape_cxy_prime = C_xy_prime.shape
    # tuple (50,50)

    c_xy.shape = (n_types, n_types, 1)
    # c_xy is now 50x50x1

    C_xy_prime.shape = (n_types, n_types, 1)
    # c_xy_prime is now 50x50x1

    # print(np.shape(C_xy))

    tmp_shape_s = s.shape
    # tuple (1,50)
    

    s.shape = (s.shape[0], s.shape[1], 1)
    # now s_m_1 becomes 1,50,1 

    # print(np.shape(s))

    tmp_shape_so = s_o.shape
    # tuple (50,1)

    s_o.shape = (s_o.shape[0], s_o.shape[1], 1)
    # now s_f_1 becomes 50,1,1 

    # print(np.shape(s_o))

    value = z_vals + z_vals_prime + c_xy + C_xy_prime - s_o - s # change here for new values of singlehood
    # 50x50x50
    
    # max{C(x,y) - s_o, s}

    # sp_pos = np.where(sp < s, s, sp)

    sp_pos = np.where(value > 0, value, 0)

    int_z = (z_max - z_min) / 2 * np.sum(sp_pos * stats.norm.pdf(z_vals, loc=z_mean, scale=z_sd) * z_weights, axis=2)

    # if np.min(int_z)<0:

    #     print(int_z)

    c_xy.shape = tmp_shape_cxy
    C_xy_prime.shape = tmp_shape_cxy_prime
    s.shape = tmp_shape_s
    s_o.shape = tmp_shape_so
    # shape back to the shape it was before
    
    return int_z 
  
def new_18(c_x, lambda_m, beta, r, delta, U_m, z, c_xy, s_f_1, s_m_1, u_f_1, result, xstep, gridsize, n_types, z_mean, z_sd):
    constant = ((lambda_m*beta)/(r+delta))
    denominator = 1 + constant* U_m
    #left = z + c_xy - s_f_1
    #max_1 = np.maximum(left, s_m_1)
    max_1 = integr_z(c_xy, s_m_1, s_f_1, gridsize, n_types, z_mean, z_sd)
    inner_integrand_1 = max_1 * u_f_1
    flow_surplus_1 = integrate_red(inner_integrand_1, result, xstep)
    norminator = c_x + constant * flow_surplus_1
    return norminator / denominator  
    
    
  
#%%

# Define dictionary
p = dict()

# model parameters
#nash-bargening power
p['c_beta'] = 0.5
# discount rate #retter til r = delta
p['c_r']= 0.01
#seperation
p['c_delta']=0.01
p['c_lampda_m']=1


print(np.shape(import_male))
p['xmin']= import_male[0,0] # vi definerer xmin i vores dictionary, p. Nu kan vi fx skrive p['xmin'] for at se xmin
print('Lowest income grid point:', p['xmin'])
p['xmax']= import_male[49,0]
print('Highest income grid point:', p['xmax'])
p['xstep']= import_male[1,0] - import_male[0,0]
print('stepsize:',p['xstep'])

# type space
p['typespace_n'] = import_male[:,0]
p['typespace'] = p['typespace_n']/np.min(p['typespace_n'])

p['n_types']=n_types
p['male_inc_dens'] = import_male[:,1] # kolon angiver at vi tager en hel kolonne
p['female_inc_dens'] = import_female[:,1]


#normalize densities
#density function for all agents 
e_m = p['male_inc_dens'] / integrate_uni(p['male_inc_dens'],p['xstep'])
e_m.shape = (1, p['n_types'])
e_f = p['female_inc_dens'] / integrate_uni(p['female_inc_dens'],p['xstep'])
e_f.shape = (p['n_types'],1)

#u_m = np.ones((1,p['n_types']))
#u_f = np.ones((p['n_types'],1))

xgrid = p['typespace'].ravel() 
ygrid = p['typespace'].ravel()

# initializing c_xy 
c_xy = np.zeros((p['n_types'],p['n_types']))

# flow utilities for couples
for xi in range(p['n_types']):
    for yi in range (p['n_types']):
        #absolute advantage as in shimer/smith
        c_xy[xi,yi]=production_function(p['typespace'][xi], p['typespace'][yi])

# for male         
c_x = np.zeros((1, p['n_types']))
c_y = np.zeros((p['n_types'], 1))
        
        
for xi in range(p['n_types']):
    c_x[0,xi]=xgrid[xi]
    
for yi in range(p['n_types']):
    c_y[yi,0]=ygrid[yi]
    

p['z_dis'] = integr_z(c_xy, c_x, c_y, n_types, n_types, 0, 1)
#p['z_dis']= z_distribution

values_s_m = np.zeros((n_types,n_types))
values_s_f = np.zeros((n_types,n_types))

# Compute the matching matrix
# alphas = (c_xy + z_density >= values_s_m + values_s_f).astype(float)

alphas = np.ones([n_types, n_types])

u_m_1 = np.ones((1, p['n_types']))
u_f_1 = np.ones((p['n_types'], 1))
#s_m_1 = np.ones((1, p['n_types']))
#s_f_1= np.ones((p['n_types'], 1))

# perform the fixed point iteration to calculate the equilibrium 
# maxiter = 1000 # max 1000 iterrations 

keep_iterating = True

#main loop
while keep_iterating:
    e = sys.float_info.max
    u_m_prev = u_m_1
    u_f_prev = u_f_1
    while e > 1e-12:
        
        
        u_m_1 = flow_update(e_m, u_m_prev, alphas, p['c_delta'], 'male', p['c_lampda_m'], p['xstep'])
        u_f_1 = flow_update(e_f, u_f_prev, alphas, p['c_delta'], 'female',p['c_lampda_m'], p['xstep'])
 
 
        e = max(
        np.linalg.norm(u_m_prev - u_m_1),
        np.linalg.norm(u_f_prev - u_f_1)
    )
        
        u_m_prev = u_m_1
        u_f_prev = u_f_1
        
        #print('um',u_m_prev[0,0])
        #print('uf',u_f_prev[0,0])
        
    int_U_m = integrate_uni(u_m_1, p['xstep'])
    int_U_f = integrate_uni(u_f_1, p['xstep'])
    
    int_U_m_p = int_U_m
    int_U_f_p = int_U_f
    
    
    #print('cm',c_x[0,0])
    #print('cf',c_y[0,0])
    #print('int_Um', int_U_m_p)
    #print('int_Uf', int_U_f_p)

    # Equation 18
    s_m_1 = new_18(c_x, p['c_lampda_m'], p['c_beta'], p['c_r'], p['c_delta'], int_U_f_p, p['z_dis'], c_xy, values_s_f, values_s_m, u_f_prev, 'male', p['xstep'], n_types, n_types, 0, 1)
    s_f_1 = new_18(c_y, p['c_lampda_m'], 1-p['c_beta'], p['c_r'], p['c_delta'], int_U_m_p, p['z_dis'], c_xy, values_s_m, values_s_f, u_m_prev, 'female', p['xstep'], n_types, n_types, 0, 1)
    
    #print('sm1',s_m_1[0,0])
    #print('sf1',s_f_1[0,0])
    #print('cxy', c_xy[0,0])
    
    values_s_m = s_m_1
    values_s_f = s_f_1
    
     
    matrix = values_s_m + values_s_f
    
    print('matrix', matrix[0,0])
    
    new_alphas = np.zeros([n_types, n_types])
    for i in range(n_types):
        for j in range(n_types):
            if c_xy[i,j] + p['z_dis'][i,j] >= matrix[i,j]:
                    new_alphas[i,j] = 1
                    
    print('alpha', new_alphas[0,0])
                    
    print (n_types**2 - (new_alphas == alphas).sum())
    
    if (new_alphas == alphas).all():
            is_convergence = True
            keep_iterating = False
    else:
            alphas = new_alphas


#Calculating joint density of matches
n_xy = (p["c_lampda_m"]*u_m_1*u_f_1*alphas)/p["c_delta"]

    
#%% Plotting
    
def wireframeplot2(z, space, azim, elev, title):

    """

    Produces a simple wireframe plot.

    """

    X, Y = np.meshgrid(space, space)

    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(projection='3d')

    ax.plot_wireframe(X, Y, z,

                          rstride=2,

                          cstride=2,

                          color='DarkSlateBlue',

                          linewidth=1,

                          antialiased=True)

    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.set_rotate_label(False)  # disable automatic rotation

    ax.yaxis.set_rotate_label(False)  # disable automatic rotation

#    ax.zaxis.set_rotate_label(False)  # disable automatic rotation

 

#    ax.set_ylabel(r'\textbf{Women}', labelpad=20,rotation='horizontal')

#    ax.set_xlabel(r'\textbf{Men}', labelpad=10,rotation='horizontal')

    ax.set_ylabel('Women', labelpad=20,rotation='horizontal')

    ax.set_xlabel('Men', labelpad=10,rotation='horizontal')

 

#    ax.yaxis.set_label_coords(-5,0)

#    ax.set_zlabel(r zlabel, rotation=0)      

#    plt.xlabel(rotation='horizontal')

#    ax.set_zlabel(r'\textbf{zlabel}')

    ax.dist = 10

    plt.title(title)

    plt.show()

    return fig
    
fig = wireframeplot2(alphas, p['typespace'],250,30,r'$\alpha(x,y)$')

fig.savefig("alpha.png")


#Plot of Values_s_f
x = np.linspace(0,50)
y = values_s_f

plt.plot(x,y)
plt.xlabel("Female type")
plt.ylabel("values_s_f")
plt.grid(True)
plt.show()

#Plot of Values_s_m
x = np.linspace(0,50)
y = np.transpose(values_s_m)

plt.plot(x,y)
plt.xlabel("Male type")
plt.ylabel("values_s_m")
plt.grid(True)
plt.show()

#Plot of u_m_1
x = np.linspace(0,50)
y = np.transpose(u_m_1)

plt.plot(x,y)
plt.xlabel("Male type")
plt.ylabel("u_m_1")
plt.grid(True)
plt.show()

#Plot of u_m_1
x = np.linspace(0,50)
y = u_f_1

plt.plot(x,y)
plt.xlabel("Female type")
plt.ylabel("u_f_1")
plt.grid(True)
plt.show()

#Plot of n_xy
def wireframeplot2(z, space, azim, elev, title):

    """

    Produces a simple wireframe plot.

    """

    X, Y = np.meshgrid(space, space)

    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(projection='3d')

    ax.plot_wireframe(X, Y, z,

                          rstride=2,

                          cstride=2,

                          color='DarkSlateBlue',

                          linewidth=1,

                          antialiased=True)

    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.set_rotate_label(False)  # disable automatic rotation

    ax.yaxis.set_rotate_label(False)  # disable automatic rotation

#    ax.zaxis.set_rotate_label(False)  # disable automatic rotation

 

#    ax.set_ylabel(r'\textbf{Women}', labelpad=20,rotation='horizontal')

#    ax.set_xlabel(r'\textbf{Men}', labelpad=10,rotation='horizontal')

    ax.set_ylabel('Women', labelpad=20,rotation='horizontal')

    ax.set_xlabel('Men', labelpad=10,rotation='horizontal')

 

#    ax.yaxis.set_label_coords(-5,0)

#    ax.set_zlabel(r zlabel, rotation=0)      

#    plt.xlabel(rotation='horizontal')

#    ax.set_zlabel(r'\textbf{zlabel}')

    ax.dist = 10

    plt.title(title)

    plt.show()

    return fig
    
fig = wireframeplot2(n_xy, p['typespace'],250,30,r'$n(x,y)$')

fig.savefig("n_xy.png")