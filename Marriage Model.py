#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:00:48 2025

@author: louisebertelsen
"""


import pandas as pd # to import eg. excel
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#%% Distributions

n_types = 500

income_grid = np.linspace(1/n_types/2,1-1/n_types/2,n_types)

# Creating distributions for male and female types as alternative to data from class
male_density = np.exp(-0.5 * (income_grid - 5) ** 2)
female_density = np.exp(-0.5 * (income_grid - 4.5) ** 2)

# Random normal distribution with mean zero for love shocks
z_distribution = np.random.normal(0, 1, size=(n_types, n_types))


#%%

# Checking whether we integrate to 1
def integrate_uni(values, xstep):
    "integrates the grid with the specified values with stepsize xstep"
    #spacing = x1/values.size
    copy = np.copy(values)
    copy.shape = (1, values.size)
    return integrate.simpson(copy, dx=xstep)

def integrate_red(matrix, result, xstep): #integrating in 2 dimensions
    n = matrix.shape[0]
    if result == 'male':
        inner = np.zeros((1,n))
        for i in range(0,n):
            inner[0,i] = np.squeeze(integrate_uni(matrix[:,i],xstep))
    elif result == 'female':
        inner = np.zeros((n,1))
        for i in range(0,n):
            inner[i,0] = np.squeeze(integrate_uni(matrix[i,:],xstep))
    return inner

# production function (simple)
def production_function(x,y):
    return x*y

# Equation (16) and (17) from Jaquemet & Robin 
def flow_update(dens_e, dens_u_o, alphas, c_delta, result, c_lambda_m, xstep): 
    int_u_o = integrate_red(dens_u_o * alphas, result, xstep) 
    int_u_o.shape = dens_e.shape
    return c_delta*dens_e / (c_delta + c_lambda_m * int_u_o)

def inner_integrand(z_distribution, values_s_m, values_s_f, c_xy):
    result = c_xy + z_distribution - values_s_m - values_s_f
    return np.maximum(result, 0)

def inner_integral(z_distribution, values_s_m, values_s_f, c_xy, result):
    return integrate_red(inner_integrand(z_distribution, values_s_m, values_s_f, c_xy), result, z_distribution)

# Equation (14) and (15) from Jaquemet & Robin 
def flow_surplus(z_distribution, c_lambda_m, c_r, c_delta, dens_c, dens_u_o, result, xstep):
    int_u_o_1 = integrate_red(inner_integral(z_distribution, values_s_m, values_s_f, c_xy, result)*dens_u_o, result, xstep)
    int_u_o_1.shape = dens_c.shape
    return dens_c + c_lambda_m/(c_r + c_delta)*int_u_o_1
    

#%%

# Define dictionary
p = dict()

# model parameters
#nash-bargening power
p['c_beta'] = 0.5
# discount rate
p['c_r']= 0.05
#seperation
p['c_delta']=0.1
p['c_lampda_m']=1

print(np.shape(male_density))
p['xmin']= male_density[0] 
print('Lowest grid point:', p['xmin'])
p['xmax']= male_density[49]
print('Highest grid point:', p['xmax'])
p['xstep']= male_density[1] - male_density[0]
print('stepsize:',p['xstep'])

# type space
p['typespace_n'] = male_density
p['typespace'] = p['typespace_n']/np.min(p['typespace_n'])


p['n_types']=n_types
p['male_dens'] = male_density # kolon angiver at vi tager en hel kolonne
p['female_dens'] = female_density
p['z_dis'] = z_distribution


#normalize densities
#density function for all agents 
e_m = p['male_dens'] / integrate_uni(p['male_dens'],p['xstep'])
e_m.shape = (1, p['n_types'])
e_f = p['female_dens'] / integrate_uni(p['female_dens'],p['xstep'])
e_f.shape = (p['n_types'],1)

u_m = np.ones((1,p['n_types']))
u_f = np.ones((p['n_types'],1))

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
c_x = np.zeros((p['n_types'] ,p['n_types']))
c_y = np.zeros((p['n_types'], p['n_types']))
        
        
for xi in range(p['n_types']):
    c_x[0,xi]=xgrid[xi]
    
for yi in range(p['n_types']):
    c_y[yi,0]=ygrid[yi]


values_s_m = c_x
values_s_f = c_y

# Compute the matching matrix
# alphas = (c_xy + z_density >= values_s_m + values_s_f).astype(float)

alphas = np.zeros([n_types, n_types])
for i in range(n_types):
    for j in range(n_types):
        if c_xy[i,j] + z_distribution[i,j] >= values_s_m[0,i] + values_s_f[j,0]:
            alphas[i,j] = 1

# perform the fixed point iteration to calculate the equilibrium 
maxiter = 1000 # max 1000 iterrations 

# Equation (16) and (17)
for iter in range(maxiter):
    
    int_U_m = integrate_uni(u_m, p['xstep'])
    int_U_f = integrate_uni(u_f,p['xstep'])
    
    u_m_1 = flow_update(e_m, u_f, alphas, p['c_delta'], 'male', p['c_lampda_m'], p['xstep'])
    u_f_1 = flow_update(e_f, u_f, alphas, p['c_delta'], 'female',p['c_lampda_m'], p['xstep'])

# Equation (14) and (15)
for iter in range(maxiter):
    
    s_m_1 = flow_surplus(p['z_dis'], p['c_lampda_m'], p['c_r'], p['c_delta'], c_x, u_m_1, 'male', p['xstep'])
    s_f_1 = flow_surplus(p['z_dis'], p['c_lampda_m'], p['c_r'], p['c_delta'], c_y, u_f_1, 'female', p['xstep'])
    
    
    
    
    
    
    
    
    
    
    

