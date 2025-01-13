#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:50:02 2024

@author: jarrah
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from torch.distributions import MultivariateNormal
import matplotlib
# from toy_data import inf_train_gen
# import seaborn as sns
# import sklearn.datasets
# import sys
plt.close('all')

plt.rc('font', size=14)          # controls default text sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

torch.manual_seed(0)


def generate_phit(t, n):
    """
    t shape is (T, N, 1), output shape is (T, N, n, n)
    """
    T, N, _ = t.shape
    phi_t = torch.zeros((T, N, n, n))
    phi_t[:,:,0,0] = t[:,:,0]**3/3
    phi_t[:,:,0,1] = t[:,:,0]**2/2
    phi_t[:,:,1,0] = t[:,:,0]**2/2
    phi_t[:,:,1,1] = t[:,:,0]
    return phi_t

def generate_expAt(A, t):
    """
    A shape is (n,n), t shape is (T,N,1), output shape is (T,N,n,n)
    """
    T, N, _ = t.shape
    A_rep = A.repeat(T,N,1,1)
    A_t = A_rep * t.unsqueeze(-1)
    A_t_exp = torch.matrix_exp(A_t)

    return A_t_exp


#%%
N = 1 # number of samples
T = 100*1 # number of time steps
n = 2 # system dimension
m = 1 # control dimension

tf = 1.0 # time horizon
# dt = tf/T # time step

A = torch.tensor([[0.0, 1.0],[0.0, 0.0]])
B = torch.tensor([[0.0],[1.0]])

#x_0 = MultivariateNormal(torch.zeros(n), torch.eye(n)).sample((N,))
x_0 = torch.zeros([N,n])

mu_target = torch.tensor([3.0,3.0])*2

y = torch.zeros((N,n))
# y[:int(N/2)] = MultivariateNormal(mu_target, torch.eye(n)).sample((int(N/2),))
# y[int(N/2):] = MultivariateNormal(-mu_target, torch.eye(n)).sample((N//2,))
y = torch.ones([N,n])*3
# y = torch.ones([N,n])

plt.figure(figsize=(8,8))
plt.scatter(x_0.numpy()[:,0],x_0.numpy()[:,1])
plt.scatter(y.numpy()[:,0],y.numpy()[:,1])

X_f = torch.zeros((T, N, n))
W_f = torch.zeros((T, N, m))


X_f[0] = x_0

t_N = torch.linspace(0.0, tf, T).repeat(N,1).reshape(N,T,1).permute(1,0,2) # shape (T,N,1)
dt = t_N[1,0,0] - t_N[0,0,0] 

expAt = generate_expAt(A, t_N)
exp1tAtrans = generate_expAt(A.T, 1-t_N)
expA = generate_expAt(A, torch.ones_like(t_N))
exp1tA = generate_expAt(A, 1-t_N)
phi_t = generate_phit(t_N, n)
phi_1 = generate_phit(torch.ones_like(t_N), n)
phi_1_t = generate_phit(1-t_N, n)

mat_for_x = expAt - torch.einsum('tnij,tnjk->tnik', torch.einsum('tnij,tnjk->tnik', torch.einsum('tnij,tnjk->tnik', phi_t, exp1tAtrans), torch.linalg.inv(phi_1)), expA)

mat_for_y = torch.einsum('tnij,tnjk->tnik', torch.einsum('tnij,tnjk->tnik', phi_t, exp1tAtrans), torch.linalg.pinv(phi_1))

phi_t_exp1tAtrans = torch.einsum('tnij,tnjk->tnik', phi_t, exp1tAtrans)
Sigma_t = phi_t - torch.einsum('tnij,tnjk->tnik', torch.einsum('tnij,tnjk->tnik', phi_t_exp1tAtrans, torch.linalg.pinv(phi_1)), phi_t_exp1tAtrans.transpose(-1,-2))

Z_f = torch.zeros((T, N, n))

for i in range(T):
    try:
        Z_f[i,:,:] = MultivariateNormal(torch.zeros(n), Sigma_t[i,0,:,:]).sample((N,))
    except:
        print(i)



plt.figure()

for n_sigma in[0,0.5,1.0]:
# for n_sigma in[0]:
# n_sigma = 1 # noise level

    # X_f = torch.einsum('tnij,tnj->tni', mat_for_x, x_0.unsqueeze(0).repeat(T,1,1)) + torch.einsum('tnij,tnj->tni', mat_for_y, y.unsqueeze(0).repeat(T,1,1)) + n_sigma * Z_f
    # u_f = torch.einsum('tnij,tnj->tni',torch.einsum('tnij,tnjk->tnik', torch.einsum('ij,tnjk->tnik', B.T, exp1tAtrans), torch.linalg.pinv(phi_1_t)), y[None,:,:] - torch.einsum('tnij, tnj->tni', exp1tA, X_f))
    
    X_n = torch.zeros_like(X_f)
    X_n[0] = x_0
    for i in range(1,T):
        if i%10==0:
            print(i)
        u_t = torch.einsum('nij,nj->ni',torch.einsum('nij,njk->nik', 
            torch.einsum('ij,njk->nik', B.T, exp1tAtrans[i-1]), torch.linalg.pinv(phi_1_t)[i-1]),
                           y - torch.einsum('nij, nj->ni', exp1tA[i-1], X_n[i-1]))
        X_n[i] = X_n[i-1] + X_n[i-1] @ A.T * dt +  torch.einsum('ij,nj->ni', B, u_t*dt + n_sigma*torch.sqrt(dt)*torch.randn((N,1)))

    plt.plot(X_n[:,0,0],X_n[:,0,1],label=r'$\epsilon = %.1f$'%(n_sigma),lw=2)
    
    # plt.subplot(1,2,1)
    # plt.plot(t_N[:,0,0],X_n[:,:,0].detach().numpy())
    
    # plt.subplot(1,2,2)
    # plt.plot(t_N[:,0,0],X_n[:,:,1].detach().numpy())
    
plt.xlabel(r'$X_t(1)$')
plt.ylabel(r'$X_t(2)$')
plt.legend()   
plt.show()

#%%
# plt.figure(figsize=(10,10))
# for i in range(N):
#     plt.plot(X_n[:,i,0],X_n[:,i,1],label=n_sigma)

# plt.scatter(X_n[0,:,0],X_n[0,:,1])
# plt.scatter(X_n[-1,:,0],X_n[-1,:,1])


