import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from score_neural_network import score_nn
import time
from utils import generate_phit, generate_expAt, jacobian, batched_jacobian, train_score_nn, trajectory_optimization
from scipy.optimize import minimize

### Experiment parameters
# N_list = [20, 50, 100, 200, 400, 800, 2000] ##Number of samples
N_list = [50000] ##Number of samples
# T_list = [501, 251, 101, 51, 21, 11]## number of time steps dt = [0.002, 0.004, 0.01, 0.02, 0.05, 0.1]
# T_list = [2501, 1251, 501, 251, 101, 51]## number of time steps dt = [0.002, 0.004, 0.01, 0.02, 0.05, 0.1]
T_list = [1251]
n = 3 ## dimension of the state space
m = 1 ## dimension of the control space
tf = 5.0 ## terminal time

epsilon = 0.3 ## noise level

B = torch.tensor([[0.0],[0.0],[1.0]]) 

x_0 = torch.tensor([[torch.pi],[0.0],[0.0]])
y = torch.tensor([[0.0],[0.0],[0.0]])

# sigma_list = [0.0, 0.02, 0.04, 0.05, 0.1, 0.2, 0.3] ##std of the terminal distribution
# sigma_list = [0.0, 0.02] ##std of the terminal distribution
sigma_list = [0.3] ##std of the terminal distribution

exp_num = 1 ## number of experiments


MSE_record = np.zeros((len(T_list), 5, exp_num))
# u_norm_record = np.zeros((len(sigma_list),5,exp_num))
# MSE_det_record = np.zeros((len(T_list),len(sigma_list),exp_num))
dt_list = np.zeros(len(T_list))
total_exp_num = len(T_list)*exp_num*len(sigma_list)*len(N_list)


for i, T in enumerate(T_list):
    t = torch.linspace(0,tf,T).reshape(-1,1) ## time grid, shape is (T,1)
    dt = t[1] - t[0]
    dt_list[i] = dt
    dt = dt.item()
    # U_d = trajectory_optimization(dt, T, x_0, y)
    U_d = torch.zeros(T, m)
    for j, sigma in enumerate(sigma_list):
        for sample_idx, N in enumerate(N_list):

            for exp in range(exp_num):
                print('starting experiment: ', exp+1, 'T: ', T, 'sigma: ', sigma, 'N: ', N)
                time_start = time.time()
                
            
                ### Generate backward samples 
                X_backward = torch.zeros((T, N, n))
                X_backward[-1,:,2] = (torch.randn(N,1)*sigma).reshape(N)
                for k in range(T-1, 0, -1):
                    df1 = X_backward[k,:,1]
                    df2 = torch.sin(X_backward[k,:,0]) - 0.01 * X_backward[k,:,1] + X_backward[k,:,2]
                    df3 = torch.zeros(N)
                    df = torch.stack((df1, df2, df3), dim=1)
                    dX = (df.T).T * dt + (B @ (epsilon * torch.randn(N,m)*np.sqrt(dt)).T).T
                    X_backward[k-1,:,:] = X_backward[k,:,:] - dX
                    

                ### initialize the neural network
                hidden_dim = 32
                learning_rate = 3e-4
                batch_size = 32
                t_batch_size = 24
                iterations = 24000 


                model = score_nn(n, m, hidden_dim)
                train_score_nn(X_backward, t, B, learning_rate, iterations, batch_size, t_batch_size, N, model)
                
                N = 1500
                W_forward = torch.zeros((T, N, m))
                for k in range(T):
                    W_forward[k,:,:] = torch.randn(N,m)*np.sqrt(dt)
                
                X_pred = torch.zeros((T, N, n)) 
                X_pred[0,:int(N/2),:] = x_0.repeat(1,int(N/2)).T 
                X_pred[0,int(N/2):,:] = -x_0.repeat(1,int(N/2)).T 
                noise_term = torch.zeros((N, n))
                noise_term[:,2] = (torch.randn(N,1)*sigma).reshape(N)
                X_pred[0,:,:] = X_pred[0,:,:] + noise_term

                u1_record = torch.zeros((T, N, m))

                # model.eval()
                # model_u.eval()
                for k in range(1, T):

                    # NN method without control
                    model_pred = model.forward(X_pred[k-1,:,:], t[k-1].repeat(N,1))
                    u1 = model_pred * epsilon**2
                    u1_record[k-1,:,:] = u1
                    df1 = X_pred[k-1,:,1]
                    df2 = torch.sin(X_pred[k-1,:,0]) - 0.01 * X_pred[k-1,:,1] + X_pred[k-1,:,2]
                    df3 = torch.zeros(N)
                    df = torch.stack((df1, df2, df3), dim=1)
                    dX = (df.T  + B @ u1.T).T * dt + (B @ (epsilon * W_forward[k-1,:,:]).T).T
                    X_pred[k,:,:] = X_pred[k-1,:,:] + dX
                    

                time_end = time.time()
                done_exp_num = i*exp_num*len(sigma_list)*len(N_list) + j*exp_num*len(N_list) + sample_idx*exp_num + exp + 1
                print('time: ', time_end - time_start)
                rest_exp_num = total_exp_num - done_exp_num
                rest_time = (time_end - time_start)*rest_exp_num/60
                print('rest time: ', rest_time, 'minutes')



torch.save(X_pred, f'data/3dIP_traj_sigma{sigma}_epsilon{epsilon}_N{N}_T{T}.pt')
torch.save(X_backward, f'data/3dIP_Z_traj_sigma{sigma}_epsilon{epsilon}_N{N}_T{T}.pt')
