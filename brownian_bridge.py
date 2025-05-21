import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from score_neural_network import score_nn
import time
from utils import generate_phit, generate_expAt, jacobian, batched_jacobian, train_score_nn


plt.rc('font', size = 14)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

### Experiment parameters
# N = 2000 ##Number of samples
# N_list = [20, 50, 100, 200, 400, 800, 2000] ##Number of samples
N_list = [1000] ##Number of samples
# T_list = [501, 251, 101, 51, 21, 11]## number of time steps dt = [0.002, 0.004, 0.01, 0.02, 0.05, 0.1]
T_list = [251]
n = 2 ## dimension of the state space
m = 2 ## dimension of the control space
tf = 1.0 ## terminal time

epsilon = 0.3 ## noise level

A = torch.tensor([[0.0,0.0],[0.0,0.0]])
B = torch.eye(2)

x_0 = torch.tensor([[0.0],[0.0]])
y = torch.tensor([[2.0],[2.0]])

# sigma_list = [0.0, 0.02, 0.04, 0.05, 0.1, 0.2, 0.3] ##std of the terminal distribution
# sigma_list = [0.0, 0.02] ##std of the terminal distribution
sigma_list = [0.0] ##std of the terminal distribution

exp_num = 1 ## number of experiments


# MSE_record = np.zeros((len(T_list), 5, exp_num))
# u_norm_record = np.zeros((len(sigma_list),5,exp_num))
# MSE_det_record = np.zeros((len(T_list),len(sigma_list),exp_num))
dt_list = np.zeros(len(T_list))
total_exp_num = len(T_list)*exp_num*len(sigma_list)*len(N_list)


for i, T in enumerate(T_list):
    t = torch.linspace(0,tf,T).reshape(-1,1) ## time grid, shape is (T,1)
    dt = t[1] - t[0]
    dt_list[i] = dt
    exp1tAtrans = generate_expAt(A.T, tf-t)
    exp1tA = generate_expAt(A, tf-t)
    phi1t = generate_phit(tf-t, n)
    expA = generate_expAt(A, torch.ones_like(t))[0,:,:]
    phi_1 = generate_phit(torch.ones_like(t), n)
    expt1A = generate_expAt(A, t-tf)
    expt1Atrans = generate_expAt(A.T, t-tf)
    # exp_minustA = generate_expAt(-A, t)
    # exp_minustAtrans = generate_expAt(-A.T, t)
    phi_t = generate_phit(t, n)
    ## compute deterministic control
    U_d = torch.einsum('tij,tjk->tik', torch.einsum('ij,tjk->tik', B.T, exp1tAtrans), torch.linalg.pinv(phi_1)) @ (y - expA @ x_0).squeeze(-1)
    for j, sigma in enumerate(sigma_list):
        for sample_idx, N in enumerate(N_list):

            for exp in range(exp_num):
                print('starting experiment: ', exp+1, 'T: ', T, 'sigma: ', sigma, 'N: ', N)
                time_start = time.time()
                
            
                ### Generate backward samples 
                X_backward_u = torch.zeros((T, N, n))
                X_backward_u[-1,:,:] = torch.randn(N,n)*sigma + y.T
                X_backward = torch.zeros((T, N, n))
                X_backward[-1,:,:] = torch.randn(N,n)*sigma + y.T
                for k in range(T-1, 0, -1):
                    W_backward = torch.randn(N,n)*np.sqrt(dt)
                    dX_u = (A @ X_backward_u[k,:,:].T  + B @ U_d[k-1,:].repeat(N,1).T).T * dt + (B @ (epsilon * W_backward).T).T
                    X_backward_u[k-1,:,:] = X_backward_u[k,:,:] - dX_u
                    dX = (A @ X_backward[k,:,:].T).T * dt + (B @ (epsilon * W_backward).T).T
                    X_backward[k-1,:,:] = X_backward[k,:,:] - dX
                
                ## Calculate mean and covariance of the backward samples without control
                Mean_Xb = X_backward.mean(dim=1)
                Cov_Xb = torch.zeros((T, n, n))
                for k in range(T):
                    Cov_Xb[k,:,:] = torch.cov(X_backward[k,:,:].T)

                ### initialize the neural network
                hidden_dim = 32
                learning_rate = 3e-4
                batch_size = 32
                t_batch_size = T//10 + 1
                iterations = 12000 

                model_u = score_nn(n, m, hidden_dim)
                train_score_nn(X_backward_u, t, B, learning_rate, iterations, batch_size, t_batch_size, N, model_u)

                model = score_nn(n, m, hidden_dim)
                train_score_nn(X_backward, t, B, learning_rate, iterations, batch_size, t_batch_size, N, model)

                W_forward = torch.zeros((T, N, m))
                for k in range(T):
                    W_forward[k,:,:] = torch.randn(N,m)*np.sqrt(dt)
                
                X_pred_u = torch.zeros((T, N, n))## NN method with control
                X_pred_det = torch.zeros((T, N, n)) ## Deterministic 
                X_pred = torch.zeros((T, N, n)) ## NN method without control
                X_pred_sol_u = torch.zeros((T, N, n))## mean covariance approximation method
                X_pred_sol = torch.zeros((T, N, n))## exact solution
                u1_record = torch.zeros((T, N, m))
                u2_record = torch.zeros((T, N, m))
                u3_record = torch.zeros((T, N, m))
                u4_record = torch.zeros((T, N, m))
                u5_record = torch.zeros((T, N, m))

                model.eval()
                model_u.eval()
                for k in range(1, T):

                    ## NN method without control
                    model_pred = model.forward(X_pred[k-1,:,:], t[k-1].repeat(N,1))
                    u1 = model_pred * epsilon**2
                    u1_record[k-1,:,:] = u1
                    dX = (A @ X_pred[k-1,:,:].T + B @ u1.T).T * dt + (B @(epsilon * W_forward[k-1,:,:]).T).T
                    X_pred[k,:,:] = X_pred[k-1,:,:] + dX
                    
                    ## NN method with control
                    model_pred_u = model_u.forward(X_pred_u[k-1,:,:], t[k-1].repeat(N,1))
                    u2 = U_d[k-1,:].repeat(N,1) + model_pred_u * epsilon**2
                    u2_record[k-1,:,:] = u2
                    dX_u = (A @ X_pred_u[k-1,:,:].T + B @ u2.T).T * dt + (B @(epsilon * W_forward[k-1,:,:]).T).T
                    X_pred_u[k,:,:] = X_pred_u[k-1,:,:] + dX_u

                    ## Exact solution without control
                    Q_1t = sigma**2 * torch.eye(n) + epsilon**2  * expt1A[k-1,:,:] @ phi1t[k-1,:,:] @ expt1Atrans[k-1,:,:]
                    u3 = -epsilon**2 * (X_pred_sol[k-1,:,:] - (expt1A[k-1,:,:] @ y).repeat(1,N).T) @ (B.T @ torch.linalg.pinv(Q_1t)).T
                    u3_record[k-1,:,:] = u3
                    dX_sol = (A @ X_pred_sol[k-1,:,:].T + B @ u3.T).T * dt + (B @(epsilon * W_forward[k-1,:,:]).T).T
                    X_pred_sol[k,:,:] = X_pred_sol[k-1,:,:] + dX_sol
                    
                    ## Exact solution with control
                    u4 = U_d[k-1,:].repeat(N,1) - epsilon**2 *(X_pred_sol_u[k-1,:,:] - (t[k-1])* y.T)/(epsilon**2 * (1-t[k-1]) + sigma**2)
                    u4_record[k-1,:,:] = u4
                    dX_sol_u = (A @ X_pred_sol_u[k-1,:,:].T + B @ u4.T).T * dt + (B @(epsilon * W_forward[k-1,:,:]).T).T
                    X_pred_sol_u[k,:,:] = X_pred_sol_u[k-1,:,:] + dX_sol_u 
                    ## Approximation k method (without control)
                    # u4 = (-(epsilon**2) * B.T @ torch.linalg.pinv(Cov_Xb[k-1,:]) @ (X_pred_k_approx[k-1,:,:] - Mean_Xb[k-1,:].repeat(N,1)).T).T 
                    # dX_k_approx = (A @ X_pred_k_approx[k-1,:,:].T + B @ u4.T).T * dt + (B @(epsilon * W_forward[k-1,:,:]).T).T
                    # X_pred_k_approx[k,:,:] = X_pred_k_approx[k-1,:,:] + dX_k_approx

                    ## Deterministic open loop control
                    u5 = U_d[k-1,:].repeat(N,1)
                    u5_record[k-1,:,:] = u5
                    dX_det = (A @ X_pred_det[k-1,:,:].T + B @ u5.T).T * dt + (B @(epsilon * W_forward[k-1,:,:]).T).T
                    X_pred_det[k,:,:] = X_pred_det[k-1,:,:] + dX_det
                    
                # MSE_record[i,0,exp] = ((X_pred[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the NN method without control
                # MSE_record[i,1,exp] = ((X_pred_u[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the NN method with control
                # MSE_record[i,2,exp] = ((X_pred_sol[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the exact solution without control
                # MSE_record[i,3,exp] = ((X_pred_sol_u[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the exact solution with control
                # MSE_record[i,4,exp] = ((X_pred_det[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the deterministic open loop control
                # MSE_record[j,sample_idx,0,exp] = ((X_pred[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the NN method without control
                # MSE_record[j,sample_idx,1,exp] = ((X_pred_u[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the NN method with control
                # MSE_record[j,sample_idx,2,exp] = ((X_pred_sol[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the exact solution without control
                # MSE_record[j,sample_idx,3,exp] = ((X_pred_sol_u[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the exact solution with control
                # MSE_record[j,sample_idx,4,exp] = ((X_pred_det[-1,:,:] - y.T)**2).sum(dim=1).mean() ## MSE of the deterministic open loop control
                # # MSE_det_record[i,j,exp] = ((X_pred_det[-1,:,:] - y.T)**2).sum(dim=1).mean()
                # u_norm_record[j,0,exp] = ((u1_record**2).sum(dim=2).mean(dim=1) * dt).sum()/tf ## control norm of the NN method without control
                # u_norm_record[j,1,exp] = ((u2_record**2).sum(dim=2).mean(dim=1) * dt).sum()/tf ## control norm of the NN method with control
                # u_norm_record[j,2,exp] = ((u3_record**2).sum(dim=2).mean(dim=1) * dt).sum()/tf ## control norm of the exact solution without control
                # u_norm_record[j,3,exp] = ((u4_record**2).sum(dim=2).mean(dim=1) * dt).sum()/tf ## control norm of the exact solution with control
                # u_norm_record[j,4,exp] = ((u5_record**2).sum(dim=2).mean(dim=1) * dt).sum()/tf ## control norm of the deterministic open loop control

                time_end = time.time()
                done_exp_num = i*exp_num*len(sigma_list)*len(N_list) + j*exp_num*len(N_list) + sample_idx*exp_num + exp + 1
                print('time: ', time_end - time_start)
                rest_exp_num = total_exp_num - done_exp_num
                rest_time = (time_end - time_start)*rest_exp_num/60
                print('rest time: ', rest_time, 'minutes')


# torch.save(u_norm_record, f'unorm_sigma_epsilon{epsilon}_N{N}_dt{dt}.pt')
# torch.save(MSE_record, f'MSE_dt_epsilon{epsilon}_N{N}_sigma{sigma}.pt')
# torch.save(MSE_det_record, f'MSE_det_record_sigma{sigma}.pt')
# torch.save(dt_list, 'dt_list.pt')
torch.save(X_pred_u, f'NNu_sigma{sigma}_epsilon{epsilon}_N{N}_T{T}4fig1.pt')
# torch.save(X_pred_det, f'Openloop_sigma{sigma}_epsilon{epsilon}_N{N}_T{T}.pt')
torch.save(X_backward_u, f'X_backward_u_sigma{sigma}_epsilon{epsilon}_N{N}_T{T}4fig1.pt')
torch.save(u2_record, f'U_sigma{sigma}_epsilon{epsilon}_N{N}_T{T}4fig1.pt')

