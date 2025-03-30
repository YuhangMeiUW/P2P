import torch
from scipy.optimize import minimize
import numpy as np

def generate_phit(t, n):
    """
    t shape is (T, 1), output shape is (T, n, n)
    """
    T, _ = t.shape
    phi_t = torch.zeros((T, n, n))
    phi_t[:,0,0] = t[:,0]
    phi_t[:,0,1] = torch.zeros_like(t[:,0])
    phi_t[:,1,0] = torch.zeros_like(t[:,0])
    phi_t[:,1,1] = t[:,0]
    return phi_t

def generate_expAt(A, t):
    """
    A shape is (n,n), t shape is (T,1), output shape is (T,n,n)
    """
    T, _ = t.shape
    A_rep = A.repeat(T,1,1)
    A_t = A_rep * t.unsqueeze(-1)
    A_t_exp = torch.matrix_exp(A_t)

    return A_t_exp

def jacobian(y: torch.Tensor, x: torch.Tensor, device='cpu', need_higher_grad=True) -> torch.Tensor:
    (Jac,) = torch.autograd.grad(
        outputs=(y.flatten(),),
        inputs=(x,),
        grad_outputs=(torch.eye(torch.numel(y)).to(device),),
        create_graph=need_higher_grad,
        allow_unused=True,
        is_grads_batched=True
    )
    if Jac is None:
        Jac = torch.zeros(size=(y.shape + x.shape))
    else:
        Jac.reshape(shape=(y.shape + x.shape))
    return Jac

def batched_jacobian(batched_y:torch.Tensor,batched_x:torch.Tensor,device='cpu', need_higher_grad = True) -> torch.Tensor:
    sumed_y = batched_y.sum(dim = 0) # y_shape
    J = jacobian(sumed_y,batched_x,device, need_higher_grad) # y_shape x N x x_shape
    
    dims = list(range(J.dim()))
    dims[0],dims[sumed_y.dim()] = dims[sumed_y.dim()],dims[0]
    J = J.permute(dims = dims) # N x y_shape x x_shape
    return J

def train_score_nn(X_backward, time_grid, B, learning_rate, iterations, batch_size, t_batch_size, sample_size, model):
    T = time_grid.shape[0]
    n = X_backward.shape[2]
    t_N = time_grid.repeat(1,sample_size).reshape(T, sample_size, 1)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
    for i in range(iterations):
        idx = torch.randperm(sample_size)[:batch_size]
        t_idx = torch.randperm(T)[:t_batch_size]
        X_train = X_backward[:,idx,:]
        X_train = X_train[t_idx,:,:]
        X_train = X_train.view(-1, n)
        t_train = t_N[:,idx,:]
        t_train = t_train[t_idx,:,:]
        t_train = t_train.view(-1, 1)
        X_train.requires_grad = True
        k_value = model(X_train, t_train)
        gk = k_value @ B.T
        batch_norm = torch.einsum('tij,tjk->tik', gk.unsqueeze(1), gk.unsqueeze(2)).squeeze(-1)
        batch_jac = batched_jacobian(gk, X_train)
        temp = torch.einsum('ij,tjk->tik', B@B.T, batch_jac)
        batch_trace = temp.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1, keepdim=True)
        loss = (0.5*batch_norm + batch_trace).sum()/batch_size/t_batch_size
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if  (i+1)==iterations or (i+1)%1000==0:
            k_value = model.forward(X_train, t_train) 
            gk = k_value @ B.T
            score_norm = torch.sum(gk**2)/batch_size/t_batch_size
            loss = score_norm
        
            BatchJac = batched_jacobian(gk, X_train)
            temp = torch.einsum('ij,tjk->tik', B@B.T, BatchJac)
            BatchTrace = temp.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1, keepdim=True) # shape (T*B, 1)
            loss = loss*0.5 + BatchTrace.sum()/batch_size/t_batch_size

        
            print("Iteration: %d/%d, loss = %.12f" %(i+1,iterations,loss.item()))






    # Terminal state constraint: x(T) ≈ xT



def trajectory_optimization(dt, T, x_0, y):
    u0 = np.zeros(T)
    y = np.array(y).reshape(2)
    x_0 = np.array(x_0).reshape(2)
    def simulate_trajectory(u_flat, dt, T, x_0):
        u_seq = u_flat.reshape(T, 1)
        x_seq = np.zeros((T+1, 2))
        x_seq[0] = np.array(x_0).reshape(2)
        for k in range(T):
            x = x_seq[k]
            u = u_seq[k, 0]
            fx = np.array([x[1], np.sin(x[0]) - 0.01 * x[1]])
            dx = fx + np.array([0.0, 1.0]) * u
            x_seq[k+1] = x + dt * dx
        return x_seq
    
    def terminal_constraint(u_flat):
        x_seq = simulate_trajectory(u_flat, dt, T, x_0)
        return x_seq[-1] - y
    
    def cost(u_flat):
        return np.sum(u_flat**2) * dt
    
    res = minimize(
        cost,
        u0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': terminal_constraint},
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
        )
    u_traj = res.x
    U_d = torch.tensor(u_traj).reshape(T, 1).to(torch.float32)
    return U_d