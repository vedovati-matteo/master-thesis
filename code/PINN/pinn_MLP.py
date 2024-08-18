import torch.nn as nn
import torch.optim as optim
from torch import vmap
from torch.func import jacrev
import torch

from .models import LinearNN

def pin_MLP_exe(net_size, lr, batch_size, ode, stopping_loss, device, max_iter=50_000, wd=None):
    num_layers, num_neurons = net_size
    func = ode['func']
    y0 = ode['y0']
    name = ode['name']
    dim = ode['dim']
    T = ode['T']
    
    model = LinearNN(1, num_layers, num_neurons, dim).to(device)
    
    # Define the Jacobian function for a single input
    def jacobian_fn(x):
        return jacrev(model)(x)
    # Use vmap to compute Jacobians for each element in the batch
    batched_jacobian_fn = vmap(jacobian_fn)
    
    ODE_batch = vmap(func, in_dims=0)
    
    if wd is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    def loss_fn(t: torch.Tensor):
        # BOUNDARY LOSS
        y_0 = model(torch.tensor([0.0], requires_grad=True).to(device))
        
        boundary = y_0 - y0
        
        # INTERIOR LOSS
        # NN grads
        y = model(t)
        jacobians = batched_jacobian_fn(t)
        dydt = jacobians.squeeze(-1)
        # ODE actual grads
        ode_grads = ODE_batch(t, y)
        interior = dydt - ode_grads
        
        # Combine losses
        loss_t = torch.cat([boundary.view(1, -1), interior])
        loss_t = (loss_t**2)
        
        # weights of the loss
        epsilon = 1
        w = torch.exp(- epsilon * (torch.cumsum(loss_t.detach(), dim=0) - loss_t.detach()))
        
        loss = torch.mean(w * loss_t)
        
        return loss, w
    
    epsilon_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    delta = 0.99
    
    train_iter = 0
    for epsilon in epsilon_list:
        for i in range(max_iter // 4):
            train_iter += 1
            t = (torch.rand(batch_size) * T).to(device)
            t, _ = torch.sort(t)
            t.requires_grad_(True)
            
            loss, w = loss_fn(t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            min_w = torch.min(w)
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}, Min w: {min_w.item()}")
            
            if min_w > delta:
                print(f"Early break at iteration {i} --------------------------------")
                break
    
    return model, train_iter
