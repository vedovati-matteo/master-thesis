import torch.nn as nn
import torch.optim as optim
from torch import vmap
from torch.func import jacrev
import torch

from .models import LinearNN

def pin_base_exe(net_size, lr, batch_size, ode, stopping_loss, device, max_iter=50_000, wd=None):
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
        # INTERIOR LOSS
        # NN grads
        y = model(t)
        
        jacobians = batched_jacobian_fn(t)
        dydt = jacobians.squeeze(-1)
        # ODE actual grads
        ode_grads = ODE_batch(t, y)
        interior = dydt - ode_grads
        
        # BOUNDARY LOSS
        y_0 = model(torch.tensor([0.0]).to(device))
        
        boundary = y_0 - y0
        
        lambda_i = 1.0
        lambda_b = 0.5
        
        loss = nn.MSELoss(reduction='mean')
        interior_loss = loss(interior, torch.zeros_like(interior))
        boundary_loss = loss(boundary, torch.zeros_like(boundary))
        
        return lambda_i * interior_loss + lambda_b * boundary_loss
    
    train_iter = 0
    for i in range(max_iter):
        train_iter += 1
        t = (torch.rand(batch_size) * T).to(device) 
        t.to(device)
        t.requires_grad_(True)
        
        loss = loss_fn(t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
        if loss.item() < stopping_loss:
            print(f"Stopping early at iteration {i}")
            break
        
    
    return model, train_iter
