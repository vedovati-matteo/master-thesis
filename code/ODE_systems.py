import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ODE_1(t, y): # Exponential decay
    # y: R^1, lambda: 0.1
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    return -0.1 * y

ODE_1_y0 = torch.tensor([2.0]).to(device)

def ODE_2(t, y): # Van der Pol Oscillator
    # y: R^2, mu: 2
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    
    x = y[0]
    dy = y[1]
    # Parameters
    omega = 1.0
    zeta = 0.5
    
    dydt = torch.stack([dy, -omega**2 * x - 2 * zeta * omega * dy]).to(device)
    return dydt

ODE_2_y0 = torch.tensor([2.0, 0.0]).to(device)

def ODE_3(t, y):  # Modified Lorenz System (converging version)
    # y: R^3
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    
    x, y, z = y[0], y[1], y[2]
    # Parameters
    sigma = 3.0
    rho = 5.0
    beta = 1.0
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    return torch.stack([dx, dy, dz]).to(device)

ODE_3_y0 = torch.tensor([1.0, 1.0, 1.0]).to(device)

def ODE_4(t, y):  # Simple 3D Linear System (converging)
    # y: R^3
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    
    A = torch.tensor([[-0.1, 0.2, 0.0],
                      [0.0, -0.1, 0.2],
                      [0.0, 0.0, -0.1]]).to(device)
    
    return torch.matmul(A, y)

ODE_4_y0 = torch.tensor([1.0, 1.0, 1.0]).to(device)

def ODE_5(t, y):  # Simple 4D linear system
    # y: R^4
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    
    A = torch.tensor([[-0.1, 0.2, 0.0, 0.0],
                      [0.0, -0.1, 0.2, 0.0],
                      [0.0, 0.0, -0.1, 0.2],
                      [0.0, 0.0, 0.0, -0.1]]).to(device)
    
    return torch.matmul(A, y)

ODE_5_y0 = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)

def ODE_6(t, y):  # Damped 6D Coupled Oscillators
    # y: R^6
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    
    x1, y1, x2, y2, x3, y3 = y.unbind()
    # Parameters
    omega = [1.0, 1.5, 2.0]
    k = 0.1  # coupling strength
    damping = 0.05  # damping factor
    
    dx1 = y1
    dy1 = -omega[0]**2 * x1 + k*(x2 - x1) + k*(x3 - x1) - damping * y1
    dx2 = y2
    dy2 = -omega[1]**2 * x2 + k*(x1 - x2) + k*(x3 - x2) - damping * y2
    dx3 = y3
    dy3 = -omega[2]**2 * x3 + k*(x1 - x3) + k*(x2 - x3) - damping * y3
    
    return torch.stack([dx1, dy1, dx2, dy2, dx3, dy3]).to(device)

ODE_6_y0 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).to(device)

def ODE_7(t, y):  # 9D Damped Harmonic Oscillators
    # y: R^9
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    
    if y.size(0) != 9:
        raise ValueError(f"Expected y to have 9 elements, but got {y.size(0)} elements.")
    
    omega = torch.linspace(0.5, 1.5, 9).to(device)
    damping = 0.1  # damping factor
    
    dydt = torch.zeros_like(y)
    dydt[::2] = y[1::2]
    dydt[1::2] = -omega**2 * y[::2] - damping * y[1::2]
    
    return dydt

ODE_7_y0 = torch.tensor([1.0, 0.0] * 4 + [1.0]).to(device)

def ODE_8(t, y):  # Simplified 15D linear system
    # y: R^15
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    
    A = torch.diag(torch.linspace(-0.1, -0.5, 15)).to(device)
    
    return torch.matmul(A, y)

ODE_8_y0 = torch.rand(15).to(device)

def ODE_9(t, y):  # Simplified 25D linear system
    # y: R^25
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    
    A = torch.diag(torch.linspace(-0.1, -0.5, 25)).to(device)
    
    return torch.matmul(A, y)

ODE_9_y0 = torch.rand(25).to(device)

def solve_and_plot_ode_system(ode_system):
    # Extract ODE function and initial conditions from the dictionary
    ODE_func = ode_system['func']
    ODE_y0 = ode_system['y0'].cpu().numpy()  # Convert to numpy array for solve_ivp
    ODE_name = ode_system['name']
    ODE_dim = ode_system['dim']
    T = ode_system['T']

    def ODE_func_np(t, y):
        dydt = ODE_func(t, y)
        return dydt.cpu().numpy()
    
    # Solve the ODE system using solve_ivp
    sol = solve_ivp(ODE_func_np, (0, T), ODE_y0, method='RK45', t_eval=np.linspace(0, T, 100))

    # Plotting the results
    plt.figure(figsize=(10, 6))
    
    if ODE_dim == 1:
        plt.plot(sol.t, sol.y[0], label='y(t)')
        plt.title(f'{ODE_name}')
        plt.xlabel('Time t')
        plt.ylabel('y(t)')
    else:
        for i in range(ODE_dim):
            plt.plot(sol.t, sol.y[i], label=f'y{i+1}(t)')
        plt.title(f'{ODE_name}')
        plt.xlabel('Time t')
        plt.ylabel('State variables')
    
    plt.legend()
    plt.grid(True)
    plt.show()

ODE_systems = [
    #{'func': ODE_1, 'y0': ODE_1_y0, 'name': 'Exponential Decay', 'dim': 1, 'T': 60},
    {'func': ODE_2, 'y0': ODE_2_y0, 'name': 'Van der Pol Oscillator', 'dim': 2, 'T': 15},
    #{'func': ODE_3, 'y0': ODE_3_y0, 'name': 'Modified Lorenz System', 'dim': 3, 'T': 20},
    #{'func': ODE_5, 'y0': ODE_5_y0, 'name': 'Simple 4D Linear System', 'dim': 4, 'T': 125},
    {'func': ODE_8, 'y0': ODE_8_y0, 'name': 'Simplified 15D Linear System', 'dim': 15, 'T': 60}
]
