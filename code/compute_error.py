import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch

def solve(model, ode, method='RK45', device = 'cuda'):
    # Extract ODE function and initial conditions from the dictionary
    ODE_func = ode['func']
    ODE_y0 = ode['y0'].cpu().numpy()  # Convert to numpy array for solve_ivp
    ODE_name = ode['name']
    ODE_dim = ode['dim']
    T = ode['T']
    
    t = torch.linspace(0, T, 100).view(-1, 1).to(device)
    
    y_pred_np = model(t).cpu().detach().numpy()
    
    t_np = t.cpu().detach().numpy()
    
    def ODE_func_np(t, y):
        dydt = ODE_func(t, y)
        return dydt.cpu().numpy()
    
    # Solve the ODE system using solve_ivp
    sol = solve_ivp(ODE_func_np, (0, T), ODE_y0, method=method, t_eval=np.linspace(0, T, 100))
    
    return sol, y_pred_np, t_np

def plot_results(sol, y_pred_np, t_np, ode):
    num_y = y_pred_np.shape[1]
    for i in range(num_y):
        plt.plot(sol.t, sol.y.T[:, i], color='blue', label=f'Analytical Solution y_{i+1}')
        plt.plot(t_np, y_pred_np[:, i], color='orange', label=f'PINN Prediction y_{i+1}')

    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.title(ode['name'])
    plt.show()

def compute_error_norms(y_true, y_pred, t):
    """
    Compute L1 and L2 norms of the difference between true and predicted values,
    including specific norms for the end point.
    
    Parameters:
    y_true (np.array): True values from the ODE solver
    y_pred (np.array): Predicted values from the PINN
    t (np.array): Time points
    
    Returns:
    dict: A dictionary containing L1 and L2 norms for each component,
          their averages, and end point norms
    """
    diff = np.abs(y_true - y_pred)
    
    num_components = y_true.shape[1]
    
    l1_norms = np.mean(diff, axis=0)
    l2_norms = np.sqrt(np.mean(diff**2, axis=0))
    
    # End point norms
    end_point_diff = np.abs(y_true[-1] - y_pred[-1])
    end_point_l1 = np.mean(end_point_diff)
    end_point_l2 = np.sqrt(np.mean(end_point_diff**2))
    
    results = {
        'L1': {f'y_{i+1}': l1_norms[i] for i in range(num_components)},
        'L2': {f'y_{i+1}': l2_norms[i] for i in range(num_components)},
        'L1_avg': np.mean(l1_norms),
        'L2_avg': np.mean(l2_norms),
        'End_point_L1': {f'y_{i+1}': end_point_diff[i] for i in range(num_components)},
        'End_point_L2': {f'y_{i+1}': np.sqrt(end_point_diff[i]**2) for i in range(num_components)},
        'End_point_L1_avg': end_point_l1,
        'End_point_L2_avg': end_point_l2,
        'End_point_time': t[-1]
    }
    
    return results

def compare_error_norms(*error_norms_list, names=None):
    """
    Compare multiple error_norms dictionaries.
    
    Parameters:
    *error_norms_list: Variable number of error_norms dictionaries
    names: List of names for each error_norms dictionary (optional)
    
    Returns:
    pandas.DataFrame: A dataframe comparing all the error norms
    """
    if names is None:
        names = [f"Model_{i+1}" for i in range(len(error_norms_list))]
    
    if len(names) != len(error_norms_list):
        raise ValueError("Number of names should match number of error_norms dictionaries")
    
    comparison = {}
    
    for name, error_norms in zip(names, error_norms_list):
        model_metrics = {
            f"{name}_L1_avg": error_norms['L1_avg'],
            f"{name}_L2_avg": error_norms['L2_avg'],
            f"{name}_End_L1_avg": error_norms['End_point_L1_avg'],
            f"{name}_End_L2_avg": error_norms['End_point_L2_avg'],
        }
        
        for component in error_norms['L1']:
            model_metrics.update({
                f"{name}_L1_{component}": error_norms['L1'][component],
                f"{name}_L2_{component}": error_norms['L2'][component],
                f"{name}_End_L1_{component}": error_norms['End_point_L1'][component],
                f"{name}_End_L2_{component}": error_norms['End_point_L2'][component],
            })
        
        comparison.update(model_metrics)
    
    df = pd.DataFrame(comparison, index=['Value'])
    df = df.T  # Transpose for better readability
    
    # Sort the dataframe
    df = df.sort_index()
    
    return df