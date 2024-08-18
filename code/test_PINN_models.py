import itertools
import csv
import pandas as pd
from tqdm import tqdm

from compute_error import solve

def test_pinn_models(ode_systems, PINNS, param_grid, compute_error_norms, device='cpu'):
    csv_filename = "results/pinn_results.csv"
    
    # Open a CSV file for writing
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Total_Iterations', 'ODE', 'ODE_dim'] + list(param_grid.keys()) + \
                     ['L1_avg', 'L2_avg', 'End_point_L1_avg', 'End_point_L2_avg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()

        # Create all combinations of parameters
        param_combinations = list(itertools.product(*param_grid.values()))

        for ode in ode_systems:
            print(f"Testing {ode['name']}")
            for pinn in PINNS:
                pinn_func = pinn['model']
                pinn_name = pinn['name']
                for params in tqdm(param_combinations, desc=f"Testing {pinn_name}"):
                    # Create a dictionary of parameters
                    param_dict = dict(zip(param_grid.keys(), params))

                    # Train the model
                    model, total_iterations = pinn_func(
                        net_size=param_dict['net_size'],
                        lr=param_dict['lr'],
                        batch_size=param_dict['batch_size'],
                        ode=ode,
                        stopping_loss=param_dict['stopping_loss'],
                        device=device,
                        max_iter = 5_000
                    )

                    # Solve
                    sol, y_pred_np, t_np = solve(model, ode)

                    # Compute error norms
                    error_norms = compute_error_norms(sol.y.T, y_pred_np, t_np)

                    # Create result dictionary
                    result = {
                        'Model': pinn_name,
                        'Total_Iterations': total_iterations,
                        'ODE': ode['name'],
                        'ODE_dim': ode['dim'],
                        **param_dict,
                        'L1_avg': error_norms['L1_avg'],
                        'L2_avg': error_norms['L2_avg'],
                        'End_point_L1_avg': error_norms['End_point_L1_avg'],
                        'End_point_L2_avg': error_norms['End_point_L2_avg'],
                    }
                    
                    # Write result to CSV file
                    writer.writerow(result)
                    print(result)

    print(f"Results saved to {csv_filename}")