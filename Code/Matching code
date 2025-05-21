import numpy as np          
import pandas as pd                 
from scipy.optimize import minimize 
import seaborn as sns              
import matplotlib.pyplot as plt     
# Global parameter settings
SPEED = 30.0                       
MAX_PICKUP_TIME = 0.2          
RHO_INIT = 1.0               
RHO_MULTIPLIER = 2               
MAX_ITER = 1000                  

# Load data from Excel files
df_withloc = pd.read_excel("Drivers and Riders' Data.xlsx")
df_dist = pd.read_excel("Distance Data.xlsx")

# Define labels for drivers and passengers (i1-i15, j1-j15)
drivers = [f'i{i}' for i in range(1, 16)]
passengers = [f'j{j}' for j in range(1, 16)]
n = len(drivers)                  

# Extract model parameters from data
m = df_withloc['m'].iloc[0]        # Base fare (fixed fee)
alpha = df_withloc['\\alpha'].iloc[0]  # Per-kilometer fare
beta = [0]*n                       # Driver cost per kilometer (initialized)
gamma = [0]*n                      # Passenger quality preference coefficient
d_o_w = [0]*n                      # Trip distance from origin to destination

# Populate parameters from DataFrame
for _, row in df_withloc.iterrows():
    driver_id = int(row['Driver ID'][1:]) - 1 
    rider_id = int(row['Rider ID'][1:]) - 1    
    beta[driver_id] = row['\\beta']         
    gamma[rider_id] = row['\\gamma']         
    d_o_w[rider_id] = row['Trip_Distance']    

# Convert lists to numpy arrays for vectorized operations
beta = np.array(beta)
gamma = np.array(gamma)
d_i_o = df_dist.set_index('Distance').T.values  

# Initialize matrices for welfare, cost violations, and time violations
W = np.zeros((n,n))               # Social welfare matrix (driver-passenger utility)
cost_violation = np.zeros((n,n))  # Cost feasibility violations
time_violation = np.zeros((n,n))  # Pickup time violations

# Calculate matrix values using nested loops
for i in range(n):
    for j in range(n):
        revenue = m + alpha * d_o_w[j]          # Total fare for driver i + passenger j
        cost = beta[i] * (d_i_o[i,j] + d_o_w[j])  # Total cost (pickup + trip)
        wait_time = d_i_o[i,j] / SPEED          # Pickup time in hours
        Ud = revenue - cost                     # Driver's net utility
        Ur = gamma[j] * beta[i] - wait_time     # Passenger's net utility
        W[i,j] = Ud + Ur                        # Combined social welfare
        cost_violation[i,j] = max(0, cost - revenue)  # Penalty for unprofitable trips
        time_violation[i,j] = max(0, wait_time - MAX_PICKUP_TIME)  # Penalty for late pickups

# Objective function for optimization
def objective_flat(x, W_sub, C_sub, T_sub, nd, npas, rho):
    Z = x.reshape((nd, npas))                   # Reshape to submatrix
    row_sum = Z.sum(axis=1)                     # Sum of each driver's matches
    col_sum = Z.sum(axis=0)                     # Sum of each passenger's matches
    
    # Penalty terms for matching constraints
    match_pen = np.sum((1 - row_sum)**2) + np.sum((1 - col_sum)**2)  # Ensure 1-1 matching
    cost_pen = np.sum(Z * (C_sub**2))            # Cost violation penalty
    time_pen = np.sum(Z * (T_sub**2))            # Time violation penalty
    
    return -np.sum(W_sub * Z) + rho * (match_pen + cost_pen + time_pen)
    # Negative sign converts maximization to minimization problem

# Gradient of the objective function (for BFGS algorithm)
def grad_flat(x, W_sub, C_sub, T_sub, nd, npas, rho):
    Z = x.reshape((nd, npas))
    row_sum = Z.sum(axis=1)
    col_sum = Z.sum(axis=0)
    
    # Gradient for matching constraints
    match_grad = -2 * ((1 - row_sum)[:, np.newaxis] + (1 - col_sum)[np.newaxis, :])
    grad = -W_sub + rho * (match_grad + C_sub**2 + T_sub**2)  
    return grad.flatten()                                   

# Initialize final matching matrix and tracking sets
Z_final = np.zeros((n,n), int)            
matched_drivers = set()                  
matched_passengers = set()               
rho = RHO_INIT                            

# Store intermediate results for first 3 iterations 
s_list, y_list, B_list, rho_list, Z_list = [], [], [], [], []

# Function to plot matching matrix heatmap
def plot_matrix(matrix, title, path, drivers, passengers):
    plt.figure(figsize=(6, 6))
    # Create heatmap with binary values (0/1) and driver/passenger labels
    sns.heatmap(matrix, annot=True, fmt="d", cbar=False,
                yticklabels=drivers, xticklabels=passengers,
                cmap="Blues")
    plt.xlabel("Passenger")
    plt.ylabel("Driver")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)  # Save plot to file
    plt.close()        # Close figure to free memory

# Main optimization loop
for iteration in range(1, MAX_ITER + 1):
    # Get indices of unmatched drivers and passengers
    rem_i = [i for i in range(n) if i not in matched_drivers]
    rem_j = [j for j in range(n) if j not in matched_passengers]
    nd, npas = len(rem_i), len(rem_j)
    
    if nd == 0 or npas == 0:
        break  # Exit if no more matches possible
    
    # Submatrix extraction for active drivers/passengers
    W_sub = W[np.ix_(rem_i, rem_j)]
    C_sub = cost_violation[np.ix_(rem_i, rem_j)]
    T_sub = time_violation[np.ix_(rem_i, rem_j)]
    
    # Initial guess for matching: identity matrix or uniform distribution
    Z0 = np.ones((nd, npas))*0.5 if nd==2 and npas==2 else np.eye(nd, npas)
    thresh = 0.4 if nd==2 and npas==2 else 0.5  # Threshold for binary conversion
    x_k = Z0.flatten()                           # Flatten for optimization
    
    # Minimize the objective function using BFGS algorithm
    res = minimize(lambda x: objective_flat(x, W_sub, C_sub, T_sub, nd, npas, rho),
                   x_k,
                   jac=lambda x: grad_flat(x, W_sub, C_sub, T_sub, nd, npas, rho),
                   method='BFGS',
                   options={'disp': False})  # Suppress optimization output
    
    x_k1 = res.x
    grad_k = grad_flat(x_k, W_sub, C_sub, T_sub, nd, npas, rho)
    grad_k1 = grad_flat(x_k1, W_sub, C_sub, T_sub, nd, npas, rho)
    s = x_k1 - x_k          # Step vector
    y = grad_k1 - grad_k    # Gradient difference
    
    # Store intermediate results for first 3 iterations
    if iteration <= 3:
        s_list.append(s)
        y_list.append(y)
        B_list.append(res.hess_inv)  # Approximated inverse Hessian matrix
        rho_list.append(rho)
        Z_list.append(res.x.reshape((nd, npas)))
        
        # Print debug information for clarity
        print(f"\n--- Iteration {iteration} ---")
        print(f"Penalty coefficient (rho): {rho:.4f}")
        print(f"Step vector (s):\n{np.array2string(s.reshape((nd, npas)), formatter={'float_kind':lambda x: '%.3g' % x})}")
        print(f"Gradient difference (y):\n{np.array2string(y.reshape((nd, npas)), formatter={'float_kind':lambda x: '%.3g' % x})}")
        print(f"Inverse Hessian (B):\n{np.array2string(res.hess_inv, formatter={'float_kind':lambda x: '%.3g' % x})}")
        
        # Convert continuous solution to binary and plot
        Z_bin = (res.x.reshape((nd, npas)) > thresh).astype(int)
        current_drivers = [drivers[i] for i in rem_i]
        current_passengers = [passengers[j] for j in rem_j]
        plot_matrix(Z_bin, f"Iteration {iteration}: Binary Matching", 
                    f"iteration_{iteration}_Z_bin.png", current_drivers, current_passengers)
    
    # Convert continuous solution to binary and select valid matches
    Z_bin = (res.x.reshape((nd, npas)) > thresh).astype(int)
    candidates = []
    for a in range(nd):
        for b in range(npas):
            if Z_bin[a, b]:
                # Map submatrix indices to global indices
                i_glob, j_glob = rem_i[a], rem_j[b]
                candidates.append((i_glob, j_glob, W[i_glob, j_glob], d_i_o[i_glob, j_glob]))
    
    # Sort candidates by welfare (descending) and pickup distance (ascending)
    candidates.sort(key=lambda x: (-x[2], x[3]))
    
    # Assign matches while respecting uniqueness constraints
    for (i_glob, j_glob, w, d) in candidates:
        if i_glob not in matched_drivers and j_glob not in matched_passengers:
            Z_final[i_glob, j_glob] = 1
            matched_drivers.add(i_glob)
            matched_passengers.add(j_glob)
    
    # Adjust penalty coefficient and check for convergence
    if iteration % 2 == 0:
        rho *= RHO_MULTIPLIER  # Increase penalty strength every 2 iterations
        if np.linalg.norm(grad_k1) < 1e-6:
            break  # Early exit if gradients are sufficiently small

# Initial matching: identity matrix (driver i matches passenger i)
Z0 = np.eye(n)  
plot_matrix(Z0, "Initial Matching Matrix (Z0)", "initial_matching_matrix.png", drivers, passengers)

# Generate final matching plot
plot_matrix(Z_final, "Final Matching Matrix (Z_final)", "final_matching_matrix.png", drivers, passengers)
print("Final matching matrix plot saved as: final_matching_matrix.png")

# Calculate social welfare metrics
initial_welfare = np.sum(W * Z0)       # Welfare under initial matching
final_welfare = np.sum(W * Z_final)     # Welfare under optimized matching

# Output results
print(f"\nInitial social welfare (default matching): {initial_welfare:.2f}")
print(f"Optimized social welfare (final matching): {final_welfare:.2f}")
print(f"Welfare improvement: {final_welfare - initial_welfare:.2f}")
