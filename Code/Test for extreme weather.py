import numpy as np
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# Global parameter settings
SPEED = 30.0  # Speed in km/h
MAX_PICKUP_TIME = 0.2  # Maximum pickup time in hours
RHO_INIT = 1.0  # Initial penalty coefficient
RHO_MULTIPLIER = 2  # Multiplier for penalty coefficient
MAX_ITER = 100  # Maximum number of iterations

# Extreme weather coefficient (parameter for sensitivity test)
extreme_weather_factor = 1.5  # Assume a (100+50)% change in parameters under extreme weather

# Read data
df_withloc = pd.read_excel("Drivers and Riders' Data.xlsx")
df_dist = pd.read_excel("Distance data.xlsx")

# Define driver and passenger labels
drivers = [f'i{i}' for i in range(1, 16)]
passengers = [f'j{j}' for j in range(1, 16)]
n = len(drivers)

# Extract relevant parameters
m = df_withloc['m'].iloc[0] * extreme_weather_factor  # Fixed fee increase
alpha = df_withloc['\\alpha'].iloc[0] * extreme_weather_factor  # Cost per km increase
beta = [0] * n
gamma = [0] * n
d_o_w = [0] * n

for _, row in df_withloc.iterrows():
    idx_i = int(row['Driver ID'][1:]) - 1
    idx_j = int(row['Rider ID'][1:]) - 1
    beta[idx_i] = row['\\beta'] * extreme_weather_factor  # Driver's cost per km increase
    gamma[idx_j] = row['\\gamma'] * extreme_weather_factor  # Passenger's tolerance rate increase
    d_o_w[idx_j] = row['Trip_Distance']

beta = np.array(beta)
gamma = np.array(gamma)
d_i_o = df_dist.set_index('Distance').T.values

# Calculate weight matrix, cost violation matrix, and time violation matrix
W = np.zeros((n, n))
cost_violation = np.zeros((n, n))
time_violation = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        revenue = m + alpha * d_o_w[j]
        cost = beta[i] * (d_i_o[i, j] + d_o_w[j])
        wait_time = d_i_o[i, j] / SPEED
        Ud = revenue - cost
        Ur = gamma[j] * beta[i] - wait_time
        W[i, j] = Ud + Ur
        cost_violation[i, j] = max(0, cost - revenue)
        time_violation[i, j] = max(0, wait_time - MAX_PICKUP_TIME)

# Define the objective function
def objective_flat(x, W_sub, C_sub, T_sub, nd, npas, rho):
    Z = x.reshape((nd, npas))
    row_sum = Z.sum(axis=1)
    col_sum = Z.sum(axis=0)
    match_pen = np.sum((1 - row_sum) ** 2) + np.sum((1 - col_sum) ** 2)
    cost_pen = np.sum(Z * (C_sub ** 2))
    time_pen = np.sum(Z * (T_sub ** 2))
    return -np.sum(W_sub * Z) + rho * (match_pen + cost_pen + time_pen)

# Define the gradient of the objective function
def grad_flat(x, W_sub, C_sub, T_sub, nd, npas, rho):
    Z = x.reshape((nd, npas))
    row_sum = Z.sum(axis=1)
    col_sum = Z.sum(axis=0)
    match_grad = -2 * ((1 - row_sum)[:, None] + (1 - col_sum)[None, :])
    grad = -W_sub + rho * (match_grad + C_sub ** 2 + T_sub ** 2)
    return grad.flatten()

# Initialize the final matching matrix and related sets
Z_final = np.zeros((n, n), int)
matched_drivers = set()
matched_passengers = set()
rho = RHO_INIT

# Store intermediate variables for the first three iterations
s_list, y_list, B_list, rho_list, Z_list = [], [], [], [], []

# Function to plot the matching matrix heatmap
def plot_matrix(matrix, title, path, drivers, passengers):
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cbar=False,
                yticklabels=drivers, xticklabels=passengers,
                cmap="Blues")
    plt.xlabel("Passenger")
    plt.ylabel("Driver")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Main optimization loop
for iteration in range(1, MAX_ITER + 1):
    rem_i = [i for i in range(n) if i not in matched_drivers]
    # Modified to use range(n)
    rem_j = [j for j in range(n) if j not in matched_passengers]
    nd, npas = len(rem_i), len(rem_j)
    if nd == 0 or npas == 0:
        break
    W_sub = W[np.ix_(rem_i, rem_j)]
    C_sub = cost_violation[np.ix_(rem_i, rem_j)]
    T_sub = time_violation[np.ix_(rem_i, rem_j)]
    Z0 = np.ones((nd, npas)) * 0.5 if nd == 2 and npas == 2 else np.eye(nd, npas)
    thresh = 0.4 if nd == 2 and npas == 2 else 0.5
    x_k = Z0.flatten()

    res = minimize(lambda x: objective_flat(x, W_sub, C_sub, T_sub, nd, npas, rho),
                   x_k,
                   jac=lambda x: grad_flat(x, W_sub, C_sub, T_sub, nd, npas, rho),
                   method='BFGS')

    x_k1 = res.x
    grad_k = grad_flat(x_k, W_sub, C_sub, T_sub, nd, npas, rho)
    grad_k1 = grad_flat(x_k1, W_sub, C_sub, T_sub, nd, npas, rho)
    s = x_k1 - x_k
    y = grad_k1 - grad_k

    if iteration <= 3:
        s_list.append(s)
        y_list.append(y)
        B_list.append(res.hess_inv)
        rho_list.append(rho)
        Z_list.append(res.x.reshape((nd, npas)))

        print(f"\n--- Iteration {iteration} ---")
        print(f"rho_{iteration} = {rho:.4f}")
        print(f"s_{iteration} (step vector):\n{np.array2string(s.reshape((nd, npas)), formatter={'float_kind':lambda x: '%.3g' % x})}")
        print(f"y_{iteration} (grad diff):\n{np.array2string(y.reshape((nd, npas)), formatter={'float_kind':lambda x: '%.3g' % x})}")
        print(f"B_{iteration} (inverse Hessian approx):\n{np.array2string(res.hess_inv, formatter={'float_kind':lambda x: '%.3g' % x})}")

        Z_bin = (res.x.reshape((nd, npas)) > thresh).astype(int)
        current_drivers = [drivers[i] for i in rem_i]
        current_passengers = [passengers[j] for j in rem_j]
        plot_matrix(
            matrix=Z_bin,
            title=f"Iteration {iteration}: Binary Matching Matrix (Zbin)",
            path=f"iteration_{iteration}_Z_bin.png",
            drivers=current_drivers,
            passengers=current_passengers
        )

    Z_bin = (res.x.reshape((nd, npas)) > thresh).astype(int)
    candidates = []
    for a in range(nd):
        for b in range(npas):
            if Z_bin[a, b]:
                i_glob, j_glob = rem_i[a], rem_j[b]
                candidates.append((i_glob, j_glob, W[i_glob, j_glob], d_i_o[i_glob, j_glob]))
    candidates.sort(key=lambda x: (-x[2], x[3]))
    for (i_glob, j_glob, w, d) in candidates:
        if i_glob not in matched_drivers and j_glob not in matched_passengers:
            Z_final[i_glob, j_glob] = 1
            matched_drivers.add(i_glob)
            matched_passengers.add(j_glob)

    if iteration % 2 == 0:
        rho *= RHO_MULTIPLIER
        if np.linalg.norm(grad_k1) < 1e-6:
            break

# Initial default matching Z0
Z0 = np.eye(n)

# Plot the final matching result
plot_matrix(
    matrix=Z_final,
    title="Final Matching Matrix (Zfinal, Extreme factor 3)",
    path="final_matching_matrix.png",
    drivers=drivers,
    passengers=passengers
)

# Social welfare before and after optimization
initial_welfare = np.sum(W * Z0)
final_welfare = np.sum(W * Z_final)
difference = final_welfare - initial_welfare
print(f"\nInitial welfare: {initial_welfare:.2f}")
print(f"Final welfare: {final_welfare:.2f}")
print(f"Difference in welfare: {difference:.2f}")
