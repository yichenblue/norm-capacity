import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import sqrtm
from scipy.integrate import quad


def solve_lambda_star(Sigma, lambd, n, initial_guess=1e-5):
    """
    Solve for λ_star such that the equation n - λ / λ_star = Tr(Σ (Σ + λ_star)^-1) holds.

    Parameters:
    Sigma (ndarray): Covariance matrix (d x d).
    lambd (float): Regularization parameter λ.
    n (int): Number of samples.

    Returns:
    float: Solution for λ_star.
    """
    d = Sigma.shape[0]  # feature dimension

    # Define degrees of freedom function
    def df_lambda_star(lambda_star):
        Sigma_inv = np.linalg.solve(Sigma + lambda_star * np.eye(d), np.eye(d))
        return np.trace(Sigma @ Sigma_inv)

    # Define equation
    def equation(lambda_star):
        return n - lambd / lambda_star - df_lambda_star(lambda_star)

    # Use fsolve to solve for λ_star
    lambda_star = fsolve(equation, initial_guess)[0]
    return lambda_star



def solve_self_consistent_equations(Sigma, n, p, lambda_reg, max_iter=500000, epsilon=1e-14):

    v2 = torch.tensor(1.0, dtype=torch.float64)
    for t in range(max_iter):
        diag = torch.diagonal(Sigma)
        trace_val = torch.sum(diag / (diag + v2))

        # Update ν1 using equation (167)
        temp = (1 - n / p)
        sqrt_term = torch.sqrt(temp**2 + 4 * lambda_reg / (p * v2))
        v1_new = (v2 / 2) * (1 - n / p + sqrt_term)

        # Update ν2 using equation (168)
        v2_new = v1_new + (v2 / p) * trace_val

        print("v2:", v2_new)

        # Convergence check
        if torch.abs(v2_new - v2) < epsilon:
            return v1_new.item(), v2_new.item()

        v2 = v2_new


def compute_variance_beta(Z, sigma, lambd):
    n, p = Z.shape
    ZZ = Z.T @ Z
    ZZ_inv = np.linalg.solve(ZZ + lambd * np.eye(p), np.eye(p))

    # Variance Norm
    variance = sigma**2 * np.trace(ZZ @ ZZ_inv @ ZZ_inv)
    return variance


def compute_bias_beta(G, Z, beta_star, lambd):
    n, p = Z.shape
    ZZ = Z.T @ Z
    ZZ_inv = np.linalg.solve(ZZ + lambd * np.eye(p), np.eye(p))

    # Bias Norm
    bias = np.linalg.norm(ZZ_inv @ Z.T @ G @ beta_star, 2)**2
    return bias


def compute_variance_risk(Z, Sigma_F, sigma, lambd):
    n, p = Z.shape
    ZZ = Z.T @ Z
    ZZ_inv = np.linalg.solve(ZZ + lambd * np.eye(p), np.eye(p))

    # Variance Risk
    variance_risk = sigma ** 2 * np.trace(Sigma_F @ ZZ @ ZZ_inv @ ZZ_inv)
    return variance_risk


def compute_bias_risk(G, F, Z, beta_star, lambd):
    n, p = Z.shape
    ZZ = Z.T @ Z
    ZZ_inv = np.linalg.solve(ZZ + lambd * np.eye(p), np.eye(p))

    # Bias Risk
    bias_risk = np.linalg.norm(beta_star - 1/math.sqrt(p) * F.T @ ZZ_inv @ Z.T @ G @ beta_star, 2) ** 2
    return bias_risk


def compute_Upsilon(Sigma, nu1, nu2, n, p):
    """Compute Upsilon using diagonal matrix optimization"""
    diag = torch.diagonal(Sigma)

    # Calculate df2_nu2 = Tr(Σ²(Σ + ν2)^-2)
    df2_nu2 = torch.sum(diag * diag / (diag + nu2)**2)

    term1 = (1 - nu1/nu2)**2
    term2 = (nu1/nu2)**2 * (df2_nu2 / (p - df2_nu2))
    Upsilon = p/n * (term1 + term2)
    return Upsilon


def compute_V_risk(Sigma, nu1, nu2, sigma, n, p, Upsilon):
    """
    sigma^2 Upsilon_rp(ν1, ν2) / (n - Upsilon_rp(ν1, ν2))
    """
    # Check denominator validity
    denominator = 1 - Upsilon

    # Calculate Variance_R
    variance_R = sigma ** 2 * Upsilon / denominator

    return variance_R

def compute_chi(Sigma, nu2, p):
    """Compute chi using diagonal matrix optimization"""
    diag = torch.diagonal(Sigma)

    # Calculate numerator Tr(Σ(Σ + ν2)^-2)
    numerator = torch.sum(diag / (diag + nu2)**2)

    # Calculate denominator p - Tr(Σ²(Σ + ν2)^-2)
    df2_nu2 = torch.sum(diag * diag / (diag + nu2)**2)

    chi = numerator / (p - df2_nu2)
    return chi


def compute_B_risk(Sigma, nu2, beta_star, n, p, Upsilon):
    """Compute B_risk using diagonal matrix optimization"""
    diag = torch.diagonal(Sigma)

    # Calculate term1 = β*^T (Σ + ν2)^-2 β*
    term1 = torch.sum(beta_star.squeeze()**2 / (diag + nu2)**2)

    # Calculate term2 = β*^T Σ(Σ + ν2)^-2 β*
    term2 = torch.sum(diag * beta_star.squeeze()**2 / (diag + nu2)**2)

    chi = compute_chi(Sigma, nu2, p)

    bias_R = nu2**2 / (1 - Upsilon) * (term1 + chi * term2)
    return bias_R.item()


def compute_V_beta(Sigma, nu1, nu2, sigma, n, p, Upsilon):
    """
    Calculate variance component V:
    σ^2 Tr(Σ(Σ + ν2)^-2) * [1 + ν2^2 Tr((Σ + ν2)^-2) / (m - df2(ν2))] / (n - Upsilon_rp(ν1, ν2))
    """

    chi = compute_chi(Sigma, nu2, p)

    V = sigma**2 * p/n * chi / (1 - Upsilon)
    return V


def compute_B_beta(Sigma, lambd, nu1, nu2, beta_star, n, p, Upsilon):
    """Compute B_beta using diagonal matrix optimization"""
    diag = torch.diagonal(Sigma)

    # Calculate terms
    term1 = torch.sum(beta_star.squeeze()**2 / (diag + nu2))
    term2 = torch.sum(beta_star.squeeze()**2 / (diag + nu2)**2)
    term3 = torch.sum(beta_star.squeeze()**2 * diag / (diag + nu2)**2)

    chi = compute_chi(Sigma, nu2, p)

    B = nu2/nu1 * term1 - lambd/n * nu2**2/nu1**2 * (term2 + chi * term3)/(1 - Upsilon)
    return B.item()



# ============== Main modification: Define a set of lambd values for testing ==============
lambd_list = [1e-6, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 1, 5]  # Can define more as needed

# Parameter definitions
n = 100
k = 800
alpha = 1.5
r = 0.4
sigma = 0.1
num_experiments = 50  # Number of experiments
gamma_vals = np.linspace(0.1, 4, num=100)  # Fewer points for demonstration, can be changed to num=100

beta_star = np.array([i ** (-(1 + 2 * alpha * r) / 2) for i in range(1, k + 1)]).reshape(-1, 1)
xi = 1 / ((np.arange(k) + 1) ** alpha)
Sigma = np.diag(xi)

# Store all results in a dictionary for easy comparison
results_dict = {}

# ---------------------Multiple lambd loop---------------------
for lambd in lambd_list:
    print(f"\n\n===== Testing lambd = {lambd} =====")

    # Initialize lists to store results from multiple experiments (for current lambd)
    bias_beta_experiments = []
    variance_beta_experiments = []
    beta_experiments = []
    bias_risk_experiments = []
    variance_risk_experiments = []
    risk_experiments = []

    V_beta_experiments = []
    B_beta_experiments = []
    D_beta_experiments = []
    V_risk_experiments = []
    B_risk_experiments = []
    D_risk_experiments = []

    # ============ Perform multiple experiments =============
    for exp_id in range(num_experiments):
        print(f"  - Experiment {exp_id+1}/{num_experiments}")

        G = np.array([np.random.randn(k) for _ in range(n)])  # G is n x k matrix
        # G = np.random.choice([-1, 1], size=(n, k))  # n x k matrix, randomly choose -1 or 1

        # Store results for different m values in a single experiment
        bias_beta_vals = []
        variance_beta_vals = []
        beta_vals = []
        bias_risk_vals = []
        variance_risk_vals = []
        risk_vals = []

        V_beta_vals = []
        B_beta_vals = []
        D_beta_vals = []
        V_risk_vals = []
        B_risk_vals = []
        D_risk_vals = []

        for gamma in gamma_vals:
            p = int(n * gamma)

            lambda_star = solve_lambda_star(Sigma, lambd, n, lambd/n)

            F = np.array([np.sqrt(xi) * np.random.randn(k) for _ in range(p)])  # F is p x k matrix
            # F = np.random.choice([-1, 1], size=(p, k))

            Sigma_F = F @ F.T / p

            nu1 = solve_lambda_star(Sigma_F, lambd, n, lambd/n**2)
            print("nu1:", nu1)
            nu2 = solve_lambda_star(Sigma, p*nu1, p, nu1)
            print("nu2:", nu2)

            Z = G @ F.T / math.sqrt(p)

            bias_risk = compute_bias_risk(G, F, Z, beta_star, lambd)
            variance_risk = compute_variance_risk(Z, Sigma_F, sigma, lambd)
            bias_beta = compute_bias_beta(G, Z, beta_star, lambd)
            variance_beta = compute_variance_beta(Z, sigma, lambd)

            Upsilon = compute_Upsilon(torch.from_numpy(Sigma), nu1, nu2, n, p)

            V_risk = compute_V_risk(torch.from_numpy(Sigma), nu1, nu2, sigma, n, p, Upsilon)
            B_risk = compute_B_risk(torch.from_numpy(Sigma), nu2, torch.from_numpy(beta_star), n, p, Upsilon)

            V_beta = compute_V_beta(torch.from_numpy(Sigma), nu1, nu2, sigma, n, p, Upsilon)
            B_beta = compute_B_beta(torch.from_numpy(Sigma), lambd, nu1, nu2, torch.from_numpy(beta_star), n, p, Upsilon)

            # Save results
            bias_beta_vals.append(bias_beta)
            variance_beta_vals.append(variance_beta)
            beta_vals.append(bias_beta + variance_beta)
            bias_risk_vals.append(bias_risk)
            variance_risk_vals.append(variance_risk)
            risk_vals.append(bias_risk + variance_risk)

            V_risk_vals.append(V_risk)
            B_risk_vals.append(B_risk)
            D_risk_vals.append(V_risk + B_risk)
            V_beta_vals.append(V_beta)
            B_beta_vals.append(B_beta)
            D_beta_vals.append(V_beta + B_beta)

        # Store results for each gamma value from a single experiment
        bias_beta_experiments.append(bias_beta_vals)
        variance_beta_experiments.append(variance_beta_vals)
        beta_experiments.append(beta_vals)
        bias_risk_experiments.append(bias_risk_vals)
        variance_risk_experiments.append(variance_risk_vals)
        risk_experiments.append(risk_vals)

        V_beta_experiments.append(V_beta_vals)
        B_beta_experiments.append(B_beta_vals)
        D_beta_experiments.append(D_beta_vals)
        V_risk_experiments.append(V_risk_vals)
        B_risk_experiments.append(B_risk_vals)
        D_risk_experiments.append(D_risk_vals)

    # ============ Average results from multiple experiments (or other statistics) ============
    bias_beta_mean = np.nanmean(bias_beta_experiments, axis=0)
    variance_beta_mean = np.nanmean(variance_beta_experiments, axis=0)
    beta_mean = np.nanmean(beta_experiments, axis=0)
    bias_risk_mean = np.nanmean(bias_risk_experiments, axis=0)
    variance_risk_mean = np.nanmean(variance_risk_experiments, axis=0)
    risk_mean = np.nanmean(risk_experiments, axis=0)

    B_beta_mean = np.nanmean(B_beta_experiments, axis=0)
    V_beta_mean = np.nanmean(V_beta_experiments, axis=0)
    D_beta_mean = np.nanmean(D_beta_experiments, axis=0)
    B_risk_mean = np.nanmean(B_risk_experiments, axis=0)
    V_risk_mean = np.nanmean(V_risk_experiments, axis=0)
    D_risk_mean = np.nanmean(D_risk_experiments, axis=0)

    # Store average results for current lambd in dictionary
    results_dict[lambd] = {
        "gamma_vals": gamma_vals,
        "bias_beta_mean": bias_beta_mean,
        "variance_beta_mean": variance_beta_mean,
        "beta_mean": beta_mean,
        "bias_risk_mean": bias_risk_mean,
        "variance_risk_mean": variance_risk_mean,
        "risk_mean": risk_mean,
        "B_beta_mean": B_beta_mean,
        "V_beta_mean": V_beta_mean,
        "D_beta_mean": D_beta_mean,
        "B_risk_mean": B_risk_mean,
        "V_risk_mean": V_risk_mean,
        "D_risk_mean": D_risk_mean,
    }

# ============ Finally, save all lambd results to an npz file ============
# For easy reading and comparison of different lambd performances
np.savez("experiment_results_all_lambda.npz", results=results_dict)
print("All data saved to 'experiment_results_all_lambda.npz'.")


