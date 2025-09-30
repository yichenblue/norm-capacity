import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import sqrtm
from scipy.integrate import quad


def solve_lambda_star(Sigma, lambd, n, initial_guess=1e-6):
    """
    Solve for λ_star such that the equation n - λ / λ_star = Tr(Σ (Σ + λ_star)^-1) holds.

    Parameters:
    Sigma (ndarray): Covariance matrix (d x d).
    lambd (float): Regularization parameter λ.
    n (int): Number of samples.

    Returns:
    float: Solution for λ_star.
    """
    d = Sigma.shape[0]  # Feature dimension

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


def compute_variance_beta(X, sigma, lambd):
    n, d = X.shape
    XX = X.T @ X
    XX_inv = np.linalg.solve(XX + lambd * np.eye(d), np.eye(d))

    # Variance
    variance = sigma**2 * np.trace(XX @ XX_inv @ XX_inv)
    return variance


def compute_bias_beta(X, beta_star, lambd):
    """
    Compute bias component:
    ||S (S.T X.T X S + λI)^-1 S.T X.T X θ_* ||_2^2
    """
    n, d = X.shape
    XX = X.T @ X
    XX_inv = np.linalg.solve(XX + lambd * np.eye(d), np.eye(d))

    # Bias
    bias = np.linalg.norm(XX @ XX_inv @ beta_star, 2)**2
    return bias


def compute_variance_risk(X, Sigma, sigma, lambd):
    """
    Compute E[R^{var}]:
    sigma^2 Tr(S.T Σ S (S.T X.T X S + λI)^-2 S.T X.T X S)
    """
    n, d = X.shape
    XX = X.T @ X
    XX_inv = np.linalg.solve(XX + lambd * np.eye(d), np.eye(d))

    # Variance Risk
    variance_risk = sigma ** 2 * np.trace(Sigma @ XX @ XX_inv @ XX_inv)
    return variance_risk


def compute_bias_risk(X, Sigma, beta_star, lambd):
    """
    Compute R^{bias}:
    θ_*^T Σ θ_* - 2 θ_*^T M.T Σ θ_* + θ_*^T M.T Σ M θ_*
    This is a simplified expression for demonstration. The actual definition may vary based on requirements.
    """
    n, d = X.shape
    XX = X.T @ X
    XX_inv = np.linalg.solve(XX + lambd * np.eye(d), np.eye(d))

    # Bias Risk (demonstrative implementation, can be adjusted based on actual requirements)
    # Example: lambd^2 * trace( beta_star.T @ XX_inv @ Sigma @ XX_inv @ beta_star )
    bias_risk = lambd ** 2 * np.trace(
        beta_star.T @ XX_inv @ Sigma @ XX_inv @ beta_star
    )
    return bias_risk


def compute_V_risk(Sigma, lambd, sigma, n, d):
    """
    Provide a large-dimensional analysis approximation formula for variance (example).
    Can be replaced with actual theoretical values as needed.
    sigma^2 * df2 / (n - df2)
    """
    Sigma_inv = np.linalg.solve(Sigma + lambd * np.eye(d), np.eye(d))
    df2 = np.trace(Sigma @ Sigma @ Sigma_inv @ Sigma_inv)

    variance_R = sigma ** 2 * df2 / (n - df2)
    return variance_R


def compute_B_risk(Sigma, lambd, beta_star, n, d):
    """
    Provide a large-dimensional analysis approximation formula for bias (example).
    Can be replaced with actual theoretical values as needed.
    lambd^2 * beta_star^T Sigma Sigma_inv^2 beta_star / (1 - 1/n * df2)
    """
    Sigma_inv = np.linalg.solve(Sigma + lambd * np.eye(d), np.eye(d))
    Sigma_inv2 = Sigma_inv @ Sigma_inv
    df2 = np.trace(Sigma @ Sigma @ Sigma_inv @ Sigma_inv)

    bias_R = (
        lambd ** 2 * beta_star.T @ Sigma @ Sigma_inv2 @ beta_star
        / (1 - 1 / n * df2)
    )
    return bias_R.item()


def compute_V_beta(Sigma, lambd, sigma, n, d):
    """
    Provide a large-dimensional analysis approximation formula for coefficient estimation variance (example).
    sigma^2 * trace(Sigma (Sigma + lambd I)^-2) / (n - df2)
    """
    Sigma_inv = np.linalg.solve(Sigma + lambd * np.eye(d), np.eye(d))
    Sigma_inv2 = Sigma_inv @ Sigma_inv
    df2 = np.trace(Sigma @ Sigma @ Sigma_inv @ Sigma_inv)

    V = sigma ** 2 * np.trace(Sigma @ Sigma_inv2) / (n - df2)
    return V


def compute_B_beta(Sigma, lambd, lambda_star, beta_star, n, d):
    """
    Provide a large-dimensional analysis approximation formula for coefficient estimation bias (example).
    """
    Sigma_inv = np.linalg.solve(Sigma + lambda_star * np.eye(d), np.eye(d))
    Sigma_inv2 = Sigma_inv @ Sigma_inv

    df2 = np.trace(Sigma @ Sigma @ Sigma_inv @ Sigma_inv)

    term1 = beta_star.T @ Sigma @ Sigma_inv @ beta_star
    term2 = beta_star.T @ Sigma @ Sigma_inv2 @ beta_star

    # An example form (not necessarily the most rigorous formula)
    B = term1 - lambd / n * term2 / (1 - 1 / n * df2)
    return B.item()


# ============== Main modification: Define a set of lambd values for testing ==============
lambd_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]  # Can define more as needed


# Other fixed parameters
d = 1000
alpha = 1
r = 1
sigma = 0.02
num_experiments = 10  # Number of experiments
gamma_vals = np.linspace(0.1, 4, num=100)  # Fewer points for demonstration, can be changed to num=100


# Construct beta_star, Sigma
beta_star = np.array([i ** (-(1 + 2 * alpha * r) / 2) for i in range(1, d + 1)]).reshape(-1, 1)
xi = 1 / ((np.arange(d) + 1) ** alpha)
Sigma = np.diag(xi)

# Store all results in a dictionary for comparison
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

        # Store results for different gamma values in a single experiment
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

        # ============ Loop through different gamma values =============
        for gamma in gamma_vals:
            # Let n = int(d / gamma)
            # If gamma is small, be careful to avoid n being too large causing time/memory issues
            n = int(d / gamma)
            if n < 1:
                # Avoid extreme cases like n=0
                bias_beta_vals.append(np.nan)
                variance_beta_vals.append(np.nan)
                beta_vals.append(np.nan)
                bias_risk_vals.append(np.nan)
                variance_risk_vals.append(np.nan)
                risk_vals.append(np.nan)

                V_risk_vals.append(np.nan)
                B_risk_vals.append(np.nan)
                D_risk_vals.append(np.nan)
                V_beta_vals.append(np.nan)
                B_beta_vals.append(np.nan)
                D_beta_vals.append(np.nan)
                continue

            print(f"gamma={gamma:.2f}, n={n}")

            # X is an n x d matrix, generate data based on given Sigma
            # Assuming X_i ~ N(0, Sigma), equivalent to generating standard normal first, then multiply by sqrt(Sigma)
            # For simplicity, when Sigma is diagonal, use √(xi) * randn
            X = np.array([np.sqrt(xi) * np.random.randn(d) for _ in range(n)])

            # Solve for lambda_star corresponding to current lambd (initial value can be modified)
            lambda_star = solve_lambda_star(Sigma, lambd, n, lambd/100)

            print(r"$\lambda_*$:", lambda_star)

            # Compute risk (experimental measurement)
            bias_risk = compute_bias_risk(X, Sigma, beta_star, lambd)
            variance_risk = compute_variance_risk(X, Sigma, sigma, lambd)
            bias_beta = compute_bias_beta(X, beta_star, lambd)
            variance_beta = compute_variance_beta(X, sigma, lambd)

            # Compute risk (large-dimensional theoretical approximation)
            V_risk = compute_V_risk(Sigma, lambda_star, sigma, n, d)
            B_risk = compute_B_risk(Sigma, lambda_star, beta_star, n, d)
            V_beta = compute_V_beta(Sigma, lambda_star, sigma, n, d)
            B_beta = compute_B_beta(Sigma, lambd, lambda_star, beta_star, n, d)

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

        # Store results for all gamma values from single experiment into corresponding lists
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
# For easy reading and comparison of different lambd performances later
np.savez("experiment_results_all_lambda.npz", results=results_dict)
print("All data saved to 'experiment_results_all_lambda.npz'.")
