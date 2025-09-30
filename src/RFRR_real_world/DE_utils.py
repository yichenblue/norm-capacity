import math
import numpy as np
from scipy.optimize import fsolve
import torch


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



class RidgeRegression:
    def __init__(self, lambda_reg=1.0):
        self.lambda_reg = lambda_reg
        self.beta = None

    def fit(self, X, y):
        """
        Fit Ridge regression using the analytical solution
        β = (X^T X + λI)^(-1) X^T y
        """
        n = X.shape[1]
        identity = torch.eye(n, device=X.device, dtype=X.dtype)

        # Compute X^T X + λI
        XTX = torch.matmul(X.T, X)
        reg_term = self.lambda_reg * identity
        XTX_reg = XTX + reg_term

        # Compute (X^T X + λI)^(-1)
        XTX_reg_inv = torch.linalg.inv(XTX_reg)

        # Compute X^T y
        XTy = torch.matmul(X.T, y)

        # Compute β
        self.beta = torch.matmul(XTX_reg_inv, XTy)

    def predict(self, X):
        """Make predictions using the fitted parameters"""
        return torch.matmul(X, self.beta)