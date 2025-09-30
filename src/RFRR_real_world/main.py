import utils
import argparse
from DE_utils import *

dataset_name = "MNIST"

PATH_TO_RESULTS = utils.get_path_to('results')
PATH_TO_DATA = utils.get_path_to('data')


# process arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=dataset_name)
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()
dataset = args.dataset
device = args.device

prepare_data = True

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
N_TOTAL = 10000  # Total number of training samples
N_SAMPLES = 300  # Number of samples for each experiment
N_MAX = 10000
NOISE_LEVEL = 0  # Noise level
sigma = NOISE_LEVEL

if prepare_data:
    print(f'\n\n\nLoading {dataset} data...')
    X_train, y_train, X_test, y_test = utils.process_data(dataset, N_TOTAL, device)
    np.save(PATH_TO_DATA + 'RFF_y_train.npy', y_train)
    np.save(PATH_TO_DATA + 'RFF_y_test.npy', y_test)
    del y_train
    del y_test

    print('Preparing RFFs...')
    # prepare and save the RFF up front
    # v = torch.randn(size=(X_train.shape[1], N_MAX)).to(device) * (1/28)**0.5   # coefficents of RFF v: (784, N_MAX)
    v = torch.randn(size=(X_train.shape[1], N_MAX)).to(device)  # coefficents of RFF v: (784, N_MAX)
    np.save(PATH_TO_DATA + 'RFF_v.npy', v.cpu().numpy())  # save the random coefficients v

    RFF_train_full = utils.generate_erf(v, X_train).to(torch.float64)  # RFF_train_full: (N_TOTAL, N_MAX)
    np.save(PATH_TO_DATA + 'RFF_train_full.npy', RFF_train_full)  # save the RFF to memory
    del RFF_train_full
    RFF_test_full = utils.generate_erf(v, X_test).to(torch.float64)  # RFF_test_full: (10000, N_MAX)
    np.save(PATH_TO_DATA + 'RFF_test_full.npy', RFF_test_full)  # save the RFF to memory
    del RFF_test_full


# load all y values and RFF features
y_train_full = np.load(PATH_TO_DATA + 'RFF_y_train.npy')
y_test = np.load(PATH_TO_DATA + 'RFF_y_test.npy')
RFF_train_full = np.load(PATH_TO_DATA + 'RFF_train_full.npy')
RFF_test_full = np.load(PATH_TO_DATA + 'RFF_test_full.npy')
v = np.load(PATH_TO_DATA + 'RFF_v.npy')  # Load the random coefficients v

# Convert to torch tensors
y_train_full = torch.from_numpy(y_train_full).to(torch.float64).to(device)
y_test = torch.from_numpy(y_test).to(torch.float64).to(device)
RFF_train_full = torch.from_numpy(RFF_train_full).to(torch.float64).to(device)
RFF_test_full = torch.from_numpy(RFF_test_full).to(torch.float64).to(device)
v = torch.from_numpy(v).to(torch.float64).to(device)  # Convert v to torch tensor


# Convert to torch.Tensor
X = RFF_train_full  # (N, P)
y = y_train_full  # (N,)

N, P = X.shape

# Step 1: Construct empirical kernel matrix K~ (N x N)
print("Constructing empirical kernel matrix...")
K_tilde = (1 / (N * P)) * X @ X.T

# Step 2: Eigendecomposition K~ = \sum \xi_k \psi_k \psi_k^T
print("Performing eigendecomposition...")
xi_tilde, psi_tilde = torch.linalg.eigh(K_tilde)  # (N,), (N, N)

xi_tilde = xi_tilde.clamp(min=0.0)
psi_tilde = psi_tilde * torch.sqrt(torch.tensor(N_TOTAL, dtype=psi_tilde.dtype, device=psi_tilde.device))

# Step 3: Compute beta_k coefficients in empirical coordinate system
print("Computing beta_k coefficients...")
beta_star = torch.matmul(psi_tilde.T, y) / N  # (N,)

# Step 4: Construct diagonal matrix Σ = diag(ξ₁,...,ξ_N)
Sigma = torch.diag(xi_tilde)  # (N, N)

Sigma_np = Sigma.numpy()

A = psi_tilde @ psi_tilde.T




lambda_list = [1e-4, 0.005]  # MNIST

p_vals = (np.arange(1, 1200, 10).tolist())

n_experiments = 1  # Number of experiment repetitions

def calculate_path_norm_for_ridge(model, RFF_train_p, v_p):
    """
    Calculate path norm for Ridge regression model
    model: Ridge regression model
    RFF_train_p: Current RFF features
    v_p: Current random feature matrix
    """
    with torch.no_grad():
        # Get model parameters
        beta = model.beta  # (p,)

        # Calculate complete path norm from input to output
        # v_p: (d, p), beta: (p,)
        path_norm = torch.sum((v_p ** 2) @ (beta ** 2))

    return path_norm.item()

# Initialize results dictionary
results_dict = {str(lambd): {
    'test_errors_all': [],  # Store error rates for all experiments
    'test_losses_all': [],  # Store losses for all experiments
    'l2_norms_all': [],
    'test_losses_DE_all': [],
    'l2_norms_DE_all': [],
    'v_frob_norms_all': [],     # Modified to _all suffix
    'path_norms_all': [],      # Modified to _all suffix
    'V_risk_all': [],          # New: Store V_risk
    'B_risk_all': [],          # New: Store B_risk
    'test_errors_mean': [],  # Store mean error rates
    'test_errors_std': [],   # Store error rate standard deviations
    'test_losses_mean': [],  # Store mean losses
    'test_losses_std': [],   # Store loss standard deviations
    'l2_norms_mean': [],
    'l2_norms_std': [],
    'test_losses_DE_mean': [],
    'test_losses_DE_std': [],
    'l2_norms_DE_mean': [],
    'l2_norms_DE_std': [],
    'v_frob_norms_mean': [],    # Add mean
    'v_frob_norms_std': [],     # Add standard deviation
    'path_norms_mean': [],      # Add mean
    'path_norms_std': [],       # Add standard deviation
    'V_risk_mean': [],          # New: V_risk mean
    'V_risk_std': [],           # New: V_risk standard deviation
    'B_risk_mean': [],          # New: B_risk mean
    'B_risk_std': [],           # New: B_risk standard deviation
    'p_x_axis': p_vals
} for lambd in lambda_list}

for exp_id in range(n_experiments):
    print(f'\n\nRunning experiment {exp_id + 1}/{n_experiments}')

    # Generate new random seed for each experiment
    current_seed = SEED + exp_id
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)

    # Randomly select training sample indices (choose N_SAMPLES from 0 to N_TOTAL-1)
    indices = np.random.choice(N_TOTAL, N_SAMPLES, replace=False)
    y_train_clean = y_train_full[indices]

    # Add Gaussian noise
    noise = torch.randn_like(y_train_clean) * NOISE_LEVEL
    y_train = y_train_clean + noise
    #y_train = y_train_clean

    RFF_train = RFF_train_full[indices]

    for lambda_idx, lambd in enumerate(lambda_list):
        print(f'\nRunning lambda = {lambd}...')
        l2_norms = []
        test_errors = []
        test_losses = []        # New: Store current experiment's loss
        l2_norms_DE = []
        test_losses_DE = []
        v_frob_norms = []    # Add list to collect single experiment data
        path_norms = []      # Add list to collect single experiment data
        V_risks = []         # New: Store V_risk
        B_risks = []         # New: Store B_risk

        for p in p_vals:
            print(f'Working on p = {p}')
            n = N_SAMPLES

            nu1, nu2 = solve_self_consistent_equations(Sigma, n, p, lambd)
            print(f"nu1 = {nu1}, nu2 = {nu2}")

            # Get RFF feature subset for current p
            RFF_train_p = RFF_train[:, :p] / np.sqrt(p)
            RFF_test_p = RFF_test_full[:, :p] / np.sqrt(p)
            v_p = v[:, :p]

            # Calculate Frobenius norm of v_p
            v_frob_norm = torch.norm(v_p, p='fro').item()
            v_frob_norms.append(v_frob_norm)

            # fit models
            model = RidgeRegression(lambda_reg=lambd)
            model.fit(RFF_train_p, y_train)

            # Calculate path norm
            path_norm = calculate_path_norm_for_ridge(model, RFF_train_p, v_p)
            path_norms.append(path_norm)

            # evaluate error on the test set
            y_pred = model.predict(RFF_test_p)
            # Calculate error rate
            y_pred_rounded = torch.round(y_pred)
            test_error = torch.mean((y_pred_rounded != y_test).float()).item()
            test_errors.append(test_error)

            # Calculate loss
            test_loss = torch.mean((y_pred - y_test)**2).item()
            test_losses.append(test_loss)

            l2_norm = torch.norm(model.beta, p=2)**2
            l2_norms.append(l2_norm.item())

            Upsilon = compute_Upsilon(Sigma, nu1, nu2, n, p)
            V_risk = compute_V_risk(Sigma, nu1, nu2, sigma, n, p, Upsilon)
            B_risk = compute_B_risk(Sigma, nu2, beta_star, n, p, Upsilon)
            test_losses_DE.append(V_risk.item() + B_risk)
            V_risks.append(V_risk.item())  # New: Record V_risk
            B_risks.append(B_risk)         # New: Record B_risk

            V_beta = compute_V_beta(Sigma, nu1, nu2, sigma, n, p, Upsilon)
            B_beta = compute_B_beta(Sigma, lambd, nu1, nu2, beta_star, n, p, Upsilon)
            l2_norms_DE.append(V_beta.item() + B_beta)

            print("error rate:", test_error)
            print("loss:", test_loss)
            print("DE:", V_risk + B_risk)
            print("norm:", l2_norm)
            print("DE:", V_beta + B_beta)

        # Store current experiment results
        results_dict[str(lambd)]['test_errors_all'].append(test_errors)
        results_dict[str(lambd)]['test_losses_all'].append(test_losses)  # New: Store loss
        results_dict[str(lambd)]['l2_norms_all'].append(l2_norms)
        results_dict[str(lambd)]['test_losses_DE_all'].append(test_losses_DE)
        results_dict[str(lambd)]['l2_norms_DE_all'].append(l2_norms_DE)
        results_dict[str(lambd)]['v_frob_norms_all'].append(v_frob_norms)
        results_dict[str(lambd)]['path_norms_all'].append(path_norms)
        results_dict[str(lambd)]['V_risk_all'].append(V_risks)           # New: Store V_risk
        results_dict[str(lambd)]['B_risk_all'].append(B_risks)           # New: Store B_risk

# Calculate means and standard deviations
for lambd in lambda_list:
    lambda_key = str(lambd)
    test_errors_all = np.array(results_dict[lambda_key]['test_errors_all'])
    test_losses_all = np.array(results_dict[lambda_key]['test_losses_all'])  # New: Get loss data
    l2_norms_all = np.array(results_dict[lambda_key]['l2_norms_all'])
    test_losses_DE_all = np.array(results_dict[lambda_key]['test_losses_DE_all'])
    l2_norms_DE_all = np.array(results_dict[lambda_key]['l2_norms_DE_all'])
    v_frob_norms_all = np.array(results_dict[lambda_key]['v_frob_norms_all'])
    path_norms_all = np.array(results_dict[lambda_key]['path_norms_all'])
    V_risk_all = np.array(results_dict[lambda_key]['V_risk_all'])        # New: Get V_risk data
    B_risk_all = np.array(results_dict[lambda_key]['B_risk_all'])        # New: Get B_risk data

    # Calculate means and standard deviations
    results_dict[lambda_key]['test_errors_mean'] = np.mean(test_errors_all, axis=0).tolist()
    results_dict[lambda_key]['test_errors_std'] = np.std(test_errors_all, axis=0).tolist()
    results_dict[lambda_key]['test_losses_mean'] = np.mean(test_losses_all, axis=0).tolist()  # New: Calculate loss mean
    results_dict[lambda_key]['test_losses_std'] = np.std(test_losses_all, axis=0).tolist()    # New: Calculate loss std
    results_dict[lambda_key]['l2_norms_mean'] = np.mean(l2_norms_all, axis=0).tolist()
    results_dict[lambda_key]['l2_norms_std'] = np.std(l2_norms_all, axis=0).tolist()
    results_dict[lambda_key]['test_losses_DE_mean'] = np.mean(test_losses_DE_all, axis=0).tolist()
    results_dict[lambda_key]['test_losses_DE_std'] = np.std(test_losses_DE_all, axis=0).tolist()
    results_dict[lambda_key]['l2_norms_DE_mean'] = np.mean(l2_norms_DE_all, axis=0).tolist()
    results_dict[lambda_key]['l2_norms_DE_std'] = np.std(l2_norms_DE_all, axis=0).tolist()
    results_dict[lambda_key]['v_frob_norms_mean'] = np.mean(v_frob_norms_all, axis=0).tolist()
    results_dict[lambda_key]['v_frob_norms_std'] = np.std(v_frob_norms_all, axis=0).tolist()
    results_dict[lambda_key]['path_norms_mean'] = np.mean(path_norms_all, axis=0).tolist()
    results_dict[lambda_key]['path_norms_std'] = np.std(path_norms_all, axis=0).tolist()
    results_dict[lambda_key]['V_risk_mean'] = np.mean(V_risk_all, axis=0).tolist()           # New: Calculate V_risk mean
    results_dict[lambda_key]['V_risk_std'] = np.std(V_risk_all, axis=0).tolist()             # New: Calculate V_risk std
    results_dict[lambda_key]['B_risk_mean'] = np.mean(B_risk_all, axis=0).tolist()           # New: Calculate B_risk mean
    results_dict[lambda_key]['B_risk_std'] = np.std(B_risk_all, axis=0).tolist()             # New: Calculate B_risk std

# Save results
utils.save_results(results_dict, PATH_TO_RESULTS + dataset + 'RFRR.json')
print("\nAll experiments completed and results saved.")
