import json
import matplotlib.pyplot as plt
import numpy as np
import utils
import os

datasetname = 'MNIST'


def load_results(dataset):
    PATH_TO_RESULTS = utils.get_path_to('results')
    result_file = os.path.join(PATH_TO_RESULTS, dataset + 'RFRR.json')
    with open(result_file, 'r') as f:
        results = json.load(f)
    return results


def plot_test_loss_vs_features(results, save_path=None):
    plt.figure(figsize=(10, 8))  # Increase figure size
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
    lambda_vals = list(results.keys())
    lambda_vals_sorted = sorted(lambda_vals, key=lambda x: float(x))
    min_lambda = lambda_vals_sorted[0]

    # Add dummy plots for legend
    plt.plot([], [], 'o', color='black', markersize=8, label='Empirical')
    plt.plot([], [], '-', color='black', linewidth=3, label='Theoretical')
    plt.plot([], [], '--', color='black', linewidth=4, label=r'ridgeless ($\lambda \to 0$)')

    # Plot ridgeless first with zorder=1 (bottom layer)
    data = results[min_lambda]
    if 'test_losses_DE_mean' in data:
        plt.plot(data['p_x_axis'], data['test_losses_DE_mean'], linestyle='--', linewidth=6,
                 color='#404040', zorder=1)

    # Plot lambda=0.005 with viridis colormap and zorder=2 (top layer)
    target_lambda = '0.005'
    if target_lambda in results:
        data = results[target_lambda]
        p_vals = data['p_x_axis']
        # Create viridis colormap
        viridis = plt.colormaps['viridis']
        # Normalize p values to [0,1] for colormap
        norm = plt.Normalize(min(p_vals), max(p_vals))
        # Plot each point with its corresponding color
        for p, loss in zip(p_vals, data['test_losses_mean']):
            color = viridis(norm(p))
            plt.plot(p, loss, 'o', color=color, markersize=8, zorder=2)
        # Plot theoretical line with gradient
        if 'test_losses_DE_mean' in data:
            for i in range(len(p_vals)-1):
                color = viridis(norm(p_vals[i]))
                plt.plot(p_vals[i:i+2], data['test_losses_DE_mean'][i:i+2], '-', color=color, linewidth=5, zorder=2)

    plt.xlabel(r'$p$', fontsize=24)
    plt.ylabel('Test risk', fontsize=24)
    plt.title(r'Test risk vs. $p$', fontsize=24)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if datasetname == 'MNIST':
        plt.xlim(-50, 1250)
        plt.ylim(1, 20)
        plt.xticks(np.linspace(0, 1200, 5))
        plt.yticks(np.linspace(2, 20, 5))
    elif datasetname == 'FashionMNIST':
        plt.xlim(-50, 1250)
        plt.ylim(2.5, 20)
        plt.xticks(np.linspace(0, 1200, 5))
        plt.yticks(np.linspace(5, 20, 5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_norm_vs_features(results, save_path=None):
    plt.figure(figsize=(10, 8))
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
    lambda_vals = list(results.keys())
    lambda_vals_sorted = sorted(lambda_vals, key=lambda x: float(x))
    min_lambda = lambda_vals_sorted[0]

    # Plot ridgeless first with zorder=1 (bottom layer)
    data = results[min_lambda]
    if 'l2_norms_DE_mean' in data:
        plt.plot(data['p_x_axis'], data['l2_norms_DE_mean'], linestyle='--', linewidth=6,
                 color='#404040', label='ridgeless', zorder=1)

    # Plot lambda=0.005 with viridis colormap and zorder=2 (top layer)
    target_lambda = '0.005'
    if target_lambda in results:
        data = results[target_lambda]
        p_vals = data['p_x_axis']
        # Create viridis colormap
        viridis = plt.colormaps['viridis']
        # Normalize p values to [0,1] for colormap
        norm = plt.Normalize(min(p_vals), max(p_vals))
        # Plot each point with its corresponding color
        for p, norm_val in zip(p_vals, data['l2_norms_mean']):
            color = viridis(norm(p))
            plt.plot(p, norm_val, 'o', color=color, markersize=8, zorder=2)
        # Plot theoretical line with gradient
        if 'l2_norms_DE_mean' in data:
            for i in range(len(p_vals)-1):
                color = viridis(norm(p_vals[i]))
                plt.plot(p_vals[i:i+2], data['l2_norms_DE_mean'][i:i+2], '-', color=color, linewidth=5, zorder=2)

    plt.xlabel(r'$p$', fontsize=24)
    plt.ylabel(r'$\ell_2$ norm', fontsize=24)
    plt.title(r'$\ell_2$ norm vs. $p$', fontsize=24)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if datasetname == 'MNIST':
        plt.xlim(-50, 1250)
        plt.ylim(-1000, 20000)
        plt.xticks(np.linspace(0, 1200, 5))
        plt.yticks(np.linspace(0, 20000, 5))
    elif datasetname == 'FashionMNIST':
        plt.xlim(-50, 1250)
        plt.ylim(-100000, 1.2e6)
        plt.xticks(np.linspace(0, 1200, 5))
        plt.yticks(np.linspace(0, 1.2e6, 5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_test_error_vs_norm(results, save_path=None):
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
    lambda_vals = list(results.keys())
    lambda_vals_sorted = sorted(lambda_vals, key=lambda x: float(x))
    min_lambda = lambda_vals_sorted[0]

    # Plot ridgeless first with zorder=1 (bottom layer)
    data = results[min_lambda]
    if 'l2_norms_DE_mean' in data and 'test_losses_DE_mean' in data:
        plt.plot(data['l2_norms_DE_mean'], data['test_losses_DE_mean'],
                 linestyle='--', linewidth=6, color='#404040', label='ridgeless', zorder=1)

    # Plot lambda=0.005 with viridis colormap and zorder=2 (top layer)
    target_lambda = '0.005'
    if target_lambda in results:
        data = results[target_lambda]
        p_vals = data['p_x_axis']
        # Create viridis colormap
        viridis = plt.colormaps['viridis']
        # Normalize p values to [0,1] for colormap
        norm = plt.Normalize(min(p_vals), max(p_vals))
        # Plot each point with its corresponding color
        for p, norm_val, error in zip(p_vals, data['l2_norms_mean'], data['test_losses_mean']):
            color = viridis(norm(p))
            plt.plot(norm_val, error, 'o', color=color, markersize=8, zorder=2)
        # Plot theoretical line with gradient
        if 'l2_norms_DE_mean' in data and 'test_losses_DE_mean' in data:
            for i in range(len(p_vals)-1):
                color = viridis(norm(p_vals[i]))
                plt.plot(data['l2_norms_DE_mean'][i:i+2], data['test_losses_DE_mean'][i:i+2], '-', color=color, linewidth=5, zorder=2)
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
        sm.set_array([])  # This line is necessary
        cbar = plt.colorbar(sm, ax=plt.gca(), label='p')
        cbar.ax.tick_params(labelsize=24)
        cbar.set_label(r'$p$', size=24)

    plt.xlabel(r'$\ell_2$ norm', fontsize=24)
    plt.ylabel('Test risk', fontsize=24)
    plt.title(r'Test risk vs. $\ell_2$ norm', fontsize=24)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if datasetname == 'MNIST':
        plt.xlim(-1000, 15000)
        plt.ylim(1, 20)
        plt.xticks(np.linspace(0, 15000, 5))
        plt.yticks(np.linspace(2, 20, 5))
    elif datasetname == 'FashionMNIST':
        plt.xlim(-100000, 1e6)
        plt.ylim(2.5, 20)
        plt.xticks(np.linspace(0, 1e6, 5))
        plt.yticks(np.linspace(5, 20, 5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()



def main():
    results = load_results(datasetname)
    save_dir = utils.get_path_to('results')
    os.makedirs(save_dir, exist_ok=True)

    plot_test_loss_vs_features(results, os.path.join(save_dir, f'test_error_vs_features_{datasetname}.pdf'))
    plot_norm_vs_features(results, os.path.join(save_dir, f'norm_vs_features_{datasetname}.pdf'))
    plot_test_error_vs_norm(results, os.path.join(save_dir, f'test_error_vs_norm_{datasetname}.pdf'))

if __name__ == "__main__":
    main()
