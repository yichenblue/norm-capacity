import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D


# ---------- Load Data ----------
data_all = np.load("experiment_results_all_lambda.npz", allow_pickle=True)

results_dict = data_all["results"].item()

# ---------- Set Lambda Values ----------
lambd_keys = sorted(results_dict.keys())  # Automatically get all lambda values and sort them
min_lambda = min(lambd_keys)  # Get the minimum lambda value as the ridgeless case

# colormap excludes the minimum lambda (black line), assigns colors to the rest
cmap = colormaps.get_cmap('plasma')
colors = [cmap(i / (len(lambd_keys) - 2)) for i in range(len(lambd_keys) - 1)]

# ---------- Unified Legend ----------
legend_lines = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=3, label="ridgeless"),
    Line2D([0], [0], color='black', linewidth=2, label='theory'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='empirical')
]

# ========== Figure 1: Norm vs Gamma ==========
plt.figure(figsize=(10, 8), dpi=120)

for i, lambd in enumerate(lambd_keys):
    gamma_vals = results_dict[lambd]["gamma_vals"]
    beta_mean = results_dict[lambd]["beta_mean"]
    D_beta_mean = results_dict[lambd]["D_beta_mean"]

    if np.isclose(lambd, min_lambda):
        plt.plot(gamma_vals, D_beta_mean, color='black', linestyle='--', linewidth=6)
    else:
        color = colors[i - 1]  # shift index because min_lambda is excluded from colormap
        plt.plot(gamma_vals, D_beta_mean, color=color, linewidth=4)
        plt.scatter(gamma_vals, beta_mean, color=color, s=20)

plt.legend(handles=legend_lines, fontsize=28, loc='best')
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
plt.title(r"$\ell_2$ norm vs. $\gamma$", fontsize=32, pad=20)
plt.xlabel(r'$\gamma \quad (\frac{d}{n})$', fontsize=32)
plt.ylabel(r'$\ell_2$ norm', fontsize=32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(0, 3)
plt.ylim(1, 2)
plt.tight_layout()
plt.savefig("norm.pdf")
plt.show()

# ========== Figure 2: Risk vs Gamma ==========
plt.figure(figsize=(10, 8), dpi=120)
for i, lambd in enumerate(lambd_keys):
    gamma_vals = results_dict[lambd]["gamma_vals"]
    risk_mean = results_dict[lambd]["risk_mean"]
    D_risk_mean = results_dict[lambd]["D_risk_mean"]

    if np.isclose(lambd, min_lambda):
        plt.plot(gamma_vals, D_risk_mean, color='black', linestyle='--', linewidth=6, label='ridgeless')
    else:
        color = colors[i - 1]
        plt.plot(gamma_vals, D_risk_mean, color=color, linewidth=4, label=f'$\\lambda={lambd}$')
        plt.scatter(gamma_vals, risk_mean, color=color, s=20)

# Set legend
plt.legend(fontsize=20, loc='best', ncol=2)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
plt.title(r"Test Risk vs. $\gamma$", fontsize=32, pad=20)
plt.xlabel(r'$\gamma \quad (\frac{d}{n})$', fontsize=32)
plt.ylabel("test risk", fontsize=32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(0, 3)
plt.ylim(0, 0.004)
plt.tight_layout()
plt.savefig("risk.pdf")
plt.show()

# ========== Figure 3: Risk vs Norm ==========
plt.figure(figsize=(10, 8), dpi=120)
for i, lambd in enumerate(lambd_keys):
    beta_mean = results_dict[lambd]["beta_mean"]
    D_beta_mean = results_dict[lambd]["D_beta_mean"]
    risk_mean = results_dict[lambd]["risk_mean"]
    D_risk_mean = results_dict[lambd]["D_risk_mean"]

    if np.isclose(lambd, min_lambda):
        plt.plot(D_beta_mean, D_risk_mean, color='black', linestyle='--', linewidth=6)
    else:
        color = colors[i - 1]
        plt.plot(D_beta_mean, D_risk_mean, color=color, linewidth=4)
        plt.scatter(beta_mean, risk_mean, color=color, s=20)

plt.legend(handles=legend_lines, fontsize=28, loc='best')
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
plt.title(r"Test Risk vs. $\ell_2$ norm", fontsize=32, pad=20)
plt.xlabel(r'$\ell_2$ norm', fontsize=32)
plt.ylabel("test risk", fontsize=32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(1.1, 1.6)
plt.ylim(0, 0.0015)
plt.tight_layout()
plt.savefig("risk_vs_norm.pdf")
plt.show()


