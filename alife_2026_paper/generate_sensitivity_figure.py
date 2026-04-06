"""Generate a compact sensitivity analysis figure for the ALIFE paper."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

INPUT_CSV = ROOT / "sensitivity_analysis_results" / "sensitivity_results_twin_top50pct.csv"

PARAM_LABELS = {
    'tau_v':   r'$\tau_v$',
    'tau_w':   r'$\tau_w$',
    'a':       r'$a$',
    'b':       r'$b$',
    'v_scale': r'$v_{scale}$',
    'R_off':   r'$R_{off}$',
    'R_on':    r'$R_{on}$',
    'alpha':   r'$\alpha$',
}

PERTURBATIONS = [5, 10, 20, 30]

def main():
    df = pd.read_csv(INPUT_CSV)
    sns.set_style("whitegrid")

    perturbed = df[df['perturbed_param'] != 'none'].copy()
    perturbed['abs_pct'] = perturbed['perturbation_pct'].abs()

    agg = (
        perturbed
        .groupby(['perturbed_param', 'abs_pct'])['twin_accuracy_drop']
        .agg(['mean', 'std'])
        .reset_index()
    )
    agg.columns = ['param', 'abs_pct', 'mean_drop', 'std_drop']

    max_pct = max(PERTURBATIONS)
    max_impact = agg[agg['abs_pct'] == max_pct].set_index('param')['mean_drop']
    param_order = max_impact.sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left: tornado bar chart at max perturbation
    ax = axes[0]
    colors = sns.color_palette("RdYlGn_r", len(param_order))
    y_labels = [PARAM_LABELS.get(p, p) for p in param_order]
    values = [max_impact.get(p, 0) for p in param_order]
    ax.barh(y_labels, values, color=colors, edgecolor='white', height=0.6)
    ax.set_xlabel(
        f'Mean Twin XOR Accuracy Drop\nat \u00b1{max_pct}% Perturbation',
        fontsize=13, fontweight='bold'
    )
    ax.set_title(
        f'A. Parameter Sensitivity Ranking',
        fontsize=15, fontweight='bold'
    )
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Right: drop vs perturbation magnitude
    ax = axes[1]
    palette = sns.color_palette("husl", len(param_order))
    for idx, param in enumerate(param_order):
        sub = agg[agg['param'] == param].sort_values('abs_pct')
        ax.plot(sub['abs_pct'], sub['mean_drop'],
                marker='o', linewidth=2, markersize=6,
                color=palette[idx], label=PARAM_LABELS.get(param, param))
        ax.fill_between(
            sub['abs_pct'],
            sub['mean_drop'] - sub['std_drop'],
            sub['mean_drop'] + sub['std_drop'],
            alpha=0.12, color=palette[idx]
        )

    ax.set_xlabel('Perturbation Magnitude (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Twin XOR Accuracy Drop', fontsize=13, fontweight='bold')
    ax.set_title('B. Accuracy Drop vs Perturbation Size',
                 fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    leg = ax.legend(fontsize=11, ncol=2, loc='upper left')
    for text in leg.get_texts():
        text.set_fontweight('bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PERTURBATIONS)
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')

    n_specimens = df['specimen_idx'].nunique()
    plt.suptitle(
        f'Sensitivity of XOR Twin Accuracy to Biophysical Parameter Perturbations (n={n_specimens})',
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        out_path = FIG_DIR / f'fig_sensitivity.{ext}'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
    plt.close()

    # Also print summary table for paper text
    print(f"\nSensitivity summary (n={n_specimens} specimens, top 50% by baseline accuracy):")
    print(f"{'Parameter':<12} {'±5%':<10} {'±10%':<10} {'±20%':<10} {'±30%':<10}")
    print("-" * 52)
    for param in param_order:
        row = []
        for pct in PERTURBATIONS:
            val = agg[(agg['param'] == param) & (agg['abs_pct'] == pct)]['mean_drop']
            row.append(f"{val.values[0]:.4f}" if len(val) > 0 else "N/A")
        print(f"{param:<12} {row[0]:<10} {row[1]:<10} {row[2]:<10} {row[3]:<10}")


if __name__ == "__main__":
    main()
