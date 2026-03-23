"""Re-derive sensitivity metric from existing sensitivity_results.csv.

Instead of accuracy_drop = baseline_specimen - perturbed_specimen (which is
always ~0), compute:

    twin_accuracy_drop = baseline_twin_accuracy - perturbed_twin_accuracy

where baseline_twin_accuracy is the twin's XOR accuracy when parameterized with
the TRUE params (i.e. the 'none' / perturbation_pct=0 row for that specimen).

Also lets you filter to only the top N% of specimens by baseline_accuracy,
without needing to re-run the full simulation.

Usage:
    python reanalyze_sensitivity.py
    python reanalyze_sensitivity.py --top-pct 50          # keep top 50% by baseline accuracy (default)
    python reanalyze_sensitivity.py --top-pct 25          # keep top 25% (stricter)
    python reanalyze_sensitivity.py --min-baseline 0.75   # keep specimens with baseline >= 0.75
    python reanalyze_sensitivity.py --top-pct 50 --no-filter  # use all specimens
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

INPUT_CSV  = Path("sensitivity_analysis_results/sensitivity_results.csv")
OUTPUT_DIR = Path("sensitivity_analysis_results")

PARAM_LABELS = {
    'tau_v':   r'$\tau_v$ (ms)',
    'tau_w':   r'$\tau_w$ (ms)',
    'a':       r'$a$',
    'b':       r'$b$',
    'v_scale': r'$v_{scale}$',
    'R_off':   r'$R_{off}$',
    'R_on':    r'$R_{on}$',
    'alpha':   r'$\alpha$',
}

DEFAULT_PERTURBATIONS = [5, 10, 20, 30]


def load_and_annotate(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and add a baseline_twin_accuracy column per specimen."""
    df = pd.read_csv(csv_path)

    # Baseline rows: perturbation_pct == 0
    baselines = (
        df[df['perturbed_param'] == 'none']
        [['specimen_idx', 'xor_accuracy_twin']]
        .rename(columns={'xor_accuracy_twin': 'baseline_twin_accuracy'})
    )

    df = df.merge(baselines, on='specimen_idx', how='left')

    # Re-derive the correct drop metric
    df['twin_accuracy_drop'] = df['baseline_twin_accuracy'] - df['xor_accuracy_twin']

    return df


def filter_specimens(df: pd.DataFrame,
                     top_pct: float = None,
                     min_baseline: float = None) -> pd.DataFrame:
    """Filter to high-performing specimens without re-running the simulation.

    Args:
        top_pct:      Keep only specimens in the top N% by baseline_accuracy.
                      E.g. 50 → keep the best 50%.
        min_baseline: Keep only specimens with baseline_accuracy >= this value.

    Either or both filters can be applied. If neither is given, all specimens
    are returned.
    """
    specimen_baselines = (
        df[df['perturbed_param'] == 'none']
        .set_index('specimen_idx')['baseline_accuracy']
    )

    keep_mask = pd.Series(True, index=specimen_baselines.index)

    if top_pct is not None:
        threshold = specimen_baselines.quantile(1.0 - top_pct / 100.0)
        keep_mask &= specimen_baselines >= threshold
        print(f"  top_pct={top_pct}% → baseline >= {threshold:.3f} "
              f"→ {keep_mask.sum()} / {len(specimen_baselines)} specimens kept")

    if min_baseline is not None:
        keep_mask &= specimen_baselines >= min_baseline
        print(f"  min_baseline={min_baseline} → {keep_mask.sum()} / "
              f"{len(specimen_baselines)} specimens kept")

    kept_ids = specimen_baselines[keep_mask].index
    filtered = df[df['specimen_idx'].isin(kept_ids)].copy()
    return filtered


def generate_figure(df: pd.DataFrame,
                    perturbations: list,
                    tag: str = "") -> None:
    """Regenerate Figure 7 using the corrected twin_accuracy_drop metric."""
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

    max_pct = max(perturbations)
    max_impact = agg[agg['abs_pct'] == max_pct].set_index('param')['mean_drop']
    param_order = max_impact.sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: tornado bar chart
    ax = axes[0]
    colors = sns.color_palette("RdYlGn_r", len(param_order))
    ax.barh(
        [PARAM_LABELS.get(p, p) for p in param_order],
        [max_impact.get(p, 0) for p in param_order],
        color=colors, edgecolor='white', height=0.6
    )
    ax.set_xlabel(
        f'Mean Twin XOR Accuracy Drop at ±{max_pct}% Perturbation',
        fontsize=12, fontweight='bold'
    )
    ax.set_title(
        f'Parameter Sensitivity (Twin)\n(±{max_pct}% perturbation)',
        fontsize=13, fontweight='bold'
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Right: drop vs perturbation magnitude
    ax = axes[1]
    palette = sns.color_palette("husl", len(PARAM_LABELS))
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

    ax.set_xlabel('Perturbation Magnitude (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Twin XOR Accuracy Drop', fontsize=12, fontweight='bold')
    ax.set_title('Twin Accuracy Drop vs Perturbation Magnitude',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(perturbations)
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')

    n_specimens = df['specimen_idx'].nunique()
    plt.suptitle(
        f'Figure 7 (Re-derived): XOR Twin Sensitivity to Biophysical Parameter Errors\n'
        f'(n={n_specimens} specimens, metric = baseline_twin_acc − perturbed_twin_acc)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    suffix = f"_{tag}" if tag else ""
    out_path = OUTPUT_DIR / f'fig7_sensitivity_twin{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()

    # Print summary table
    print(f"\n{'TWIN SENSITIVITY SUMMARY':=<58}")
    print(f"(mean twin_accuracy_drop at each perturbation level)")
    print(f"{'Parameter':<12} {'±5%':<12} {'±10%':<12} {'±20%':<12} {'±30%':<12}")
    print("-" * 58)
    for param in param_order:
        row = []
        for pct in perturbations:
            val = agg[(agg['param'] == param) & (agg['abs_pct'] == pct)]['mean_drop']
            row.append(f"{val.values[0]:.4f}" if len(val) > 0 else "N/A")
        print(f"{param:<12} {row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=INPUT_CSV)
    parser.add_argument('--top-pct', type=float, default=50,
                        help='Keep top N%% of specimens by baseline_accuracy (default 50)')
    parser.add_argument('--min-baseline', type=float, default=None,
                        help='Keep specimens with baseline_accuracy >= this floor')
    parser.add_argument('--no-filter', action='store_true',
                        help='Use all specimens (ignore --top-pct and --min-baseline)')
    parser.add_argument('--perturbations', type=float, nargs='+',
                        default=DEFAULT_PERTURBATIONS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("SENSITIVITY RE-ANALYSIS (corrected twin metric)")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")

    df = load_and_annotate(args.input)
    n_total = df['specimen_idx'].nunique()
    print(f"Total specimens loaded: {n_total}")

    # Print baseline accuracy distribution
    baselines = df[df['perturbed_param'] == 'none']['baseline_accuracy']
    print(f"\nBaseline accuracy distribution (n={n_total} specimens):")
    print(f"  min={baselines.min():.2f}  25%={baselines.quantile(.25):.2f}  "
          f"median={baselines.median():.2f}  75%={baselines.quantile(.75):.2f}  "
          f"max={baselines.max():.2f}")

    if args.no_filter:
        filtered = df
        tag = "all"
        print(f"\nNo filtering applied. Using all {n_total} specimens.")
    else:
        print("\nFiltering specimens:")
        filtered = filter_specimens(df,
                                    top_pct=args.top_pct if not args.no_filter else None,
                                    min_baseline=args.min_baseline)
        parts = []
        if args.top_pct is not None and not args.no_filter:
            parts.append(f"top{int(args.top_pct)}pct")
        if args.min_baseline is not None:
            parts.append(f"minb{args.min_baseline}")
        tag = "_".join(parts) if parts else "filtered"

    n_kept = filtered['specimen_idx'].nunique()
    print(f"\nProceeding with {n_kept} specimens for analysis.")

    # Save filtered CSV
    out_csv = OUTPUT_DIR / f'sensitivity_results_twin_{tag}.csv'
    filtered.to_csv(out_csv, index=False)
    print(f"Filtered data saved: {out_csv}")

    generate_figure(filtered, args.perturbations, tag=tag)

    print(f"\n{'='*60}")
    print("Re-analysis complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
