"""Define viable biophysical parameter range from optimization study results.

This script post-processes the output of systematic_optimization_study.py to:
1. Identify high-performing (viable) parameter combinations
2. Compute per-parameter marginal distributions for the viable subspace
3. Output viable_param_ranges.json for use by systematic_characterization_study.py
4. Generate diagnostic figures (Figure 1 for the paper)

Usage:
    python define_viable_range.py
    python define_viable_range.py --results-dir optimization_study_results
    python define_viable_range.py --score-percentile 75  # top 25% by tuned_score
    python define_viable_range.py --score-threshold 0.5  # absolute score threshold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
import json
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================

RESULTS_DIR = Path("optimization_study_results")
OUTPUT_DIR = Path("optimization_study_results")

# Biophysical parameters to analyse (tuned versions from physics tuning)
FUNGAL_PARAMS = ['tau_v', 'tau_w', 'a', 'b', 'v_scale', 'R_off', 'R_on', 'alpha']

# Full parameter ranges (from systematic_optimization_study.py / rediscover_fungal_parameters.py)
FULL_PARAM_RANGES = {
    'tau_v':   (30.0,   150.0),
    'tau_w':   (300.0,  1600.0),
    'a':       (0.5,    0.8),
    'b':       (0.7,    1.0),
    'v_scale': (0.5,    10.0),
    'R_off':   (50.0,   300.0),
    'R_on':    (2.0,    50.0),
    'alpha':   (0.0001, 0.02),
}

# Human-readable parameter labels for plots
PARAM_LABELS = {
    'tau_v':   r'$\tau_v$ (ms)',
    'tau_w':   r'$\tau_w$ (ms)',
    'a':       r'$a$ (FHN)',
    'b':       r'$b$ (FHN)',
    'v_scale': r'$v_{scale}$',
    'R_off':   r'$R_{off}$ ($\Omega$)',
    'R_on':    r'$R_{on}$ ($\Omega$)',
    'alpha':   r'$\alpha$ (memristor)',
}

# ==========================================
# Data Loading
# ==========================================

def load_optimization_results(results_dir: Path) -> pd.DataFrame:
    """Load the most recent optimization results CSV.

    Prefers optimization_results_*.csv over checkpoint_*.csv.
    """
    results_dir = Path(results_dir)

    # Try final results files first
    result_files = sorted(results_dir.glob("optimization_results_*.csv"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    if result_files:
        path = result_files[0]
        logger.info(f"Loading results: {path.name}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path.name}")
        return df

    # Fall back to checkpoint
    checkpoint_files = sorted(results_dir.glob("checkpoint_*.csv"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoint_files:
        path = checkpoint_files[0]
        logger.warning(f"No final results found. Loading checkpoint: {path.name}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from checkpoint")
        return df

    raise FileNotFoundError(
        f"No optimization results found in {results_dir}. "
        "Run systematic_optimization_study.py first."
    )

# ==========================================
# Viable Range Definition
# ==========================================

def identify_viable_trials(df: pd.DataFrame,
                           score_percentile: float = None,
                           score_threshold: float = None) -> pd.DataFrame:
    """Filter to high-performing (viable) trials.

    Uses tuned_score if physics tuning was enabled, otherwise falls back to score.

    Args:
        df: Full results dataframe
        score_percentile: Keep top X% by score (e.g. 75 = top 25%)
        score_threshold: Keep trials with score >= this value

    Returns:
        Filtered dataframe of viable trials
    """
    # Only use successful trials with valid scores
    successful = df[df['success'] == True].copy()
    logger.info(f"Successful trials: {len(successful)}/{len(df)}")

    # Prefer tuned_score if available
    if 'tuned_score' in successful.columns and successful['tuned_score'].notna().sum() > 0:
        score_col = 'tuned_score'
        logger.info("Using tuned_score (physics tuning was enabled)")
    else:
        score_col = 'score'
        logger.info("Using score (no physics tuning data found)")

    scores = successful[score_col].dropna()

    # Determine threshold
    if score_threshold is not None:
        threshold = score_threshold
        logger.info(f"Using absolute score threshold: {threshold:.4f}")
    elif score_percentile is not None:
        threshold = scores.quantile(score_percentile / 100.0)
        logger.info(f"Using p{score_percentile} score threshold: {threshold:.4f}")
    else:
        # Default: top 25% (p75)
        threshold = scores.quantile(0.75)
        logger.info(f"Using default p75 score threshold: {threshold:.4f}")

    viable = successful[successful[score_col] >= threshold].copy()
    logger.info(f"Viable trials: {len(viable)} / {len(successful)} "
                f"(score >= {threshold:.4f})")

    if len(viable) < 5:
        logger.warning(
            f"Only {len(viable)} viable trials found. Consider lowering the threshold "
            "or running more optimization trials."
        )

    return viable, score_col, threshold


def compute_viable_ranges(viable_df: pd.DataFrame) -> dict:
    """Compute per-parameter statistics for the viable subspace.

    For each biophysical parameter, computes:
    - min, max (observed range in viable trials)
    - mean, std (for warm-start sampling)
    - p5, p95 (robust range for characterization study bounds)
    - full_min, full_max (original search bounds, for reference)

    Returns:
        Dictionary suitable for JSON serialisation
    """
    ranges = {}

    for param in FUNGAL_PARAMS:
        tuned_col = f'tuned_{param}'
        if tuned_col in viable_df.columns and viable_df[tuned_col].notna().sum() > 0:
            values = viable_df[tuned_col].dropna()
        elif param in viable_df.columns:
            values = viable_df[param].dropna()
        else:
            logger.warning(f"No data found for parameter {param}, skipping")
            continue

        full_low, full_high = FULL_PARAM_RANGES[param]

        # Compute robust bounds: use p5/p95 but clamp to full range
        p5  = float(np.clip(np.percentile(values, 5),  full_low, full_high))
        p95 = float(np.clip(np.percentile(values, 95), full_low, full_high))

        # Add a small margin (10% of the p5-p95 width) to avoid over-constraining
        margin = 0.10 * (p95 - p5)
        viable_low  = float(np.clip(p5  - margin, full_low, full_high))
        viable_high = float(np.clip(p95 + margin, full_low, full_high))

        ranges[param] = {
            'viable_low':  viable_low,
            'viable_high': viable_high,
            'full_low':    full_low,
            'full_high':   full_high,
            'mean':        float(values.mean()),
            'std':         float(values.std()),
            'median':      float(values.median()),
            'p5':          p5,
            'p95':         p95,
            'n_samples':   int(len(values)),
        }

        logger.info(
            f"  {param:10s}: viable=[{viable_low:.4f}, {viable_high:.4f}]  "
            f"(full=[{full_low:.4f}, {full_high:.4f}], "
            f"mean={values.mean():.4f}, std={values.std():.4f})"
        )

    return ranges


def test_multimodality(viable_df: pd.DataFrame) -> dict:
    """Test whether each parameter's distribution is unimodal or multimodal.

    Uses Hartigan's dip test statistic as a simple multimodality indicator.
    Falls back to a bimodality coefficient if diptest is not available.

    Returns:
        Dictionary of {param: {'bimodality_coeff': float, 'is_multimodal': bool}}
    """
    results = {}

    for param in FUNGAL_PARAMS:
        tuned_col = f'tuned_{param}'
        col = tuned_col if tuned_col in viable_df.columns else param
        values = viable_df[col].dropna().values

        if len(values) < 4:
            results[param] = {'bimodality_coeff': np.nan, 'is_multimodal': False}
            continue

        # Bimodality coefficient: BC = (skewness^2 + 1) / kurtosis
        # BC > 0.555 suggests multimodality (Pfister et al. 2013)
        skew = stats.skew(values)
        kurt = stats.kurtosis(values, fisher=False)  # Pearson kurtosis
        if kurt > 0:
            bc = (skew**2 + 1) / kurt
        else:
            bc = np.nan

        is_multimodal = (bc > 0.555) if not np.isnan(bc) else False
        results[param] = {
            'bimodality_coeff': float(bc) if not np.isnan(bc) else None,
            'is_multimodal': bool(is_multimodal),
        }

        if is_multimodal:
            logger.warning(
                f"  {param}: possible multimodal distribution (BC={bc:.3f}). "
                "Consider stratified sampling in characterization study."
            )

    return results

# ==========================================
# Visualisation
# ==========================================

def plot_viable_range(df: pd.DataFrame, viable_df: pd.DataFrame,
                      score_col: str, threshold: float,
                      output_path: Path):
    """Figure 1: Score distribution + per-parameter violin plots (full vs viable).

    Layout:
    - Top row: score distribution with threshold line
    - Bottom 2 rows: violin plots for each of the 8 parameters
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Top row: score distribution ----
    ax_score = fig.add_subplot(gs[0, :2])
    scores_all = df[df['success'] == True][score_col].dropna()
    ax_score.hist(scores_all, bins=30, color='#4C72B0', alpha=0.7, edgecolor='white',
                  label='All trials')
    ax_score.axvline(threshold, color='#DD4444', linewidth=2.5, linestyle='--',
                     label=f'Viable threshold ({threshold:.3f})')
    ax_score.set_xlabel('XOR Score (tuned)', fontsize=12, fontweight='bold')
    ax_score.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax_score.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax_score.legend(fontsize=10)

    # ---- Top row: score vs num_nodes ----
    ax_nodes = fig.add_subplot(gs[0, 2:])
    node_counts = sorted(df['num_nodes'].unique())
    colors_nodes = sns.color_palette("husl", len(node_counts))
    for nc, color in zip(node_counts, colors_nodes):
        subset = df[(df['success'] == True) & (df['num_nodes'] == nc)][score_col].dropna()
        ax_nodes.scatter([nc] * len(subset), subset, color=color, alpha=0.6, s=40,
                         label=f'{nc} nodes')
    ax_nodes.axhline(threshold, color='#DD4444', linewidth=2, linestyle='--',
                     label='Viable threshold')
    ax_nodes.set_xlabel('Network Size (nodes)', fontsize=12, fontweight='bold')
    ax_nodes.set_ylabel('XOR Score (tuned)', fontsize=12, fontweight='bold')
    ax_nodes.set_title('Score by Network Size', fontsize=13, fontweight='bold')
    ax_nodes.legend(fontsize=8, ncol=2)

    # ---- Parameter violin plots ----
    for idx, param in enumerate(FUNGAL_PARAMS):
        row = 1 + idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])

        tuned_col = f'tuned_{param}'
        col_name = tuned_col if tuned_col in df.columns else param

        all_vals = df[df['success'] == True][col_name].dropna()
        viable_vals = viable_df[col_name].dropna()

        # Use log scale for alpha
        if param == 'alpha':
            all_vals = np.log10(all_vals)
            viable_vals = np.log10(viable_vals)
            xlabel = f'log₁₀({PARAM_LABELS[param]})'
        else:
            xlabel = PARAM_LABELS[param]

        plot_data = pd.DataFrame({
            'value': pd.concat([all_vals, viable_vals], ignore_index=True),
            'group': ['All'] * len(all_vals) + ['Viable'] * len(viable_vals)
        })

        sns.violinplot(data=plot_data, x='group', y='value', ax=ax,
                       palette={'All': '#4C72B0', 'Viable': '#55A868'},
                       inner='quartile', cut=0)
        ax.set_title(PARAM_LABELS[param], fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Value', fontsize=9)
        ax.tick_params(axis='x', labelsize=10)

    plt.suptitle('Viable Biophysical Parameter Range for XOR Computation\n'
                 f'(Viable = top {100*(1 - threshold/scores_all.max()):.0f}% by tuned XOR score)',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved Figure 1: {output_path}")
    plt.close()


def plot_score_vs_params(viable_df: pd.DataFrame, score_col: str, output_path: Path):
    """Scatter plots of each parameter vs XOR score for viable trials."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, param in enumerate(FUNGAL_PARAMS):
        ax = axes[idx]
        tuned_col = f'tuned_{param}'
        col_name = tuned_col if tuned_col in viable_df.columns else param

        x = viable_df[col_name].dropna()
        y = viable_df.loc[x.index, score_col]

        if param == 'alpha':
            x = np.log10(x)
            xlabel = f'log₁₀({PARAM_LABELS[param]})'
        else:
            xlabel = PARAM_LABELS[param]

        ax.scatter(x, y, alpha=0.6, color='#55A868', edgecolors='white', s=50)

        # Fit and plot trend line
        if len(x) >= 3:
            try:
                slope, intercept, r, p, _ = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=1.5,
                        label=f'r={r:.2f}, p={p:.3f}')
                ax.legend(fontsize=8)
            except Exception:
                pass

        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
        ax.set_ylabel('XOR Score', fontsize=10, fontweight='bold')
        ax.set_title(PARAM_LABELS[param], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Parameter vs XOR Score (Viable Trials)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved parameter-score scatter: {output_path}")
    plt.close()

# ==========================================
# Main
# ==========================================

def define_viable_range(results_dir: Path = RESULTS_DIR,
                        output_dir: Path = OUTPUT_DIR,
                        score_percentile: float = None,
                        score_threshold: float = None):
    """Main function: load results, define viable range, save JSON + figures.

    Args:
        results_dir: Directory containing optimization_results_*.csv
        output_dir: Directory to write viable_param_ranges.json and figures
        score_percentile: Keep top (100 - score_percentile)% by score
        score_threshold: Absolute score threshold (overrides score_percentile)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("DEFINING VIABLE PARAMETER RANGE")
    logger.info("=" * 70)

    # Load data
    df = load_optimization_results(results_dir)

    # Identify viable trials
    viable_df, score_col, threshold = identify_viable_trials(
        df,
        score_percentile=score_percentile,
        score_threshold=score_threshold
    )

    # Compute ranges
    logger.info("\nComputing viable parameter ranges:")
    ranges = compute_viable_ranges(viable_df)

    # Test for multimodality
    logger.info("\nTesting for multimodal distributions:")
    multimodality = test_multimodality(viable_df)
    for param, result in multimodality.items():
        bc = result['bimodality_coeff']
        flag = " *** MULTIMODAL ***" if result['is_multimodal'] else ""
        bc_str = f"{bc:.3f}" if bc is not None else "N/A"
        logger.info(f"  {param:10s}: BC={bc_str}{flag}")

    # Build output JSON
    output = {
        'metadata': {
            'source_file': str(sorted(Path(results_dir).glob("optimization_results_*.csv"),
                                      key=lambda p: p.stat().st_mtime, reverse=True)[0].name)
                           if list(Path(results_dir).glob("optimization_results_*.csv"))
                           else 'checkpoint',
            'n_total_trials': int(len(df[df['success'] == True])),
            'n_viable_trials': int(len(viable_df)),
            'score_column': score_col,
            'score_threshold': float(threshold),
            'viable_fraction': float(len(viable_df) / max(1, len(df[df['success'] == True]))),
        },
        'param_ranges': ranges,
        'multimodality': multimodality,
    }

    json_path = output_dir / 'viable_param_ranges.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nViable parameter ranges saved: {json_path}")

    # Generate figures
    logger.info("\nGenerating figures...")
    plot_viable_range(
        df, viable_df, score_col, threshold,
        output_dir / 'fig1_viable_parameter_range.png'
    )
    plot_score_vs_params(
        viable_df, score_col,
        output_dir / 'fig1b_score_vs_params.png'
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("VIABLE RANGE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Parameter':<12} {'Viable Low':<14} {'Viable High':<14} "
                f"{'Full Low':<12} {'Full High':<12} {'Coverage %':<12}")
    logger.info("-" * 76)
    for param, r in ranges.items():
        coverage = 100 * (r['viable_high'] - r['viable_low']) / (r['full_high'] - r['full_low'])
        logger.info(f"{param:<12} {r['viable_low']:<14.4f} {r['viable_high']:<14.4f} "
                    f"{r['full_low']:<12.4f} {r['full_high']:<12.4f} {coverage:<12.1f}")

    logger.info(f"\nOutput: {json_path}")
    logger.info("Next step: update systematic_characterization_study.py to load "
                "viable_param_ranges.json instead of hardcoded PARAM_RANGES")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Define viable biophysical parameter range from optimization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Use default top-25% threshold
  python define_viable_range.py

  # Use top-10% threshold (more selective)
  python define_viable_range.py --score-percentile 90

  # Use absolute score threshold
  python define_viable_range.py --score-threshold 0.5

  # Specify custom results directory
  python define_viable_range.py --results-dir my_optimization_results
        """
    )
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR,
                        help='Directory containing optimization results CSV')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help='Directory to write viable_param_ranges.json and figures')
    parser.add_argument('--score-percentile', type=float, default=None,
                        help='Score percentile threshold (e.g. 75 = top 25%%)')
    parser.add_argument('--score-threshold', type=float, default=None,
                        help='Absolute score threshold (overrides --score-percentile)')
    args = parser.parse_args()

    define_viable_range(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        score_percentile=args.score_percentile,
        score_threshold=args.score_threshold,
    )
