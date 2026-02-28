"""Sensitivity analysis: how does parameter estimation error affect XOR gate performance?

This script quantifies how sensitive XOR gate performance is to errors in each
biophysical parameter. For a set of high-scoring specimens, it perturbs each
parameter independently and measures the resulting drop in XOR accuracy.

This motivates the optimization refinement step in the digital twin pipeline
by showing that ML prediction errors (~10-20% RMSE) are large enough to
meaningfully hurt XOR transfer.

Usage:
    python sensitivity_analysis.py
    python sensitivity_analysis.py --n-specimens 20 --perturbations 5 10 20 30
    python sensitivity_analysis.py --results-dir optimization_study_results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import json
import argparse
import logging
import time
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from realistic_sim import RealisticFungalComputer, optimize_xor_gate
from rediscover_fungal_parameters import (
    test_xor_gate_performance,
    FUNGAL_PARAMS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("sensitivity_analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# Configuration
# ==========================================

DEFAULT_N_SPECIMENS = 20
DEFAULT_PERTURBATIONS = [5, 10, 20, 30]   # % perturbation of true value
DEFAULT_OPT_N_CALLS = 60                  # XOR optimization calls per specimen
DEFAULT_VIABLE_RANGES_PATH = Path("optimization_study_results/viable_param_ranges.json")

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

# ==========================================
# Helpers
# ==========================================

def load_viable_ranges(path: Path) -> Dict:
    """Load viable parameter ranges or fall back to broad defaults."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return {p: (info['viable_low'], info['viable_high'])
                for p, info in data['param_ranges'].items()}
    logger.warning(f"Viable ranges not found at {path}. Using broad defaults.")
    return {
        'tau_v':   (30.0,   150.0),
        'tau_w':   (300.0,  1600.0),
        'a':       (0.5,    0.8),
        'b':       (0.7,    1.0),
        'v_scale': (0.5,    10.0),
        'R_off':   (50.0,   300.0),
        'R_on':    (2.0,    50.0),
        'alpha':   (0.0001, 0.02),
    }


def sample_params(ranges: Dict, rng: np.random.RandomState) -> Dict:
    """Sample parameters from viable ranges."""
    params = {}
    for param, (low, high) in ranges.items():
        if param == 'alpha':
            params[param] = 10 ** rng.uniform(np.log10(low), np.log10(high))
        else:
            params[param] = rng.uniform(low, high)
    # Safety constraints
    if params['R_off'] < 1.5 * params['R_on']:
        params['R_off'] = min(1.5 * params['R_on'], ranges['R_off'][1])
    if params['b'] < params['a']:
        params['b'] = min(params['a'] + 0.1, ranges['b'][1])
    return params


def apply_params(env: RealisticFungalComputer, params: Dict):
    """Apply parameter dict to environment."""
    for param, val in params.items():
        if hasattr(env, param):
            setattr(env, param, val)


def clamp_param(param: str, value: float, ranges: Dict) -> float:
    """Clamp a perturbed parameter to its valid range."""
    if param not in ranges:
        return value
    lo, hi = ranges[param]
    return float(np.clip(value, lo, hi))


def perturb_params(true_params: Dict, param_to_perturb: str,
                   pct: float, direction: str,
                   ranges: Dict) -> Dict:
    """Return a copy of params with one parameter perturbed by pct%.

    Args:
        true_params: Ground-truth parameters
        param_to_perturb: Which parameter to perturb
        pct: Perturbation magnitude (%)
        direction: 'up' or 'down'
        ranges: Valid ranges for clamping
    """
    perturbed = true_params.copy()
    val = true_params[param_to_perturb]
    delta = val * (pct / 100.0)
    if direction == 'up':
        perturbed[param_to_perturb] = clamp_param(param_to_perturb, val + delta, ranges)
    else:
        perturbed[param_to_perturb] = clamp_param(param_to_perturb, val - delta, ranges)
    return perturbed


# ==========================================
# Core Experiment
# ==========================================

def evaluate_xor_with_params(
    specimen: RealisticFungalComputer,
    params: Dict,
    xor_params: Dict,
) -> Dict:
    """Build a twin with given params and test XOR transfer.

    Args:
        specimen: Original specimen (for XOR transfer test)
        params: Parameters to use for the twin
        xor_params: Pre-optimized XOR gate parameters (electrode positions, stimulus)

    Returns:
        Dict with xor_accuracy_twin, xor_accuracy_specimen
    """
    # Build twin
    twin = RealisticFungalComputer(
        num_nodes=specimen.num_nodes,
        random_seed=specimen._random_seed if hasattr(specimen, '_random_seed') else 42
    )
    apply_params(twin, params)

    # Test XOR on twin
    twin_xor = test_xor_gate_performance(twin, xor_params)
    # Test XOR on specimen (transfer)
    specimen_xor = test_xor_gate_performance(specimen, xor_params)

    return {
        'xor_accuracy_twin':     float(twin_xor['accuracy']),
        'xor_accuracy_specimen': float(specimen_xor['accuracy']),
    }


def run_sensitivity_analysis(
    n_specimens: int = DEFAULT_N_SPECIMENS,
    perturbations: List[float] = None,
    viable_ranges_path: Path = DEFAULT_VIABLE_RANGES_PATH,
    xor_n_calls: int = DEFAULT_OPT_N_CALLS,
    random_seed: int = 42,
):
    """Run the full sensitivity analysis.

    For each specimen:
    1. Sample true parameters from viable range
    2. Create specimen and optimize XOR gate on it (baseline)
    3. For each parameter and each perturbation level:
       - Build twin with perturbed params
       - Test XOR transfer accuracy
    4. Record XOR accuracy drop vs perturbation magnitude
    """
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    logger.info("=" * 70)
    logger.info("PARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Specimens: {n_specimens}")
    logger.info(f"Parameters: {FUNGAL_PARAMS}")
    logger.info(f"Perturbations: {perturbations}%")

    viable_ranges = load_viable_ranges(viable_ranges_path)
    master_rng = np.random.RandomState(random_seed)
    all_results = []
    study_start = time.time()

    for specimen_idx in range(n_specimens):
        seed = int(master_rng.randint(0, 1_000_000))
        rng = np.random.RandomState(seed)

        logger.info(f"\nSpecimen {specimen_idx+1}/{n_specimens} (seed={seed})")

        # Sample parameters and create specimen
        true_params = sample_params(viable_ranges, rng)
        num_nodes = int(rng.choice([30, 50, 80]))

        try:
            specimen = RealisticFungalComputer(num_nodes=num_nodes, random_seed=seed)
            apply_params(specimen, true_params)

            # Optimize XOR gate on specimen with true params (baseline)
            logger.info(f"  Optimizing XOR gate (num_nodes={num_nodes})...")
            xor_opt = optimize_xor_gate(
                num_nodes=num_nodes,
                n_calls=xor_n_calls,
                random_state=seed,
                minimizer='gp',
                tune_physics=False,
                env=specimen,
            )
            xor_params = xor_opt['params']
            baseline_score = xor_params['score']
            logger.info(f"  Baseline XOR score: {baseline_score:.4f}")

            # Baseline accuracy (true params, no perturbation)
            baseline = evaluate_xor_with_params(specimen, true_params, xor_params)
            baseline_accuracy = baseline['xor_accuracy_specimen']

            # Record baseline
            all_results.append({
                'specimen_idx': specimen_idx,
                'specimen_seed': seed,
                'num_nodes': num_nodes,
                'perturbed_param': 'none',
                'perturbation_pct': 0.0,
                'direction': 'none',
                'xor_accuracy_twin': baseline['xor_accuracy_twin'],
                'xor_accuracy_specimen': baseline_accuracy,
                'accuracy_drop': 0.0,
                'baseline_accuracy': baseline_accuracy,
                'baseline_xor_score': baseline_score,
            })

            # Perturb each parameter
            for param in FUNGAL_PARAMS:
                for pct in perturbations:
                    for direction in ['up', 'down']:
                        perturbed = perturb_params(true_params, param, pct, direction, viable_ranges)
                        result = evaluate_xor_with_params(specimen, perturbed, xor_params)

                        accuracy_drop = baseline_accuracy - result['xor_accuracy_specimen']

                        all_results.append({
                            'specimen_idx': specimen_idx,
                            'specimen_seed': seed,
                            'num_nodes': num_nodes,
                            'perturbed_param': param,
                            'perturbation_pct': pct * (1 if direction == 'up' else -1),
                            'direction': direction,
                            'xor_accuracy_twin': result['xor_accuracy_twin'],
                            'xor_accuracy_specimen': result['xor_accuracy_specimen'],
                            'accuracy_drop': float(accuracy_drop),
                            'baseline_accuracy': baseline_accuracy,
                            'baseline_xor_score': baseline_score,
                        })

            logger.info(f"  Completed {len(FUNGAL_PARAMS) * len(perturbations) * 2} perturbations")

        except Exception as e:
            logger.error(f"  Specimen {specimen_idx} failed: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = OUTPUT_DIR / 'sensitivity_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved: {results_path}")

    total_time = time.time() - study_start
    logger.info(f"Total time: {total_time/60:.1f} min")

    # Generate Figure 7
    generate_figure7(results_df, perturbations)

    return results_df


# ==========================================
# Figure 7: Tornado Chart
# ==========================================

def generate_figure7(results_df: pd.DataFrame, perturbations: List[float]):
    """Figure 7: Tornado chart of XOR accuracy drop vs parameter perturbation."""
    sns.set_style("whitegrid")

    # Compute mean accuracy drop per (param, perturbation_pct)
    perturbed = results_df[results_df['perturbed_param'] != 'none'].copy()
    perturbed['abs_pct'] = perturbed['perturbation_pct'].abs()

    # Aggregate: mean accuracy drop per param per |perturbation|
    agg = perturbed.groupby(['perturbed_param', 'abs_pct'])['accuracy_drop'].agg(
        ['mean', 'std']
    ).reset_index()
    agg.columns = ['param', 'abs_pct', 'mean_drop', 'std_drop']

    # ---- Figure 7a: Tornado chart (sorted by impact at max perturbation) ----
    max_pct = max(perturbations)
    max_impact = agg[agg['abs_pct'] == max_pct].set_index('param')['mean_drop']
    param_order = max_impact.sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: tornado bar chart at max perturbation
    ax = axes[0]
    colors = sns.color_palette("RdYlGn_r", len(param_order))
    bars = ax.barh(
        [PARAM_LABELS.get(p, p) for p in param_order],
        [max_impact.get(p, 0) for p in param_order],
        color=colors, edgecolor='white', height=0.6
    )
    ax.set_xlabel(f'Mean XOR Accuracy Drop at ±{max_pct}% Perturbation',
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Parameter Sensitivity\n(±{max_pct}% perturbation)',
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Right: accuracy drop vs perturbation magnitude (line plot per param)
    ax = axes[1]
    palette = sns.color_palette("husl", len(FUNGAL_PARAMS))
    for idx, param in enumerate(param_order):
        sub = agg[agg['param'] == param].sort_values('abs_pct')
        ax.plot(sub['abs_pct'], sub['mean_drop'],
                marker='o', linewidth=2, markersize=6,
                color=palette[idx], label=PARAM_LABELS.get(param, param))
        ax.fill_between(
            sub['abs_pct'],
            sub['mean_drop'] - sub['std_drop'],
            sub['mean_drop'] + sub['std_drop'],
            alpha=0.1, color=palette[idx]
        )

    ax.set_xlabel('Perturbation Magnitude (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean XOR Accuracy Drop', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Drop vs Perturbation Magnitude',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(perturbations)

    plt.suptitle('Figure 7: XOR Gate Sensitivity to Biophysical Parameter Errors',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig7_path = OUTPUT_DIR / 'fig7_sensitivity_analysis.png'
    plt.savefig(fig7_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved Figure 7: {fig7_path}")
    plt.close()

    # Print summary table
    logger.info("\nSENSITIVITY SUMMARY (mean accuracy drop at max perturbation):")
    logger.info(f"{'Parameter':<12} {'Drop at ±5%':<14} {'Drop at ±10%':<14} "
                f"{'Drop at ±20%':<14} {'Drop at ±30%':<14}")
    logger.info("-" * 58)
    for param in param_order:
        row = []
        for pct in [5, 10, 20, 30]:
            val = agg[(agg['param'] == param) & (agg['abs_pct'] == pct)]['mean_drop']
            row.append(f"{val.values[0]:.3f}" if len(val) > 0 else "N/A")
        logger.info(f"{param:<12} {row[0]:<14} {row[1]:<14} {row[2]:<14} {row[3]:<14}")


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parameter sensitivity analysis for ALIFE 2026',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python sensitivity_analysis.py
  python sensitivity_analysis.py --n-specimens 10 --perturbations 5 10 20
  python sensitivity_analysis.py --viable-ranges optimization_study_results/viable_param_ranges.json
        """
    )
    parser.add_argument('--n-specimens', type=int, default=DEFAULT_N_SPECIMENS,
                        help='Number of specimens to test')
    parser.add_argument('--perturbations', type=float, nargs='+',
                        default=DEFAULT_PERTURBATIONS,
                        help='Perturbation magnitudes in %% (e.g. 5 10 20 30)')
    parser.add_argument('--viable-ranges', type=Path, default=DEFAULT_VIABLE_RANGES_PATH,
                        help='Path to viable_param_ranges.json')
    parser.add_argument('--xor-n-calls', type=int, default=DEFAULT_OPT_N_CALLS,
                        help='XOR optimization calls per specimen')
    parser.add_argument('--seed', type=int, default=42,
                        help='Master random seed')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Specimens:     {args.n_specimens}")
    print(f"Perturbations: {args.perturbations}%")
    print(f"Parameters:    {FUNGAL_PARAMS}")
    print(f"Total runs:    {args.n_specimens * len(FUNGAL_PARAMS) * len(args.perturbations) * 2 + args.n_specimens}")
    print(f"{'='*70}")

    results_df = run_sensitivity_analysis(
        n_specimens=args.n_specimens,
        perturbations=args.perturbations,
        viable_ranges_path=args.viable_ranges,
        xor_n_calls=args.xor_n_calls,
        random_seed=args.seed,
    )
    print(f"\nSensitivity analysis complete. Results saved to: {OUTPUT_DIR}")
