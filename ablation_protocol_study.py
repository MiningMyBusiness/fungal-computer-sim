"""Protocol ablation study: which characterization protocols matter most for ML prediction?

This script trains ML models using subsets of the 94 response features and compares
per-parameter RMSE to quantify the contribution of each protocol.

Feature subsets tested:
  - step_only:    13 features (step response)
  - pp_only:      58 features (paired-pulse, all delays + aggregates)
  - tri_only:     23 features (triangle sweep)
  - step_pp:      71 features
  - step_tri:     36 features
  - pp_tri:       81 features
  - all:          94 features (full model)

Usage:
    python ablation_protocol_study.py
    python ablation_protocol_study.py --data-path characterization_study_results/characterization_results_*.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import argparse
import logging
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("ablation_study_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Target parameters to predict
TARGET_PARAMS = ['tau_v', 'tau_w', 'a', 'b', 'v_scale', 'R_off', 'R_on', 'alpha']

# ==========================================
# Feature Subset Definitions
# ==========================================

STEP_FEATURES = [
    'step_baseline', 'step_peak_amplitude', 'step_time_to_peak',
    'step_peak_to_baseline_ratio', 'step_half_decay_time', 'step_decay_rate',
    'step_area_under_curve', 'step_response_duration', 'step_activation_speed',
    'step_latency', 'step_initial_slope', 'step_settling_deviation', 'step_asymmetry_index',
]

PP_FEATURES = []
for delay in [200, 800, 2000]:
    prefix = f'pp_delay_{delay}'
    PP_FEATURES.extend([
        f'{prefix}_peak1_amplitude', f'{prefix}_peak2_amplitude',
        f'{prefix}_time_to_peak1', f'{prefix}_time_to_peak2',
        f'{prefix}_auc1', f'{prefix}_auc2',
        f'{prefix}_peak_width1', f'{prefix}_peak_width2',
        f'{prefix}_peak_ratio', f'{prefix}_auc_ratio',
        f'{prefix}_latency_change', f'{prefix}_width_ratio',
        f'{prefix}_baseline_shift', f'{prefix}_recovery_fraction',
        f'{prefix}_effective_ipi', f'{prefix}_waveform_correlation',
        f'{prefix}_total_response', f'{prefix}_facilitation_index',
    ])
PP_FEATURES.extend([
    'pp_avg_peak_ratio', 'pp_std_peak_ratio',
    'pp_avg_facilitation_index', 'pp_recovery_slope',
])

TRI_FEATURES = [
    'tri_total_hysteresis_area', 'tri_pos_hysteresis', 'tri_neg_hysteresis',
    'tri_max_hysteresis_width', 'tri_response_at_pos_max', 'tri_response_at_neg_max',
    'tri_pos_neg_ratio', 'tri_rectification_index', 'tri_linearity_deviation',
    'tri_slope_variation', 'tri_num_inflection_points', 'tri_response_amplitude',
    'tri_voltage_gain', 'tri_phase1_gain', 'tri_phase2_gain', 'tri_phase3_gain',
    'tri_return_point_deviation', 'tri_loop_closure_error', 'tri_loop_eccentricity',
    'tri_centroid_v_applied', 'tri_centroid_v_response', 'tri_smoothness_index',
    'tri_oscillation_count',
]

FEATURE_SUBSETS = {
    'step_only':  STEP_FEATURES,
    'pp_only':    PP_FEATURES,
    'tri_only':   TRI_FEATURES,
    'step_pp':    STEP_FEATURES + PP_FEATURES,
    'step_tri':   STEP_FEATURES + TRI_FEATURES,
    'pp_tri':     PP_FEATURES + TRI_FEATURES,
    'all':        STEP_FEATURES + PP_FEATURES + TRI_FEATURES,
}

SUBSET_LABELS = {
    'step_only': 'Step\nOnly',
    'pp_only':   'Paired-\nPulse Only',
    'tri_only':  'Triangle\nOnly',
    'step_pp':   'Step +\nPaired-Pulse',
    'step_tri':  'Step +\nTriangle',
    'pp_tri':    'PP +\nTriangle',
    'all':       'All Three\n(Full)',
}

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

# ==========================================
# Data Loading
# ==========================================

def load_characterization_data(data_path: str = None) -> pd.DataFrame:
    """Load characterization results CSV."""
    if data_path:
        path = Path(data_path)
        if path.exists():
            logger.info(f"Loading: {path}")
            return pd.read_csv(path)
        # Try glob
        matches = sorted(glob.glob(str(data_path)), key=lambda p: Path(p).stat().st_mtime, reverse=True)
        if matches:
            logger.info(f"Loading: {matches[0]}")
            return pd.read_csv(matches[0])

    # Auto-detect
    results_dir = Path("characterization_study_results")
    files = sorted(results_dir.glob("characterization_results_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        logger.info(f"Loading: {files[0]}")
        return pd.read_csv(files[0])

    checkpoints = sorted(results_dir.glob("checkpoint_*.csv"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        logger.warning(f"No final results found. Loading checkpoint: {checkpoints[0]}")
        return pd.read_csv(checkpoints[0])

    raise FileNotFoundError(
        "No characterization data found. Run systematic_characterization_study.py first."
    )

# ==========================================
# Model Training & Evaluation
# ==========================================

def train_and_evaluate_subset(X_train, X_test, y_train, y_test,
                               feature_names, subset_name) -> dict:
    """Train RF on a feature subset and evaluate per-parameter RMSE and R²."""
    results = {}

    for param in TARGET_PARAMS:
        if param not in y_train.columns:
            continue

        y_tr = y_train[param].dropna()
        y_te = y_test[param].dropna()
        X_tr = X_train.loc[y_tr.index]
        X_te = X_test.loc[y_te.index]

        if len(y_tr) < 10:
            logger.warning(f"  {param}: insufficient training data ({len(y_tr)} samples)")
            continue

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        r2   = float(r2_score(y_te, preds))

        # Normalised RMSE (as % of parameter range)
        param_range = y_tr.max() - y_tr.min()
        nrmse = rmse / param_range * 100 if param_range > 0 else np.nan

        results[param] = {
            'rmse':  rmse,
            'nrmse': nrmse,
            'r2':    r2,
            'n_train': len(y_tr),
            'n_test':  len(y_te),
        }

    return results


def run_ablation_study(data_path: str = None):
    """Run the full protocol ablation study."""
    logger.info("=" * 70)
    logger.info("PROTOCOL ABLATION STUDY")
    logger.info("=" * 70)

    # Load data
    df = load_characterization_data(data_path)
    successful = df[df['characterization_success'] == True].copy()
    logger.info(f"Successful trials: {len(successful)}/{len(df)}")

    # Prepare targets
    y = successful[TARGET_PARAMS].copy()

    # Train/test split (80/20, stratified by num_nodes)
    train_idx, test_idx = train_test_split(
        successful.index, test_size=0.2, random_state=42,
        stratify=successful['num_nodes'] if 'num_nodes' in successful.columns else None
    )
    y_train = y.loc[train_idx]
    y_test  = y.loc[test_idx]

    # Run each subset
    all_results = {}
    for subset_name, features in FEATURE_SUBSETS.items():
        # Only use features that exist in the data
        available = [f for f in features if f in successful.columns]
        missing = len(features) - len(available)
        if missing > 0:
            logger.warning(f"  {subset_name}: {missing} features missing from data")

        if len(available) == 0:
            logger.error(f"  {subset_name}: no features available, skipping")
            continue

        X = successful[available].fillna(0)
        X_train = X.loc[train_idx]
        X_test  = X.loc[test_idx]

        logger.info(f"\nSubset: {subset_name} ({len(available)} features)")
        results = train_and_evaluate_subset(
            X_train, X_test, y_train, y_test,
            available, subset_name
        )
        all_results[subset_name] = results

        for param, metrics in results.items():
            logger.info(f"  {param:10s}: RMSE={metrics['rmse']:.4f}, "
                        f"NRMSE={metrics['nrmse']:.1f}%, R²={metrics['r2']:.3f}")

    # Build results DataFrames
    rmse_data = {}
    nrmse_data = {}
    r2_data = {}
    for subset_name, param_results in all_results.items():
        rmse_data[subset_name]  = {p: v['rmse']  for p, v in param_results.items()}
        nrmse_data[subset_name] = {p: v['nrmse'] for p, v in param_results.items()}
        r2_data[subset_name]    = {p: v['r2']    for p, v in param_results.items()}

    rmse_df  = pd.DataFrame(rmse_data).T
    nrmse_df = pd.DataFrame(nrmse_data).T
    r2_df    = pd.DataFrame(r2_data).T

    # Save results
    rmse_df.to_csv(OUTPUT_DIR / 'ablation_rmse.csv')
    nrmse_df.to_csv(OUTPUT_DIR / 'ablation_nrmse.csv')
    r2_df.to_csv(OUTPUT_DIR / 'ablation_r2.csv')

    with open(OUTPUT_DIR / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {OUTPUT_DIR}/")

    # Generate Figure 6
    generate_figure6(nrmse_df, r2_df)

    return all_results


# ==========================================
# Figure 6
# ==========================================

def generate_figure6(nrmse_df: pd.DataFrame, r2_df: pd.DataFrame):
    """Figure 6: Heatmap of NRMSE by (parameter × feature subset)."""
    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Rename for display
    nrmse_plot = nrmse_df.copy()
    nrmse_plot.index = [SUBSET_LABELS.get(i, i) for i in nrmse_plot.index]
    nrmse_plot.columns = [PARAM_LABELS.get(c, c) for c in nrmse_plot.columns]

    r2_plot = r2_df.copy()
    r2_plot.index = [SUBSET_LABELS.get(i, i) for i in r2_plot.index]
    r2_plot.columns = [PARAM_LABELS.get(c, c) for c in r2_plot.columns]

    # NRMSE heatmap (lower is better)
    ax = axes[0]
    sns.heatmap(nrmse_plot, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'NRMSE (%)'},
                annot_kws={'size': 10})
    ax.set_title('Normalised RMSE (%) — Lower is Better',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Subset', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=10, rotation=0)

    # R² heatmap (higher is better)
    ax = axes[1]
    sns.heatmap(r2_plot, annot=True, fmt='.2f', cmap='YlGn',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'R²'},
                vmin=0, vmax=1, annot_kws={'size': 10})
    ax.set_title('R² Score — Higher is Better',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Subset', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=10, rotation=0)

    plt.suptitle('Figure 6: Protocol Ablation — Contribution of Each Characterization Protocol',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig6_path = OUTPUT_DIR / 'fig6_protocol_ablation.png'
    plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved Figure 6: {fig6_path}")
    plt.close()


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Protocol ablation study for ALIFE 2026',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ablation_protocol_study.py
  python ablation_protocol_study.py --data-path characterization_study_results/characterization_results_20260101_120000.csv
        """
    )
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to characterization results CSV (auto-detected if not specified)')
    args = parser.parse_args()

    run_ablation_study(data_path=args.data_path)
    print(f"\nAblation study complete. Results saved to: {OUTPUT_DIR}")
