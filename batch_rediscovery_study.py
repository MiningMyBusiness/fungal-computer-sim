"""Batch rediscovery study: evaluate digital twin pipeline at scale.

This is the headline experiment for the ALIFE 2026 paper. It runs the full
parameter rediscovery pipeline on N specimens with known (but hidden) parameters
and measures how reliably the digital twin supports XOR gate transfer.

Four conditions are compared:
  - oracle:    Use true parameters (upper bound on performance)
  - ml_only:   Use ML prediction directly, no optimization refinement
  - ml_refine: Full pipeline — ML prediction + waveform-matching optimization
  - random:    Random parameters from viable range (lower bound / baseline)

Results are saved to CSV for statistical analysis and paper figures.

Usage:
    python batch_rediscovery_study.py
    python batch_rediscovery_study.py --n-specimens 50 --node-counts 30 50 80
    python batch_rediscovery_study.py --resume
    python batch_rediscovery_study.py --conditions ml_only ml_refine random
    python batch_rediscovery_study.py --viable-ranges optimization_study_results/viable_param_ranges.json
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Import the existing pipeline functions
from realistic_sim import RealisticFungalComputer, optimize_xor_gate
from rediscover_fungal_parameters import (
    load_models,
    create_random_specimen,
    characterize_specimen,
    collect_response_waveforms,
    predict_parameters,
    create_twin,
    refine_parameters_optimization,
    compute_waveform_mismatch,
    test_xor_gate_performance,
    FUNGAL_PARAMS,
    PARAM_BOUNDS,
    OUTPUT_DIR as REDISCOVERY_OUTPUT_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================

OUTPUT_DIR = Path("batch_rediscovery_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Default study parameters
DEFAULT_N_SPECIMENS = 100
DEFAULT_NODE_COUNTS = [30, 50, 80]
DEFAULT_CONDITIONS = ['oracle', 'ml_only', 'ml_refine', 'random']
DEFAULT_MODEL_DIR = Path("ml_models")
DEFAULT_MODEL_TYPE = 'random_forest'
DEFAULT_OPT_METHOD = 'dual_annealing'  # Fast global optimizer
DEFAULT_VIABLE_RANGES_PATH = Path("optimization_study_results/viable_param_ranges.json")

# XOR optimization parameters (used for all conditions)
XOR_N_CALLS = 100
XOR_MINIMIZER = 'gp'

# ==========================================
# Viable Range Loading
# ==========================================

def load_viable_ranges(path: Path) -> Dict:
    """Load viable parameter ranges from define_viable_range.py output.

    Falls back to full PARAM_BOUNDS if file not found.
    """
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        ranges = {}
        for param, info in data['param_ranges'].items():
            ranges[param] = (info['viable_low'], info['viable_high'])
        logger.info(f"Loaded viable ranges from {path.name}")
        for param, (lo, hi) in ranges.items():
            logger.info(f"  {param:10s}: [{lo:.4f}, {hi:.4f}]")
        return ranges
    else:
        logger.warning(
            f"Viable ranges file not found: {path}. "
            "Using full PARAM_BOUNDS. Run define_viable_range.py first for best results."
        )
        return {k: v for k, v in PARAM_BOUNDS.items()}


def sample_from_ranges(ranges: Dict, rng: np.random.RandomState) -> Dict:
    """Sample random parameters from given ranges (uniform, log-uniform for alpha)."""
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
        params['b'] = min(params['a'] + rng.uniform(0.05, 0.15), ranges['b'][1])

    return params

# ==========================================
# Per-Specimen Evaluation
# ==========================================

def evaluate_specimen_condition(
    specimen: RealisticFungalComputer,
    true_params: Dict,
    specimen_waveforms: Dict,
    condition: str,
    models: Dict,
    scaler: object,
    features: Dict,
    viable_ranges: Dict,
    rng: np.random.RandomState,
    opt_method: str = DEFAULT_OPT_METHOD,
) -> Dict:
    """Run one condition on a specimen and return metrics.

    Args:
        specimen: The specimen (true system)
        true_params: Ground-truth biophysical parameters
        specimen_waveforms: Pre-collected waveforms from specimen
        condition: One of 'oracle', 'ml_only', 'ml_refine', 'random'
        models: Loaded ML models
        scaler: Feature scaler (for MLP)
        features: Pre-extracted characterization features
        viable_ranges: Viable parameter ranges for random baseline
        rng: Random state for reproducibility
        opt_method: Optimization method for ml_refine condition

    Returns:
        Dictionary of metrics for this condition
    """
    t0 = time.time()
    result = {'condition': condition}

    try:
        # ---- Determine twin parameters ----
        if condition == 'oracle':
            twin_params = true_params.copy()
            result['param_source'] = 'true'

        elif condition == 'ml_only':
            predicted_params, _ = predict_parameters(features, models, scaler, DEFAULT_MODEL_TYPE)
            twin_params = predicted_params
            result['param_source'] = 'ml_predicted'
            # Record per-parameter prediction errors
            for param in FUNGAL_PARAMS:
                if param in predicted_params and param in true_params:
                    abs_err = abs(predicted_params[param] - true_params[param])
                    rel_err = abs_err / (abs(true_params[param]) + 1e-10) * 100
                    result[f'ml_abs_err_{param}'] = float(abs_err)
                    result[f'ml_rel_err_{param}'] = float(rel_err)

        elif condition == 'ml_refine':
            predicted_params, _ = predict_parameters(features, models, scaler, DEFAULT_MODEL_TYPE)
            # Record ML errors before refinement
            for param in FUNGAL_PARAMS:
                if param in predicted_params and param in true_params:
                    abs_err = abs(predicted_params[param] - true_params[param])
                    rel_err = abs_err / (abs(true_params[param]) + 1e-10) * 100
                    result[f'ml_abs_err_{param}'] = float(abs_err)
                    result[f'ml_rel_err_{param}'] = float(rel_err)

            # Waveform mismatch before refinement
            twin_ml = create_twin(specimen, predicted_params)
            waveforms_ml = collect_response_waveforms(twin_ml)
            result['waveform_mismatch_before'] = float(
                compute_waveform_mismatch(specimen_waveforms, waveforms_ml)
            )

            # Refine
            refined_params, opt_info = refine_parameters_optimization(
                specimen, specimen_waveforms, predicted_params,
                method=opt_method, use_full_bounds=False
            )
            twin_params = refined_params
            result['param_source'] = 'ml_refined'
            result['opt_n_evals'] = opt_info.get('n_evaluations', 0)
            result['opt_success'] = opt_info.get('success', False)
            result['opt_final_mismatch'] = float(opt_info.get('final_mismatch', np.nan))

            # Record refined errors
            for param in FUNGAL_PARAMS:
                if param in refined_params and param in true_params:
                    abs_err = abs(refined_params[param] - true_params[param])
                    rel_err = abs_err / (abs(true_params[param]) + 1e-10) * 100
                    result[f'refined_abs_err_{param}'] = float(abs_err)
                    result[f'refined_rel_err_{param}'] = float(rel_err)

        elif condition == 'random':
            twin_params = sample_from_ranges(viable_ranges, rng)
            result['param_source'] = 'random'

        else:
            raise ValueError(f"Unknown condition: {condition}")

        # ---- Build twin ----
        twin = create_twin(specimen, twin_params)

        # ---- Waveform mismatch (after) ----
        twin_waveforms = collect_response_waveforms(twin)
        waveform_mismatch = compute_waveform_mismatch(specimen_waveforms, twin_waveforms)
        result['waveform_mismatch'] = float(waveform_mismatch)

        # ---- Optimize XOR gate on twin ----
        xor_opt = optimize_xor_gate(
            num_nodes=twin.num_nodes,
            n_calls=XOR_N_CALLS,
            random_state=42,
            minimizer=XOR_MINIMIZER,
            tune_physics=False,
            env=twin,
        )
        xor_params = xor_opt['params']
        result['xor_score_on_twin'] = float(xor_params['score'])

        # ---- Test XOR on twin ----
        twin_xor = test_xor_gate_performance(twin, xor_params)
        result['xor_accuracy_twin'] = float(twin_xor['accuracy'])
        result['xor_separation_twin'] = float(twin_xor['separation'])

        # ---- Transfer: test XOR on specimen ----
        specimen_xor = test_xor_gate_performance(specimen, xor_params)
        result['xor_accuracy_specimen'] = float(specimen_xor['accuracy'])
        result['xor_separation_specimen'] = float(specimen_xor['separation'])

        # ---- Transfer success flags ----
        result['transfer_success_75'] = bool(specimen_xor['accuracy'] >= 0.75)
        result['transfer_success_100'] = bool(specimen_xor['accuracy'] >= 1.0)

        result['success'] = True
        result['error_message'] = None

    except Exception as e:
        logger.error(f"    Condition {condition} failed: {e}")
        result['success'] = False
        result['error_message'] = str(e)

    result['duration_seconds'] = time.time() - t0
    return result


# ==========================================
# Top-level specimen worker (must be picklable)
# ==========================================

def _run_specimen_worker(args):
    """Run all conditions for a single specimen in a worker process.

    Args:
        args: Dict with keys: num_nodes, specimen_idx, specimen_seed,
              conditions, model_dir, model_type, opt_method,
              viable_ranges_path, random_seed

    Returns:
        List of metric dicts (one per condition).
    """
    import warnings
    warnings.filterwarnings('ignore')

    num_nodes      = args['num_nodes']
    specimen_idx   = args['specimen_idx']
    specimen_seed  = args['specimen_seed']
    conditions     = args['conditions']
    model_dir      = Path(args['model_dir'])
    model_type     = args['model_type']
    opt_method     = args['opt_method']
    viable_ranges  = args['viable_ranges']   # already-loaded dict

    # Re-import inside worker (each process has its own interpreter)
    from realistic_sim import RealisticFungalComputer, optimize_xor_gate
    from rediscover_fungal_parameters import (
        load_models, characterize_specimen, collect_response_waveforms,
        predict_parameters, create_twin, refine_parameters_optimization,
        compute_waveform_mismatch, test_xor_gate_performance,
        FUNGAL_PARAMS,
    )

    specimen_rng = np.random.RandomState(specimen_seed + 999)
    cond_results = []

    try:
        specimen = RealisticFungalComputer(num_nodes=num_nodes, random_seed=specimen_seed)
        true_params = sample_from_ranges(viable_ranges, specimen_rng)
        for param, val in true_params.items():
            setattr(specimen, param, val)

        features           = characterize_specimen(specimen)
        specimen_waveforms = collect_response_waveforms(specimen)
        models, scaler     = load_models(model_dir, model_type)

    except Exception as e:
        for cond in conditions:
            cond_results.append({
                'num_nodes': num_nodes,
                'specimen_idx': specimen_idx,
                'specimen_seed': specimen_seed,
                'condition': cond,
                'success': False,
                'error_message': f'Specimen setup failed: {e}',
            })
        return cond_results

    for condition in conditions:
        cond_rng = np.random.RandomState(specimen_seed + hash(condition) % 100000)
        metrics = evaluate_specimen_condition(
            specimen=specimen,
            true_params=true_params,
            specimen_waveforms=specimen_waveforms,
            condition=condition,
            models=models,
            scaler=scaler,
            features=features,
            viable_ranges=viable_ranges,
            rng=cond_rng,
            opt_method=opt_method,
        )
        metrics.update({
            'num_nodes': num_nodes,
            'num_edges': len(specimen.edge_list),
            'network_density': len(specimen.edge_list) / max(1, num_nodes * (num_nodes - 1) / 2),
            'specimen_idx': specimen_idx,
            'specimen_seed': specimen_seed,
        })
        for param in FUNGAL_PARAMS:
            metrics[f'true_{param}'] = true_params.get(param, float('nan'))
        cond_results.append(metrics)

    return cond_results


# ==========================================
# Main Study Loop
# ==========================================

def run_batch_rediscovery(
    n_specimens: int = DEFAULT_N_SPECIMENS,
    node_counts: List[int] = None,
    conditions: List[str] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    model_type: str = DEFAULT_MODEL_TYPE,
    opt_method: str = DEFAULT_OPT_METHOD,
    viable_ranges_path: Path = DEFAULT_VIABLE_RANGES_PATH,
    resume: bool = False,
    random_seed: int = 42,
    n_workers: int = 1,
):
    """Run the batch rediscovery study.

    Args:
        n_specimens: Number of specimens per node count
        node_counts: List of network sizes to test
        conditions: List of conditions to run per specimen
        model_dir: Directory containing trained ML models
        model_type: ML model type ('random_forest' or 'mlp')
        opt_method: Optimization method for ml_refine condition
        viable_ranges_path: Path to viable_param_ranges.json
        resume: Resume from latest checkpoint
        random_seed: Master random seed
        n_workers: Number of parallel worker processes (1 = serial)
    """
    if node_counts is None:
        node_counts = DEFAULT_NODE_COUNTS
    if conditions is None:
        conditions = DEFAULT_CONDITIONS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"batch_rediscovery_{timestamp}.csv"
    checkpoint_file = OUTPUT_DIR / f"checkpoint_{timestamp}.csv"
    config_file = OUTPUT_DIR / f"config_{timestamp}.json"

    # Handle resume
    all_results = []
    completed_keys = set()

    if resume:
        checkpoints = sorted(OUTPUT_DIR.glob("checkpoint_*.csv"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if checkpoints:
            logger.info(f"Resuming from checkpoint: {checkpoints[0].name}")
            prev = pd.read_csv(checkpoints[0])
            all_results = prev.to_dict('records')
            for row in all_results:
                if row.get('success', False):
                    completed_keys.add((row['num_nodes'], row['specimen_idx'], row['condition']))
            timestamp = checkpoints[0].stem.replace('checkpoint_', '')
            results_file = OUTPUT_DIR / f"batch_rediscovery_{timestamp}.csv"
            checkpoint_file = OUTPUT_DIR / f"checkpoint_{timestamp}.csv"
            logger.info(f"Loaded {len(completed_keys)} completed (node, specimen, condition) tuples")
        else:
            logger.info("No checkpoint found. Starting fresh.")

    # Save config
    config = {
        'n_specimens': n_specimens,
        'node_counts': node_counts,
        'conditions': conditions,
        'model_dir': str(model_dir),
        'model_type': model_type,
        'opt_method': opt_method,
        'xor_n_calls': XOR_N_CALLS,
        'viable_ranges_path': str(viable_ranges_path),
        'random_seed': random_seed,
        'timestamp': timestamp,
    }
    if not config_file.exists():
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    # Load resources
    logger.info("=" * 70)
    logger.info("BATCH REDISCOVERY STUDY")
    logger.info("=" * 70)
    logger.info(f"Specimens per node count: {n_specimens}")
    logger.info(f"Node counts: {node_counts}")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Total runs: {n_specimens * len(node_counts) * len(conditions)}")

    viable_ranges = load_viable_ranges(viable_ranges_path)
    # Note: models are loaded inside each worker to avoid pickling issues

    master_rng = np.random.RandomState(random_seed)
    study_start = time.time()

    total_specimens = n_specimens * len(node_counts)
    specimen_count = 0

    # Pre-generate all specimen seeds deterministically
    all_specimen_seeds = {
        (num_nodes, specimen_idx): int(master_rng.randint(0, 1_000_000))
        for num_nodes in node_counts
        for specimen_idx in range(n_specimens)
    }

    # Build pending task list (skip fully-completed specimens)
    pending_tasks = []
    for num_nodes in node_counts:
        for specimen_idx in range(n_specimens):
            specimen_seed = all_specimen_seeds[(num_nodes, specimen_idx)]
            remaining_conditions = [
                c for c in conditions
                if (num_nodes, specimen_idx, c) not in completed_keys
            ]
            if not remaining_conditions:
                continue
            pending_tasks.append({
                'num_nodes': num_nodes,
                'specimen_idx': specimen_idx,
                'specimen_seed': specimen_seed,
                'conditions': remaining_conditions,
                'model_dir': str(model_dir),
                'model_type': model_type,
                'opt_method': opt_method,
                'viable_ranges': viable_ranges,
            })

    logger.info(f"Pending specimens: {len(pending_tasks)} / {total_specimens}")
    logger.info(f"Workers: {n_workers}")

    def _handle_specimen_results(cond_results):
        """Merge a list of per-condition metric dicts into all_results."""
        nonlocal specimen_count
        specimen_count += 1
        for metrics in cond_results:
            num_nodes   = metrics.get('num_nodes')
            specimen_idx = metrics.get('specimen_idx')
            condition   = metrics.get('condition')
            all_results.append(metrics)
            completed_keys.add((num_nodes, specimen_idx, condition))
        # Checkpoint after each specimen
        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
        elapsed = time.time() - study_start
        avg = elapsed / specimen_count if specimen_count else 0
        eta = avg * (total_specimens - specimen_count)
        logger.info(
            f"  Specimen {specimen_count}/{total_specimens} done | "
            f"elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min"
        )
        for m in cond_results:
            if m.get('success'):
                logger.info(
                    f"    [{m['condition']}] XOR twin={m.get('xor_accuracy_twin', 0):.0%} "
                    f"spec={m.get('xor_accuracy_specimen', 0):.0%} "
                    f"transfer100={m.get('transfer_success_100', False)}"
                )

    if n_workers <= 1:
        # ---- Serial execution ----
        for task in pending_tasks:
            cond_results = _run_specimen_worker(task)
            _handle_specimen_results(cond_results)
    else:
        # ---- Parallel execution ----
        logger.info(f"Launching ProcessPoolExecutor with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {executor.submit(_run_specimen_worker, t): t for t in pending_tasks}
            for future in as_completed(future_to_task):
                try:
                    cond_results = future.result()
                except Exception as e:
                    task = future_to_task[future]
                    cond_results = [{
                        'num_nodes': task['num_nodes'],
                        'specimen_idx': task['specimen_idx'],
                        'specimen_seed': task['specimen_seed'],
                        'condition': c,
                        'success': False,
                        'error_message': str(e),
                    } for c in task['conditions']]
                _handle_specimen_results(cond_results)

    # Save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_file, index=False)

    total_time = time.time() - study_start
    logger.info(f"\n{'='*70}")
    logger.info("STUDY COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hr)")
    logger.info(f"Results saved: {results_file}")

    print_summary(results_df)
    generate_figures(results_df, OUTPUT_DIR)

    return results_df


# ==========================================
# Analysis & Figures
# ==========================================

def print_summary(df: pd.DataFrame):
    """Print summary statistics per condition."""
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"{'='*70}")

    successful = df[df['success'] == True]
    conditions = successful['condition'].unique()

    logger.info(f"\n{'Condition':<14} {'N':<6} {'XOR Acc (twin)':<18} "
                f"{'XOR Acc (spec)':<18} {'Transfer ≥75%':<16} {'Transfer 100%':<14}")
    logger.info("-" * 86)

    for cond in ['oracle', 'ml_only', 'ml_refine', 'random']:
        if cond not in conditions:
            continue
        sub = successful[successful['condition'] == cond]
        n = len(sub)
        twin_acc = sub['xor_accuracy_twin'].mean() if 'xor_accuracy_twin' in sub else np.nan
        spec_acc = sub['xor_accuracy_specimen'].mean() if 'xor_accuracy_specimen' in sub else np.nan
        t75 = sub['transfer_success_75'].mean() * 100 if 'transfer_success_75' in sub else np.nan
        t100 = sub['transfer_success_100'].mean() * 100 if 'transfer_success_100' in sub else np.nan
        logger.info(f"{cond:<14} {n:<6} {twin_acc:<18.3f} {spec_acc:<18.3f} "
                    f"{t75:<16.1f} {t100:<14.1f}")


def generate_figures(df: pd.DataFrame, output_dir: Path):
    """Generate paper figures from batch results."""
    sns.set_style("whitegrid")
    successful = df[df['success'] == True]

    if len(successful) == 0:
        logger.warning("No successful results to plot.")
        return

    conditions_order = [c for c in ['oracle', 'ml_only', 'ml_refine', 'random']
                        if c in successful['condition'].unique()]
    palette = {
        'oracle':    '#2ecc71',
        'ml_only':   '#3498db',
        'ml_refine': '#9b59b6',
        'random':    '#e74c3c',
    }

    # ---- Figure 3: XOR Transfer Accuracy ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of transfer success rates
    ax = axes[0]
    transfer_rates = []
    for cond in conditions_order:
        sub = successful[successful['condition'] == cond]
        t75 = sub['transfer_success_75'].mean() * 100 if 'transfer_success_75' in sub.columns else 0
        t100 = sub['transfer_success_100'].mean() * 100 if 'transfer_success_100' in sub.columns else 0
        transfer_rates.append({'condition': cond, 'threshold': '≥75%', 'rate': t75})
        transfer_rates.append({'condition': cond, 'threshold': '100%', 'rate': t100})

    tr_df = pd.DataFrame(transfer_rates)
    x = np.arange(len(conditions_order))
    width = 0.35
    bars75  = [tr_df[(tr_df['condition']==c) & (tr_df['threshold']=='≥75%')]['rate'].values[0]
               for c in conditions_order]
    bars100 = [tr_df[(tr_df['condition']==c) & (tr_df['threshold']=='100%')]['rate'].values[0]
               for c in conditions_order]
    ax.bar(x - width/2, bars75,  width, label='≥75% accuracy',
           color=[palette[c] for c in conditions_order], alpha=0.9)
    ax.bar(x + width/2, bars100, width, label='100% accuracy',
           color=[palette[c] for c in conditions_order], alpha=0.5, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions_order], fontsize=11)
    ax.set_ylabel('Transfer Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('XOR Gate Transfer Success Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: violin of specimen XOR accuracy by condition
    ax = axes[1]
    plot_data = successful[successful['condition'].isin(conditions_order)].copy()
    sns.violinplot(data=plot_data, x='condition', y='xor_accuracy_specimen',
                   order=conditions_order, palette=palette, ax=ax, inner='quartile', cut=0)
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('XOR Accuracy on Specimen', fontsize=12, fontweight='bold')
    ax.set_title('XOR Accuracy Distribution (Specimen)', fontsize=13, fontweight='bold')
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions_order], fontsize=10)

    plt.suptitle('Figure 3: Digital Twin XOR Gate Transfer Performance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig3_path = output_dir / 'fig3_xor_transfer_accuracy.png'
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved Figure 3: {fig3_path}")
    plt.close()

    # ---- Figure 4: Per-parameter recovery error (ML-only vs ML-refine) ----
    ml_conditions = [c for c in ['ml_only', 'ml_refine'] if c in successful['condition'].unique()]
    if ml_conditions:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        for idx, param in enumerate(FUNGAL_PARAMS):
            ax = axes[idx]
            plot_rows = []
            for cond in ml_conditions:
                sub = successful[successful['condition'] == cond]
                prefix = 'ml' if cond == 'ml_only' else 'refined'
                col = f'{prefix}_rel_err_{param}'
                if col in sub.columns:
                    for val in sub[col].dropna():
                        plot_rows.append({'condition': cond, 'rel_error': val})

            if plot_rows:
                err_df = pd.DataFrame(plot_rows)
                sns.boxplot(data=err_df, x='condition', y='rel_error',
                            palette={c: palette[c] for c in ml_conditions}, ax=ax)
                ax.set_title(param, fontsize=11, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Relative Error (%)', fontsize=9)
                ax.set_xticklabels([c.replace('_', '\n') for c in ml_conditions], fontsize=9)

        plt.suptitle('Figure 4: Per-Parameter Recovery Error (ML-only vs ML+Refine)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig4_path = output_dir / 'fig4_parameter_recovery.png'
        plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved Figure 4: {fig4_path}")
        plt.close()

    # ---- Figure 5: Waveform mismatch before/after optimization ----
    if 'ml_refine' in successful['condition'].unique():
        refine_df = successful[successful['condition'] == 'ml_refine'].copy()
        if 'waveform_mismatch_before' in refine_df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            before = refine_df['waveform_mismatch_before'].dropna()
            after  = refine_df['waveform_mismatch'].dropna()

            ax.scatter(before, after, alpha=0.6, color='#9b59b6', s=50, edgecolors='white')
            lim = max(before.max(), after.max()) * 1.05
            ax.plot([0, lim], [0, lim], 'k--', linewidth=1.5, label='No improvement')
            ax.set_xlabel('Waveform Mismatch (ML-only)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Waveform Mismatch (After Optimization)', fontsize=12, fontweight='bold')
            ax.set_title('Figure 5: Waveform Mismatch Before vs After Optimization',
                         fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.grid(True, alpha=0.3)

            pct_improved = (after < before).mean() * 100
            ax.text(0.05, 0.92, f'{pct_improved:.1f}% of specimens improved',
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            fig5_path = output_dir / 'fig5_waveform_mismatch.png'
            plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved Figure 5: {fig5_path}")
            plt.close()


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch digital twin rediscovery study for ALIFE 2026',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Full study (100 specimens × 3 node sizes × 4 conditions)
  python batch_rediscovery_study.py

  # Quick test (10 specimens, 2 node sizes, 2 conditions)
  python batch_rediscovery_study.py --n-specimens 10 --node-counts 30 50 --conditions ml_only ml_refine

  # Resume from checkpoint
  python batch_rediscovery_study.py --resume

  # Use custom viable ranges
  python batch_rediscovery_study.py --viable-ranges optimization_study_results/viable_param_ranges.json
        """
    )
    parser.add_argument('--n-specimens', type=int, default=DEFAULT_N_SPECIMENS,
                        help='Number of specimens per node count')
    parser.add_argument('--node-counts', type=int, nargs='+', default=DEFAULT_NODE_COUNTS,
                        help='Network sizes to test')
    parser.add_argument('--conditions', type=str, nargs='+', default=DEFAULT_CONDITIONS,
                        choices=['oracle', 'ml_only', 'ml_refine', 'random'],
                        help='Conditions to run')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                        help='Directory containing trained ML models')
    parser.add_argument('--model-type', type=str, default=DEFAULT_MODEL_TYPE,
                        choices=['random_forest', 'mlp'],
                        help='ML model type to use')
    parser.add_argument('--opt-method', type=str, default=DEFAULT_OPT_METHOD,
                        help='Optimization method for ml_refine condition')
    parser.add_argument('--viable-ranges', type=Path, default=DEFAULT_VIABLE_RANGES_PATH,
                        help='Path to viable_param_ranges.json')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Master random seed')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel worker processes (default: 1 = serial). '
                             'E.g. --n-workers 4')
    args = parser.parse_args()

    total_runs = args.n_specimens * len(args.node_counts) * len(args.conditions)
    print(f"\n{'='*70}")
    print("BATCH REDISCOVERY STUDY")
    print(f"{'='*70}")
    print(f"Specimens per node count: {args.n_specimens}")
    print(f"Node counts:              {args.node_counts}")
    print(f"Conditions:               {args.conditions}")
    print(f"Total runs:               {total_runs}")
    print(f"Viable ranges:            {args.viable_ranges}")
    print(f"Workers:                  {args.n_workers} ({'parallel' if args.n_workers > 1 else 'serial'})")
    print(f"{'='*70}")

    if args.yes:
        proceed = True
    else:
        response = input("\nProceed? (yes/no): ")
        proceed = response.lower() in ['yes', 'y']

    if proceed:
        results_df = run_batch_rediscovery(
            n_specimens=args.n_specimens,
            node_counts=args.node_counts,
            conditions=args.conditions,
            model_dir=args.model_dir,
            model_type=args.model_type,
            opt_method=args.opt_method,
            viable_ranges_path=args.viable_ranges,
            resume=args.resume,
            random_seed=args.seed,
            n_workers=args.n_workers,
        )
        print(f"\nStudy complete. Results saved to: {OUTPUT_DIR}")
    else:
        print("Study cancelled.")
