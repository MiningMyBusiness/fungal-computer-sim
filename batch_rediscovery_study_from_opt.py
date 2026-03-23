"""Batch rediscovery study (from pre-optimized specimens).

This is a variant of batch_rediscovery_study.py that sources its test
population exclusively from the finetuned specimens produced by
systematic_optimization_study.py, rather than generating random specimens
on the fly.

Only specimens that achieved a successful XOR gate (i.e. have a valid
tuned_score) are included.  A user-defined percentile threshold further
restricts the pool to the top-scoring fraction — analogous to the
--score-percentile filter in sensitivity_analysis.py.

Four conditions are compared:
  - oracle:    Use true (finetuned) parameters directly — upper bound
  - ml_only:   Use ML prediction directly, no optimization refinement
  - ml_refine: Full pipeline — ML prediction + waveform-matching optimisation
  - random:    Random parameters sampled from viable range — lower bound

Results are saved to CSV for statistical analysis and paper figures.

Usage:
    # Auto-detect latest optimization CSV, keep top 50% by tuned_score
    python batch_rediscovery_study_from_opt.py

    # Specify an opt-results CSV explicitly
    python batch_rediscovery_study_from_opt.py \\
        --opt-results optimization_study_results/optimization_results_<ts>.csv

    # Keep only top 25% (most successful specimens)
    python batch_rediscovery_study_from_opt.py --score-percentile 75

    # Cap the number of specimens and use 4 parallel workers
    python batch_rediscovery_study_from_opt.py --n-specimens 50 --n-workers 4

    # Resume an interrupted run
    python batch_rediscovery_study_from_opt.py --resume
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the existing pipeline functions
from realistic_sim import RealisticFungalComputer, optimize_xor_gate
from rediscover_fungal_parameters import (
    load_models,
    characterize_specimen,
    collect_response_waveforms,
    predict_parameters,
    create_twin,
    refine_parameters_optimization,
    compute_waveform_mismatch,
    test_xor_gate_performance,
    FUNGAL_PARAMS,
    PARAM_BOUNDS,
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

OUTPUT_DIR = Path("batch_rediscovery_from_opt_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_OPT_RESULTS_DIR = Path("optimization_study_results")
DEFAULT_VIABLE_RANGES_PATH = Path("optimization_study_results/viable_param_ranges.json")
DEFAULT_MODEL_DIR = Path("ml_models")
DEFAULT_MODEL_TYPE = 'random_forest'
DEFAULT_OPT_METHOD = 'dual_annealing'
DEFAULT_CONDITIONS = ['oracle', 'ml_only', 'ml_refine', 'random']

# Percentile of tuned_score *below* which specimens are excluded.
# Default 75 → keep top 25% (specimens at or above the 25th percentile).
DEFAULT_SCORE_PERCENTILE = 75

# XOR optimization parameters (used for conditions that need it)
XOR_N_CALLS = 100
XOR_MINIMIZER = 'gp'


# ==========================================
# Specimen Loading
# ==========================================

def find_latest_opt_results() -> Optional[Path]:
    """Return the most recent optimization_results_*.csv, or None."""
    candidates = sorted(
        DEFAULT_OPT_RESULTS_DIR.glob("optimization_results_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_optimized_specimens(
    opt_results_path: Path,
    score_percentile: int = DEFAULT_SCORE_PERCENTILE,
    max_specimens: Optional[int] = None,
) -> pd.DataFrame:
    """Load top-scoring pre-optimized specimens from an optimization study CSV.

    Only rows with success==True and a valid tuned_score (i.e. the XOR gate
    optimisation succeeded and physics were finetuned) are retained.

    Args:
        opt_results_path: Path to optimization_results_*.csv.
        score_percentile: Specimens whose tuned_score falls *below* this
            percentile are excluded.  Default 50 → keep top 50%.
        max_specimens: Optional cap; specimens are taken from the highest-
            scoring end of the filtered set.

    Returns:
        DataFrame sorted by tuned_score descending, with reset index.
    """
    df = pd.read_csv(opt_results_path)
    # Keep only successful rows that have a tuned_score (XOR gate succeeded)
    df = df[(df['success'] == True) & df['tuned_score'].notna()].copy()
    if len(df) == 0:
        logger.error("No successful specimens with a valid tuned_score found in CSV.")
        return df

    threshold = df['tuned_score'].quantile(score_percentile / 100.0)
    viable = df[df['tuned_score'] >= threshold].sort_values(
        'tuned_score', ascending=False
    )
    if max_specimens is not None:
        viable = viable.head(max_specimens)

    logger.info(
        f"Loaded {len(viable)} of {len(df)} successful specimens at or above the "
        f"{score_percentile}th-percentile threshold "
        f"(tuned_score >= {threshold:.4f}) from {Path(opt_results_path).name}"
    )
    return viable.reset_index(drop=True)


# ==========================================
# Viable Range (for the 'random' condition)
# ==========================================

def load_viable_ranges(path: Path) -> Dict:
    """Load viable parameter ranges from define_viable_range.py output.

    Falls back to full PARAM_BOUNDS if file not found.
    """
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        ranges = {
            param: (info['viable_low'], info['viable_high'])
            for param, info in data['param_ranges'].items()
        }
        logger.info(f"Loaded viable ranges from {path.name}")
        return ranges
    logger.warning(
        f"Viable ranges file not found: {path}. "
        "Using full PARAM_BOUNDS. Run define_viable_range.py first for best results."
    )
    return {k: v for k, v in PARAM_BOUNDS.items()}


def sample_from_ranges(ranges: Dict, rng: np.random.RandomState) -> Dict:
    """Sample random parameters from given ranges (uniform; log-uniform for alpha)."""
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
    xor_params: Dict,
    condition: str,
    models: Dict,
    scaler: object,
    features: Dict,
    viable_ranges: Dict,
    rng: np.random.RandomState,
    opt_method: str = DEFAULT_OPT_METHOD,
) -> Dict:
    """Run one condition on a specimen and return metrics.

    Unlike the original batch_rediscovery_study.py, this variant receives
    the pre-optimized XOR gate configuration (xor_params) from the
    optimization study CSV.  All conditions re-use those electrode positions
    when testing transfer — they only differ in *which* twin parameters are
    used for the XOR gate evaluation.

    Args:
        specimen: The specimen (true system, reconstructed from opt CSV)
        true_params: Ground-truth biophysical parameters (tuned_* columns)
        specimen_waveforms: Pre-collected waveforms from specimen
        xor_params: Pre-optimized XOR gate configuration from opt CSV
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
            for param in FUNGAL_PARAMS:
                if param in predicted_params and param in true_params:
                    abs_err = abs(predicted_params[param] - true_params[param])
                    rel_err = abs_err / (abs(true_params[param]) + 1e-10) * 100
                    result[f'ml_abs_err_{param}'] = float(abs_err)
                    result[f'ml_rel_err_{param}'] = float(rel_err)

        elif condition == 'ml_refine':
            predicted_params, _ = predict_parameters(features, models, scaler, DEFAULT_MODEL_TYPE)
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

        # ---- Test XOR on twin using pre-optimized gate config ----
        twin_xor = test_xor_gate_performance(twin, xor_params)
        result['xor_score_on_twin'] = float(xor_params.get('score', np.nan))
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

def _run_specimen_worker(args: Dict) -> List[Dict]:
    """Run all conditions for a single pre-optimized specimen in a worker process.

    Args:
        args: Dict with keys:
            specimen_row_dict  — a single row from the opt-results CSV as a dict
            conditions         — list of conditions to run
            model_dir          — path string to model directory
            model_type         — ML model type string
            opt_method         — waveform refinement optimiser name
            viable_ranges      — pre-loaded viable ranges dict
            specimen_idx       — index for logging / checkpointing

    Returns:
        List of metric dicts (one per condition).
    """
    import warnings
    warnings.filterwarnings('ignore')

    row          = args['specimen_row_dict']
    conditions   = args['conditions']
    model_dir    = Path(args['model_dir'])
    model_type   = args['model_type']
    opt_method   = args['opt_method']
    viable_ranges = args['viable_ranges']
    specimen_idx = args['specimen_idx']

    from realistic_sim import RealisticFungalComputer
    from rediscover_fungal_parameters import (
        load_models, characterize_specimen, collect_response_waveforms,
        predict_parameters, create_twin, refine_parameters_optimization,
        compute_waveform_mismatch, test_xor_gate_performance, FUNGAL_PARAMS,
    )

    num_nodes = int(row['num_nodes'])
    seed      = int(row['random_state'])

    # Reconstruct ground-truth (finetuned) physics parameters
    true_params = {
        'tau_v':   float(row['tuned_tau_v']),
        'tau_w':   float(row['tuned_tau_w']),
        'a':       float(row['tuned_a']),
        'b':       float(row['tuned_b']),
        'v_scale': float(row['tuned_v_scale']),
        'R_off':   float(row['tuned_R_off']),
        'R_on':    float(row['tuned_R_on']),
        'alpha':   float(row['tuned_alpha']),
    }

    # Pre-optimized XOR gate configuration (no re-optimization needed)
    xor_params = {
        'x_A':      float(row['x_A']),
        'y_A':      float(row['y_A']),
        'x_B':      float(row['x_B']),
        'y_B':      float(row['y_B']),
        'x_out':    float(row['x_out']),
        'y_out':    float(row['y_out']),
        'voltage':  float(row['voltage']),
        'duration': float(row['duration']),
        'delay':    float(row['delay']),
        'score':    float(row['tuned_score']),
    }

    cond_results = []
    shared_meta  = {
        'num_nodes':    num_nodes,
        'num_edges':    int(row.get('num_edges', 0)),
        'network_density': float(row.get('network_density', np.nan)),
        'specimen_idx': specimen_idx,
        'specimen_seed': seed,
        'tuned_score':  float(row['tuned_score']),
    }
    for param in FUNGAL_PARAMS:
        shared_meta[f'true_{param}'] = true_params.get(param, float('nan'))

    try:
        specimen = RealisticFungalComputer(num_nodes=num_nodes, random_seed=seed)
        for param, val in true_params.items():
            setattr(specimen, param, val)

        features           = characterize_specimen(specimen)
        specimen_waveforms = collect_response_waveforms(specimen)
        models, scaler     = load_models(model_dir, model_type)

    except Exception as e:
        for cond in conditions:
            rec = {**shared_meta, 'condition': cond,
                   'success': False, 'error_message': f'Specimen setup failed: {e}'}
            cond_results.append(rec)
        return cond_results

    for condition in conditions:
        cond_rng = np.random.RandomState(seed + hash(condition) % 100_000)
        metrics = evaluate_specimen_condition(
            specimen=specimen,
            true_params=true_params,
            specimen_waveforms=specimen_waveforms,
            xor_params=xor_params,
            condition=condition,
            models=models,
            scaler=scaler,
            features=features,
            viable_ranges=viable_ranges,
            rng=cond_rng,
            opt_method=opt_method,
        )
        metrics.update(shared_meta)
        cond_results.append(metrics)

    return cond_results


# ==========================================
# Main Study Loop
# ==========================================

def run_batch_rediscovery_from_opt(
    opt_results_path: Path,
    score_percentile: int = DEFAULT_SCORE_PERCENTILE,
    n_specimens: Optional[int] = None,
    conditions: List[str] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    model_type: str = DEFAULT_MODEL_TYPE,
    opt_method: str = DEFAULT_OPT_METHOD,
    viable_ranges_path: Path = DEFAULT_VIABLE_RANGES_PATH,
    resume: bool = False,
    random_seed: int = 42,
    n_workers: int = 1,
):
    """Run the batch rediscovery study using pre-optimized specimens.

    Args:
        opt_results_path: Path to optimization_results_*.csv from
            systematic_optimization_study.py.
        score_percentile: Specimens below this percentile of tuned_score are
            excluded.  Default 50 → keep top 50%.
        n_specimens: Optional cap on the number of specimens (highest-scoring
            subset of the filtered pool).
        conditions: List of conditions to run per specimen.
        model_dir: Directory containing trained ML models.
        model_type: ML model type ('random_forest' or 'mlp').
        opt_method: Optimization method for ml_refine condition.
        viable_ranges_path: Path to viable_param_ranges.json.
        resume: Resume from latest checkpoint.
        random_seed: Master random seed (used for random condition RNG).
        n_workers: Number of parallel worker processes (1 = serial).
    """
    if conditions is None:
        conditions = DEFAULT_CONDITIONS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file    = OUTPUT_DIR / f"batch_rediscovery_opt_{timestamp}.csv"
    checkpoint_file = OUTPUT_DIR / f"checkpoint_{timestamp}.csv"
    config_file     = OUTPUT_DIR / f"config_{timestamp}.json"

    # ---- Handle resume ----
    all_results    = []
    completed_keys = set()

    if resume:
        checkpoints = sorted(
            OUTPUT_DIR.glob("checkpoint_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            logger.info(f"Resuming from checkpoint: {checkpoints[0].name}")
            prev = pd.read_csv(checkpoints[0])
            all_results = prev.to_dict('records')
            for row in all_results:
                if row.get('success', False):
                    completed_keys.add((row['specimen_idx'], row['condition']))
            timestamp       = checkpoints[0].stem.replace('checkpoint_', '')
            results_file    = OUTPUT_DIR / f"batch_rediscovery_opt_{timestamp}.csv"
            checkpoint_file = OUTPUT_DIR / f"checkpoint_{timestamp}.csv"
            logger.info(f"Loaded {len(completed_keys)} completed (specimen, condition) tuples")
        else:
            logger.info("No checkpoint found. Starting fresh.")

    # ---- Load specimens ----
    specimens_df = load_optimized_specimens(opt_results_path, score_percentile, n_specimens)
    if len(specimens_df) == 0:
        logger.error("No viable specimens found. Aborting.")
        return pd.DataFrame()

    n_total_specimens = len(specimens_df)
    viable_ranges     = load_viable_ranges(viable_ranges_path)

    # Save config
    config = {
        'opt_results_path':  str(opt_results_path),
        'score_percentile':  score_percentile,
        'n_specimens':       n_total_specimens,
        'conditions':        conditions,
        'model_dir':         str(model_dir),
        'model_type':        model_type,
        'opt_method':        opt_method,
        'viable_ranges_path': str(viable_ranges_path),
        'random_seed':       random_seed,
        'timestamp':         timestamp,
    }
    if not config_file.exists():
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    logger.info("=" * 70)
    logger.info("BATCH REDISCOVERY STUDY (from optimization results)")
    logger.info("=" * 70)
    logger.info(f"Opt results:      {opt_results_path}")
    logger.info(f"Score percentile: {score_percentile} "
                f"(keep top {100 - score_percentile}%)")
    logger.info(f"Specimens loaded: {n_total_specimens}")
    logger.info(f"Conditions:       {conditions}")
    logger.info(f"Total runs:       {n_total_specimens * len(conditions)}")
    logger.info(f"Workers:          {n_workers}")

    study_start   = time.time()
    specimen_count = 0

    # ---- Build pending task list ----
    pending_tasks = []
    for specimen_idx, row in specimens_df.iterrows():
        remaining_conditions = [
            c for c in conditions
            if (specimen_idx, c) not in completed_keys
        ]
        if not remaining_conditions:
            continue
        pending_tasks.append({
            'specimen_row_dict': row.to_dict(),
            'conditions':        remaining_conditions,
            'model_dir':         str(model_dir),
            'model_type':        model_type,
            'opt_method':        opt_method,
            'viable_ranges':     viable_ranges,
            'specimen_idx':      int(specimen_idx),
        })

    logger.info(f"Pending specimens: {len(pending_tasks)} / {n_total_specimens}")

    def _handle_specimen_results(cond_results):
        nonlocal specimen_count
        specimen_count += 1
        for metrics in cond_results:
            all_results.append(metrics)
            completed_keys.add((metrics.get('specimen_idx'), metrics.get('condition')))
        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
        elapsed = time.time() - study_start
        avg     = elapsed / specimen_count if specimen_count else 0
        eta     = avg * (n_total_specimens - specimen_count)
        logger.info(
            f"  Specimen {specimen_count}/{n_total_specimens} done | "
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
        for task in pending_tasks:
            cond_results = _run_specimen_worker(task)
            _handle_specimen_results(cond_results)
    else:
        logger.info(f"Launching ProcessPoolExecutor with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(_run_specimen_worker, t): t for t in pending_tasks
            }
            for future in as_completed(future_to_task):
                try:
                    cond_results = future.result()
                except Exception as e:
                    task = future_to_task[future]
                    cond_results = [{
                        'specimen_idx': task['specimen_idx'],
                        'specimen_seed': task['specimen_row_dict'].get('random_state'),
                        'condition': c,
                        'success': False,
                        'error_message': str(e),
                    } for c in task['conditions']]
                _handle_specimen_results(cond_results)

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
        sub   = successful[successful['condition'] == cond]
        n     = len(sub)
        twin_acc = sub['xor_accuracy_twin'].mean() if 'xor_accuracy_twin' in sub else np.nan
        spec_acc = sub['xor_accuracy_specimen'].mean() if 'xor_accuracy_specimen' in sub else np.nan
        t75   = sub['transfer_success_75'].mean() * 100 if 'transfer_success_75' in sub else np.nan
        t100  = sub['transfer_success_100'].mean() * 100 if 'transfer_success_100' in sub else np.nan
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

    # ---- Figure 1: XOR Transfer Accuracy ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    transfer_rates = []
    for cond in conditions_order:
        sub = successful[successful['condition'] == cond]
        t75  = sub['transfer_success_75'].mean() * 100 if 'transfer_success_75' in sub.columns else 0
        t100 = sub['transfer_success_100'].mean() * 100 if 'transfer_success_100' in sub.columns else 0
        transfer_rates.append({'condition': cond, 'threshold': '≥75%', 'rate': t75})
        transfer_rates.append({'condition': cond, 'threshold': '100%', 'rate': t100})

    tr_df  = pd.DataFrame(transfer_rates)
    x      = np.arange(len(conditions_order))
    width  = 0.35
    bars75  = [tr_df[(tr_df['condition'] == c) & (tr_df['threshold'] == '≥75%')]['rate'].values[0]
               for c in conditions_order]
    bars100 = [tr_df[(tr_df['condition'] == c) & (tr_df['threshold'] == '100%')]['rate'].values[0]
               for c in conditions_order]

    ax.bar(x - width/2, bars75,  width, label='≥75% accuracy',
           color=[palette[c] for c in conditions_order], alpha=0.9)
    ax.bar(x + width/2, bars100, width, label='100% accuracy',
           color=[palette[c] for c in conditions_order], alpha=0.5, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions_order], fontsize=11)
    ax.set_ylabel('Transfer Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('XOR Gate Transfer Success Rate\n(pre-optimized specimens)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    plot_data = successful[successful['condition'].isin(conditions_order)].copy()
    sns.violinplot(data=plot_data, x='condition', y='xor_accuracy_specimen',
                   order=conditions_order, palette=palette, ax=ax, inner='quartile', cut=0)
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('XOR Accuracy on Specimen', fontsize=12, fontweight='bold')
    ax.set_title('XOR Accuracy Distribution (Specimen)\n(pre-optimized specimens)',
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions_order], fontsize=10)

    plt.suptitle('Digital Twin XOR Gate Transfer Performance (from Optimization Results)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig1_path = output_dir / 'xor_transfer_accuracy.png'
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {fig1_path}")
    plt.close()

    # ---- Figure 2: Per-parameter recovery error (ML-only vs ML-refine) ----
    ml_conditions = [c for c in ['ml_only', 'ml_refine'] if c in successful['condition'].unique()]
    if ml_conditions:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        for idx, param in enumerate(FUNGAL_PARAMS):
            ax = axes[idx]
            plot_rows = []
            for cond in ml_conditions:
                sub    = successful[successful['condition'] == cond]
                prefix = 'ml' if cond == 'ml_only' else 'refined'
                col    = f'{prefix}_rel_err_{param}'
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

        plt.suptitle('Per-Parameter Recovery Error (ML-only vs ML+Refine)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig2_path = output_dir / 'parameter_recovery.png'
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {fig2_path}")
        plt.close()

    # ---- Figure 3: Waveform mismatch before/after optimization ----
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
            ax.set_title('Waveform Mismatch Before vs After Optimization',
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
            fig3_path = output_dir / 'waveform_mismatch.png'
            plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {fig3_path}")
            plt.close()

    # ---- Figure 4: tuned_score vs transfer success ----
    if 'tuned_score' in successful.columns:
        oracle_df = successful[successful['condition'] == 'oracle'].copy()
        if len(oracle_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = oracle_df['transfer_success_100'].map({True: '#2ecc71', False: '#e74c3c'})
            ax.scatter(oracle_df['tuned_score'], oracle_df['xor_accuracy_specimen'],
                       c=colors, alpha=0.7, s=60, edgecolors='white')
            ax.set_xlabel('Tuned XOR Score (optimization study)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Oracle XOR Accuracy (specimen)', fontsize=12, fontweight='bold')
            ax.set_title('Oracle Baseline: Tuned Score vs Transfer Accuracy',
                         fontsize=13, fontweight='bold')
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='100% transfer'),
                Patch(facecolor='#e74c3c', label='< 100% transfer'),
            ]
            ax.legend(handles=legend_elements, fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig4_path = output_dir / 'oracle_score_vs_accuracy.png'
            plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {fig4_path}")
            plt.close()


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch digital twin rediscovery study using pre-optimized specimens',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Auto-detect latest opt-results CSV, keep top 50%% by tuned_score
  python batch_rediscovery_study_from_opt.py

  # Specify CSV explicitly, keep top 25%% (most successful XOR gates)
  python batch_rediscovery_study_from_opt.py \\
      --opt-results optimization_study_results/optimization_results_<ts>.csv \\
      --score-percentile 75

  # Cap at 60 specimens, use 4 parallel workers
  python batch_rediscovery_study_from_opt.py --n-specimens 60 --n-workers 4

  # Quick test: oracle + ml_only only
  python batch_rediscovery_study_from_opt.py --conditions oracle ml_only --n-specimens 10

  # Resume from checkpoint
  python batch_rediscovery_study_from_opt.py --resume
        """
    )
    parser.add_argument('--opt-results', type=Path, default=None,
                        help='Path to optimization_results_*.csv from '
                             'systematic_optimization_study.py. If omitted, '
                             'the most recent CSV in optimization_study_results/ '
                             'is used automatically.')
    parser.add_argument('--score-percentile', type=int, default=DEFAULT_SCORE_PERCENTILE,
                        help='Exclude specimens below this percentile of tuned_score '
                             '(default %(default)s → keep top '
                             f'{100 - DEFAULT_SCORE_PERCENTILE}%%). '
                             'Higher value = stricter filter (fewer, better specimens).')
    parser.add_argument('--n-specimens', type=int, default=None,
                        help='Cap on the number of specimens to use (highest-scoring '
                             'subset of the filtered pool). Default: all viable.')
    parser.add_argument('--conditions', type=str, nargs='+', default=DEFAULT_CONDITIONS,
                        choices=['oracle', 'ml_only', 'ml_refine', 'random'],
                        help='Conditions to run per specimen.')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                        help='Directory containing trained ML models.')
    parser.add_argument('--model-type', type=str, default=DEFAULT_MODEL_TYPE,
                        choices=['random_forest', 'mlp'],
                        help='ML model type to use.')
    parser.add_argument('--opt-method', type=str, default=DEFAULT_OPT_METHOD,
                        help='Optimization method for ml_refine condition.')
    parser.add_argument('--viable-ranges', type=Path, default=DEFAULT_VIABLE_RANGES_PATH,
                        help='Path to viable_param_ranges.json (used for random condition).')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in output directory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Master random seed.')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel worker processes (default: 1 = serial).')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt.')
    args = parser.parse_args()

    # Resolve opt-results path
    opt_path = args.opt_results
    if opt_path is None:
        opt_path = find_latest_opt_results()
        if opt_path is None:
            parser.error(
                "No optimization_results_*.csv found in optimization_study_results/. "
                "Run systematic_optimization_study.py first, or pass --opt-results <path>."
            )
        print(f"Auto-detected opt results: {opt_path.name}")

    # Quick peek at pool size
    _df = pd.read_csv(opt_path)
    _viable = _df[(_df['success'] == True) & _df['tuned_score'].notna()]
    _threshold = _viable['tuned_score'].quantile(args.score_percentile / 100.0)
    _pool = _viable[_viable['tuned_score'] >= _threshold]
    n_pool = min(len(_pool), args.n_specimens) if args.n_specimens else len(_pool)

    print(f"\n{'='*70}")
    print("BATCH REDISCOVERY STUDY  (from optimization results)")
    print(f"{'='*70}")
    print(f"Opt results:      {opt_path}")
    print(f"Score percentile: {args.score_percentile} "
          f"(keep top {100 - args.score_percentile}%  ≥ {_threshold:.4f})")
    print(f"Specimens in pool: {n_pool}")
    print(f"Conditions:       {args.conditions}")
    print(f"Total runs:       {n_pool * len(args.conditions)}")
    print(f"Viable ranges:    {args.viable_ranges}")
    print(f"Workers:          {args.n_workers} "
          f"({'parallel' if args.n_workers > 1 else 'serial'})")
    print(f"{'='*70}")

    if args.yes:
        proceed = True
    else:
        response = input("\nProceed? (yes/no): ")
        proceed = response.lower() in ['yes', 'y']

    if proceed:
        run_batch_rediscovery_from_opt(
            opt_results_path=opt_path,
            score_percentile=args.score_percentile,
            n_specimens=args.n_specimens,
            conditions=args.conditions,
            model_dir=args.model_dir,
            model_type=args.model_type,
            opt_method=args.opt_method,
            viable_ranges_path=args.viable_ranges,
            resume=args.resume,
            random_seed=args.seed,
            n_workers=args.n_workers,
        )
    else:
        print("Study cancelled.")
