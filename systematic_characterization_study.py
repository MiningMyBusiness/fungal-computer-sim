"""Systematic characterization study of fungal networks using system identification protocols.

This script runs system identification protocols (step response, paired-pulse, triangle sweep)
on fungal networks with varying:
- num_nodes: Different network sizes
- fungal constants: Varied FHN and memristor parameters
- random_state: Multiple trials for statistical robustness

The goal is to generate a dataset for training machine learning models that can predict
fungal constants from stimulus response features.

Results are saved to CSV for later ML model training.
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from realistic_sim import RealisticFungalComputer
import logging
import argparse
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# Study Configuration
# ==========================================

# Node counts to test
NODE_COUNTS = [20, 30, 40, 50, 60, 80, 100, 120]

# Number of random parameter sets to test per node count
TRIALS_PER_NODE_COUNT = 50  # Aiming for 2000+ total simulations

# Output directory
OUTPUT_DIR = Path("characterization_study_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameter ranges for fungal constants (based on optimize_fungal_constants)
PARAM_RANGES = {
    'tau_v': (30.0, 150.0),      # Voltage time constant (ms)
    'tau_w': (300.0, 1600.0),    # Recovery variable time constant (ms)
    'a': (0.5, 0.8),             # FHN parameter a
    'b': (0.7, 1.0),             # FHN parameter b
    'v_scale': (0.5, 10.0),      # Voltage scaling factor
    'R_off': (50.0, 300.0),      # High resistance state (Ohms)
    'R_on': (2.0, 50.0),         # Low resistance state (Ohms)
    'alpha': (0.0001, 0.02)      # Memristor adaptation rate
}

# System identification protocol parameters
STEP_RESPONSE_PARAMS = {
    'voltage': 2.0,
    'pulse_duration': 3000.0,
    'probe_distance': 5.0,
    'sim_time': 5000.0
}

PAIRED_PULSE_PARAMS = {
    'voltage': 2.0,
    'pulse_width': 50.0,
    'probe_distance': 5.0,
    'delays': [200.0, 800.0, 2000.0]
}

TRIANGLE_SWEEP_PARAMS = {
    'v_max': 5.0,
    'sweep_rate': 0.01,
    'probe_distance': 5.0
}

# ==========================================
# Helper Functions
# ==========================================

def sample_fungal_parameters(random_state: np.random.RandomState) -> Dict[str, float]:
    """Sample random fungal parameters from the defined ranges.
    
    Args:
        random_state: NumPy random state for reproducibility
        
    Returns:
        Dictionary of sampled parameters
    """
    params = {}
    for param_name, (low, high) in PARAM_RANGES.items():
        # Use log-uniform sampling for alpha (spans orders of magnitude)
        if param_name == 'alpha':
            log_low = np.log10(low)
            log_high = np.log10(high)
            params[param_name] = 10 ** random_state.uniform(log_low, log_high)
        else:
            params[param_name] = random_state.uniform(low, high)
    
    # Apply safety constraints
    # 1. R_off must be significantly higher than R_on
    if params['R_off'] < 1.5 * params['R_on']:
        params['R_off'] = 1.5 * params['R_on']
    
    # 2. FHN stability: b should be >= a for stable regime
    if params['b'] < params['a']:
        params['b'] = params['a'] + random_state.uniform(0.05, 0.2)
    
    return params

def apply_parameters_to_env(env: RealisticFungalComputer, params: Dict[str, float]):
    """Apply fungal parameters to an environment.
    
    Args:
        env: RealisticFungalComputer instance
        params: Dictionary of parameters to apply
    """
    env.tau_v = params['tau_v']
    env.tau_w = params['tau_w']
    env.a = params['a']
    env.b = params['b']
    env.v_scale = params['v_scale']
    env.R_off = params['R_off']
    env.R_on = params['R_on']
    env.alpha = params['alpha']

def run_characterization(env: RealisticFungalComputer) -> Dict:
    """Run all three system identification protocols on the environment.
    
    Args:
        env: RealisticFungalComputer instance with parameters already set
        
    Returns:
        Dictionary containing all response features from the three protocols
    """
    features = {}
    
    try:
        # 1. Step Response Protocol - Extract all new features
        logger.debug("Running step response protocol...")
        step_result = env.step_response_protocol(**STEP_RESPONSE_PARAMS)
        
        # Step response features (13 features)
        features['step_baseline'] = step_result['baseline']
        features['step_peak_amplitude'] = step_result['peak_amplitude']
        features['step_time_to_peak'] = step_result['time_to_peak']
        features['step_peak_to_baseline_ratio'] = step_result['peak_to_baseline_ratio']
        features['step_half_decay_time'] = step_result['half_decay_time']
        features['step_decay_rate'] = step_result['decay_rate']
        features['step_area_under_curve'] = step_result['area_under_curve']
        features['step_response_duration'] = step_result['response_duration']
        features['step_activation_speed'] = step_result['activation_speed']
        features['step_latency'] = step_result['latency']
        features['step_initial_slope'] = step_result['initial_slope']
        features['step_settling_deviation'] = step_result['settling_deviation']
        features['step_asymmetry_index'] = step_result['asymmetry_index']
        
        # 2. Paired-Pulse Protocol - Extract all new features
        logger.debug("Running paired-pulse protocol...")
        pp_result = env.paired_pulse_protocol(**PAIRED_PULSE_PARAMS)
        
        # Store features for each delay (18 features per delay)
        for result in pp_result['results']:
            delay = int(result['delay'])
            prefix = f'pp_delay_{delay}'
            
            # Individual pulse metrics
            features[f'{prefix}_peak1_amplitude'] = result['peak1_amplitude']
            features[f'{prefix}_peak2_amplitude'] = result['peak2_amplitude']
            features[f'{prefix}_time_to_peak1'] = result['time_to_peak1']
            features[f'{prefix}_time_to_peak2'] = result['time_to_peak2']
            features[f'{prefix}_auc1'] = result['auc1']
            features[f'{prefix}_auc2'] = result['auc2']
            features[f'{prefix}_peak_width1'] = result['peak_width1']
            features[f'{prefix}_peak_width2'] = result['peak_width2']
            
            # Recovery/facilitation metrics
            features[f'{prefix}_peak_ratio'] = result['peak_ratio']
            features[f'{prefix}_auc_ratio'] = result['auc_ratio']
            features[f'{prefix}_latency_change'] = result['latency_change']
            features[f'{prefix}_width_ratio'] = result['width_ratio']
            
            # Inter-pulse dynamics
            features[f'{prefix}_baseline_shift'] = result['baseline_shift']
            features[f'{prefix}_recovery_fraction'] = result['recovery_fraction']
            features[f'{prefix}_effective_ipi'] = result['effective_ipi']
            
            # Shape and aggregate metrics
            features[f'{prefix}_waveform_correlation'] = result['waveform_correlation']
            features[f'{prefix}_total_response'] = result['total_response']
            features[f'{prefix}_facilitation_index'] = result['facilitation_index']
        
        # Aggregate paired-pulse features across delays
        if len(pp_result['results']) > 0:
            peak_ratios = [r['peak_ratio'] for r in pp_result['results']]
            facilitation_indices = [r['facilitation_index'] for r in pp_result['results']]
            
            features['pp_avg_peak_ratio'] = np.mean(peak_ratios)
            features['pp_std_peak_ratio'] = np.std(peak_ratios)
            features['pp_avg_facilitation_index'] = np.mean(facilitation_indices)
            
            # Recovery slope (change in peak ratio with delay)
            if len(peak_ratios) >= 2:
                delays = [r['delay'] for r in pp_result['results']]
                features['pp_recovery_slope'] = (peak_ratios[-1] - peak_ratios[0]) / (delays[-1] - delays[0])
            else:
                features['pp_recovery_slope'] = 0.0
        else:
            features['pp_avg_peak_ratio'] = np.nan
            features['pp_std_peak_ratio'] = np.nan
            features['pp_avg_facilitation_index'] = np.nan
            features['pp_recovery_slope'] = np.nan
        
        # 3. Triangle Sweep Protocol - Extract all new features
        logger.debug("Running triangle sweep protocol...")
        tri_result = env.triangle_sweep_protocol(**TRIANGLE_SWEEP_PARAMS)
        
        # Hysteresis metrics (4 features)
        features['tri_total_hysteresis_area'] = tri_result['total_hysteresis_area']
        features['tri_pos_hysteresis'] = tri_result['pos_hysteresis']
        features['tri_neg_hysteresis'] = tri_result['neg_hysteresis']
        features['tri_max_hysteresis_width'] = tri_result['max_hysteresis_width']
        
        # Asymmetry metrics (4 features)
        features['tri_response_at_pos_max'] = tri_result['response_at_pos_max']
        features['tri_response_at_neg_max'] = tri_result['response_at_neg_max']
        features['tri_pos_neg_ratio'] = tri_result['pos_neg_ratio']
        features['tri_rectification_index'] = tri_result['rectification_index']
        
        # Nonlinearity metrics (3 features)
        features['tri_linearity_deviation'] = tri_result['linearity_deviation']
        features['tri_slope_variation'] = tri_result['slope_variation']
        features['tri_num_inflection_points'] = tri_result['num_inflection_points']
        
        # Dynamic range (2 features)
        features['tri_response_amplitude'] = tri_result['response_amplitude']
        features['tri_voltage_gain'] = tri_result['voltage_gain']
        
        # Phase-specific features (3 features)
        features['tri_phase1_gain'] = tri_result['phase1_gain']
        features['tri_phase2_gain'] = tri_result['phase2_gain']
        features['tri_phase3_gain'] = tri_result['phase3_gain']
        
        # Memory effects (2 features)
        features['tri_return_point_deviation'] = tri_result['return_point_deviation']
        features['tri_loop_closure_error'] = tri_result['loop_closure_error']
        
        # Shape descriptors (3 features)
        features['tri_loop_eccentricity'] = tri_result['loop_eccentricity']
        features['tri_centroid_v_applied'] = tri_result['centroid_v_applied']
        features['tri_centroid_v_response'] = tri_result['centroid_v_response']
        
        # Frequency content (2 features)
        features['tri_smoothness_index'] = tri_result['smoothness_index']
        features['tri_oscillation_count'] = tri_result['oscillation_count']
        
        features['characterization_success'] = True
        features['error_message'] = None
        
    except Exception as e:
        logger.error(f"Characterization failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fill with NaN on failure - create a comprehensive list of all feature keys
        feature_keys = []
        
        # Step response features
        step_features = ['step_baseline', 'step_peak_amplitude', 'step_time_to_peak', 
                        'step_peak_to_baseline_ratio', 'step_half_decay_time', 'step_decay_rate',
                        'step_area_under_curve', 'step_response_duration', 'step_activation_speed',
                        'step_latency', 'step_initial_slope', 'step_settling_deviation', 
                        'step_asymmetry_index']
        feature_keys.extend(step_features)
        
        # Paired-pulse features (for each delay)
        for delay in PAIRED_PULSE_PARAMS['delays']:
            delay_int = int(delay)
            prefix = f'pp_delay_{delay_int}'
            pp_features = [
                f'{prefix}_peak1_amplitude', f'{prefix}_peak2_amplitude',
                f'{prefix}_time_to_peak1', f'{prefix}_time_to_peak2',
                f'{prefix}_auc1', f'{prefix}_auc2',
                f'{prefix}_peak_width1', f'{prefix}_peak_width2',
                f'{prefix}_peak_ratio', f'{prefix}_auc_ratio',
                f'{prefix}_latency_change', f'{prefix}_width_ratio',
                f'{prefix}_baseline_shift', f'{prefix}_recovery_fraction',
                f'{prefix}_effective_ipi', f'{prefix}_waveform_correlation',
                f'{prefix}_total_response', f'{prefix}_facilitation_index'
            ]
            feature_keys.extend(pp_features)
        
        # Aggregate paired-pulse features
        feature_keys.extend(['pp_avg_peak_ratio', 'pp_std_peak_ratio', 
                           'pp_avg_facilitation_index', 'pp_recovery_slope'])
        
        # Triangle sweep features
        tri_features = [
            'tri_total_hysteresis_area', 'tri_pos_hysteresis', 'tri_neg_hysteresis',
            'tri_max_hysteresis_width', 'tri_response_at_pos_max', 'tri_response_at_neg_max',
            'tri_pos_neg_ratio', 'tri_rectification_index', 'tri_linearity_deviation',
            'tri_slope_variation', 'tri_num_inflection_points', 'tri_response_amplitude',
            'tri_voltage_gain', 'tri_phase1_gain', 'tri_phase2_gain', 'tri_phase3_gain',
            'tri_return_point_deviation', 'tri_loop_closure_error', 'tri_loop_eccentricity',
            'tri_centroid_v_applied', 'tri_centroid_v_response', 'tri_smoothness_index',
            'tri_oscillation_count'
        ]
        feature_keys.extend(tri_features)
        
        # Set all to NaN
        for key in feature_keys:
            features[key] = np.nan
            
        features['characterization_success'] = False
        features['error_message'] = str(e)
    
    return features

def extract_trial_data(env: RealisticFungalComputer, params: Dict[str, float], 
                      features: Dict, num_nodes: int, trial_idx: int, 
                      random_state: int, trial_duration: float) -> Dict:
    """Compile all data for a single trial into a record.
    
    Args:
        env: RealisticFungalComputer instance
        params: Fungal parameters used
        features: Response features from characterization
        num_nodes: Number of nodes in network
        trial_idx: Trial index
        random_state: Random seed used
        trial_duration: Time taken for trial (seconds)
        
    Returns:
        Dictionary containing all trial data
    """
    record = {
        # Trial metadata
        'num_nodes': num_nodes,
        'trial_idx': trial_idx,
        'random_state': random_state,
        'trial_duration_seconds': trial_duration,
        
        # Network properties
        'num_edges': len(env.edge_list),
        'network_density': len(env.edge_list) / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0,
        'area_size': env.area_size,
        'coupling_radius': env.coupling_radius,
        
        # Fungal parameters (ground truth for ML)
        'tau_v': params['tau_v'],
        'tau_w': params['tau_w'],
        'a': params['a'],
        'b': params['b'],
        'v_scale': params['v_scale'],
        'R_off': params['R_off'],
        'R_on': params['R_on'],
        'alpha': params['alpha'],
        
        # Derived parameter features
        'R_ratio': params['R_off'] / params['R_on'],
        'tau_ratio': params['tau_w'] / params['tau_v'],
        'b_minus_a': params['b'] - params['a'],
    }
    
    # Add response features
    record.update(features)
    
    return record

def save_checkpoint(results_df: pd.DataFrame, checkpoint_path: Path):
    """Save intermediate results to CSV."""
    results_df.to_csv(checkpoint_path, index=False)
    logger.info(f"Checkpoint saved: {checkpoint_path} ({len(results_df)} records)")

def find_latest_checkpoint() -> Path:
    """Find the most recent checkpoint file in the output directory."""
    checkpoint_files = list(OUTPUT_DIR.glob("checkpoint_*.csv"))
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoint_files[0]

def load_checkpoint(checkpoint_path: Path) -> Tuple[pd.DataFrame, set]:
    """Load checkpoint data and determine completed trials.
    
    Returns:
        Tuple of (results_df, completed_trials_set)
        where completed_trials_set contains (num_nodes, trial_idx) tuples
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    results_df = pd.read_csv(checkpoint_path)
    
    completed_trials = set()
    for _, row in results_df.iterrows():
        if row.get('characterization_success', False):
            completed_trials.add((int(row['num_nodes']), int(row['trial_idx'])))
    
    logger.info(f"Loaded {len(results_df)} previous results ({len(completed_trials)} successful trials)")
    return results_df, completed_trials

# ==========================================
# Main Study Loop
# ==========================================

# ==========================================
# Top-level worker (must be picklable — no lambdas)
# ==========================================

def _characterize_trial_worker(args):
    """Run a single characterization trial in a worker process.

    Args:
        args: Tuple of (num_nodes, trial_idx, random_state)

    Returns:
        Record dict (same schema as extract_trial_data output).
    """
    num_nodes, trial_idx, random_state = args
    trial_start = time.time()
    rng = np.random.RandomState(random_state)
    params = sample_fungal_parameters(rng)
    try:
        env = RealisticFungalComputer(num_nodes=num_nodes, random_seed=random_state)
        apply_parameters_to_env(env, params)
        features = run_characterization(env)
        trial_duration = time.time() - trial_start
        record = extract_trial_data(env, params, features, num_nodes,
                                    trial_idx, random_state, trial_duration)
    except Exception as e:
        trial_duration = time.time() - trial_start
        record = {
            'num_nodes': num_nodes,
            'trial_idx': trial_idx,
            'random_state': random_state,
            'characterization_success': False,
            'error_message': str(e),
            'trial_duration_seconds': trial_duration,
        }
        for param in PARAM_RANGES.keys():
            record[param] = float('nan')
    return record


def run_systematic_characterization(resume: bool = False, n_workers: int = 1):
    """Run the systematic characterization study.

    Args:
        resume: If True, attempt to resume from the most recent checkpoint
        n_workers: Number of parallel worker processes (1 = serial)
    """
    # Handle resume logic
    completed_trials = set()
    all_results = []
    
    if resume:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            try:
                results_df, completed_trials = load_checkpoint(checkpoint_path)
                all_results = results_df.to_dict('records')
                timestamp = checkpoint_path.stem.replace('checkpoint_', '')
                logger.info(f"Resuming study session: {timestamp}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting fresh study instead")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            logger.info("No checkpoint found. Starting fresh study.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = OUTPUT_DIR / f"characterization_results_{timestamp}.csv"
    checkpoint_file = OUTPUT_DIR / f"checkpoint_{timestamp}.csv"
    config_file = OUTPUT_DIR / f"study_config_{timestamp}.json"
    
    # Save study configuration
    if not config_file.exists():
        config = {
            'node_counts': NODE_COUNTS,
            'trials_per_node_count': TRIALS_PER_NODE_COUNT,
            'total_trials': len(NODE_COUNTS) * TRIALS_PER_NODE_COUNT,
            'param_ranges': PARAM_RANGES,
            'step_response_params': STEP_RESPONSE_PARAMS,
            'paired_pulse_params': PAIRED_PULSE_PARAMS,
            'triangle_sweep_params': TRIANGLE_SWEEP_PARAMS,
            'timestamp': timestamp,
            'resumed': resume and len(completed_trials) > 0,
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Study configuration saved: {config_file}")
    
    # Calculate total number of trials
    total_trials = len(NODE_COUNTS) * TRIALS_PER_NODE_COUNT
    current_trial = 0
    
    logger.info("="*70)
    if completed_trials:
        logger.info("RESUMING SYSTEMATIC CHARACTERIZATION STUDY")
        logger.info("="*70)
        logger.info(f"Previously completed trials: {len(completed_trials)}/{total_trials}")
        logger.info(f"Remaining trials: {total_trials - len(completed_trials)}")
    else:
        logger.info("STARTING SYSTEMATIC CHARACTERIZATION STUDY")
        logger.info("="*70)
    logger.info(f"Node counts to test: {NODE_COUNTS}")
    logger.info(f"Trials per node count: {TRIALS_PER_NODE_COUNT}")
    logger.info(f"Total trials: {total_trials}")
    logger.info(f"Target dataset size: {total_trials} samples")
    logger.info("="*70)
    
    study_start_time = time.time()

    # Build list of pending tasks
    pending_tasks = []
    for num_nodes in NODE_COUNTS:
        for trial_idx in range(TRIALS_PER_NODE_COUNT):
            if (num_nodes, trial_idx) not in completed_trials:
                random_state = int(np.random.randint(0, 1_000_000))
                pending_tasks.append((num_nodes, trial_idx, random_state))

    logger.info(f"Pending trials: {len(pending_tasks)} / {total_trials}")
    logger.info(f"Workers: {n_workers}")

    def _handle_record(record):
        """Append record, checkpoint every 10, log progress."""
        nonlocal current_trial
        current_trial += 1
        all_results.append(record)
        if len(all_results) % 10 == 0:
            results_df = pd.DataFrame(all_results)
            save_checkpoint(results_df, checkpoint_file)
        elapsed = time.time() - study_start_time
        done = len(all_results)
        avg = elapsed / done if done else 0
        remaining = avg * (total_trials - done)
        ok = record.get('characterization_success', False)
        status = 'OK' if ok else f"FAILED: {record.get('error_message', '')[:60]}"
        if done % 10 == 0 or not ok:
            logger.info(
                f"[{done}/{total_trials}] nodes={record['num_nodes']} trial={record['trial_idx']} "
                f"| {status} | elapsed={elapsed/60:.1f}min ETA={remaining/60:.1f}min"
            )

    if n_workers <= 1:
        # ---- Serial execution (original behaviour) ----
        for task in pending_tasks:
            record = _characterize_trial_worker(task)
            _handle_record(record)
    else:
        # ---- Parallel execution ----
        logger.info(f"Launching ProcessPoolExecutor with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {executor.submit(_characterize_trial_worker, t): t for t in pending_tasks}
            for future in as_completed(future_to_task):
                try:
                    record = future.result()
                except Exception as e:
                    task = future_to_task[future]
                    record = {
                        'num_nodes': task[0],
                        'trial_idx': task[1],
                        'random_state': task[2],
                        'characterization_success': False,
                        'error_message': str(e),
                        'trial_duration_seconds': 0,
                    }
                    for param in PARAM_RANGES.keys():
                        record[param] = float('nan')
                _handle_record(record)
    
    # Save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_file, index=False)
    
    total_time = time.time() - study_start_time
    
    logger.info("")
    logger.info("="*70)
    logger.info("STUDY COMPLETE")
    logger.info("="*70)
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Configuration saved to: {config_file}")
    
    # Print summary statistics
    successful_trials = results_df[results_df['characterization_success'] == True]
    if len(successful_trials) > 0:
        logger.info("")
        logger.info("SUMMARY STATISTICS:")
        logger.info(f"Successful trials: {len(successful_trials)}/{len(results_df)}")
        
        logger.info("")
        logger.info("PARAMETER RANGES (sampled):")
        for param in PARAM_RANGES.keys():
            if param in successful_trials.columns:
                logger.info(f"  {param}: [{successful_trials[param].min():.4f}, {successful_trials[param].max():.4f}]")
        
        logger.info("")
        logger.info("RESPONSE FEATURE STATISTICS:")
        logger.info(f"  Step time to peak: {successful_trials['step_time_to_peak'].mean():.1f} ± {successful_trials['step_time_to_peak'].std():.1f} ms")
        logger.info(f"  Step peak amplitude: {successful_trials['step_peak_amplitude'].mean():.3f} ± {successful_trials['step_peak_amplitude'].std():.3f} V")
        logger.info(f"  Step activation speed: {successful_trials['step_activation_speed'].mean():.1f} ± {successful_trials['step_activation_speed'].std():.1f} ms")
        logger.info(f"  Avg peak ratio: {successful_trials['pp_avg_peak_ratio'].mean():.3f} ± {successful_trials['pp_avg_peak_ratio'].std():.3f}")
        logger.info(f"  Total hysteresis area: {successful_trials['tri_total_hysteresis_area'].mean():.4f} ± {successful_trials['tri_total_hysteresis_area'].std():.4f}")
    
    logger.info("="*70)
    
    return results_df

# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Systematic characterization study for ML training data generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Start a new study
  python systematic_characterization_study.py
  
  # Resume from the most recent checkpoint
  python systematic_characterization_study.py --resume
  
  # Quick test with fewer trials
  python systematic_characterization_study.py --quick
        """
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the most recent checkpoint if available'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt and start immediately'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode: fewer trials for testing'
    )
    parser.add_argument(
        '--n-workers', type=int, default=1,
        help='Number of parallel worker processes (default: 1 = serial). '
             'E.g. --n-workers 4'
    )
    args = parser.parse_args()
    
    # Adjust configuration for quick mode
    if args.quick:
        NODE_COUNTS = [20, 40, 60]
        TRIALS_PER_NODE_COUNT = 10
        logger.info("QUICK MODE: Running reduced trial set for testing")
    
    print("\n" + "="*70)
    print("SYSTEMATIC CHARACTERIZATION STUDY")
    print("="*70)
    
    if args.resume:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            print(f"Found checkpoint: {checkpoint_path.name}")
            print("Will resume from this checkpoint.")
        else:
            print("No checkpoint found. Will start a new study.")
    
    total_trials = len(NODE_COUNTS) * TRIALS_PER_NODE_COUNT
    print(f"This study will run {total_trials} characterization trials")
    print(f"Node counts: {NODE_COUNTS}")
    print(f"Trials per node count: {TRIALS_PER_NODE_COUNT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Workers: {args.n_workers} ({'parallel' if args.n_workers > 1 else 'serial'})")
    print("="*70)
    
    if args.yes:
        proceed = True
    else:
        response = input("\nProceed with study? (yes/no): ")
        proceed = response.lower() in ['yes', 'y']
    
    if proceed:
        results_df = run_systematic_characterization(resume=args.resume, n_workers=args.n_workers)
        print("\nStudy completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"Dataset size: {len(results_df[results_df['characterization_success'] == True])} successful samples")
    else:
        print("Study cancelled.")
