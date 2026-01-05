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
TRIALS_PER_NODE_COUNT = 300  # Aiming for 2000+ total simulations

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
        # 1. Step Response Protocol
        logger.debug("Running step response protocol...")
        step_result = env.step_response_protocol(**STEP_RESPONSE_PARAMS)
        features['step_rise_time'] = step_result['rise_time']
        features['step_saturation_voltage'] = step_result['saturation_voltage']
        features['step_oscillation_index'] = step_result['oscillation_index']
        
        # 2. Paired-Pulse Protocol
        logger.debug("Running paired-pulse protocol...")
        pp_result = env.paired_pulse_protocol(**PAIRED_PULSE_PARAMS)
        # Store recovery ratios for each delay
        for i, delay in enumerate(pp_result['delays']):
            features[f'pp_recovery_ratio_delay_{int(delay)}'] = pp_result['recovery_ratios'][i]
            features[f'pp_first_peak_delay_{int(delay)}'] = pp_result['first_peak_heights'][i]
            features[f'pp_second_peak_delay_{int(delay)}'] = pp_result['second_peak_heights'][i]
        
        # 3. Triangle Sweep Protocol
        logger.debug("Running triangle sweep protocol...")
        tri_result = env.triangle_sweep_protocol(**TRIANGLE_SWEEP_PARAMS)
        features['tri_hysteresis_area'] = tri_result['hysteresis_area']
        
        # Calculate additional derived features
        # Average recovery ratio across delays
        features['pp_avg_recovery_ratio'] = np.mean(pp_result['recovery_ratios'])
        features['pp_std_recovery_ratio'] = np.std(pp_result['recovery_ratios'])
        
        # Peak ratio change (how much recovery improves with longer delays)
        if len(pp_result['recovery_ratios']) >= 2:
            features['pp_recovery_slope'] = (pp_result['recovery_ratios'][-1] - 
                                            pp_result['recovery_ratios'][0]) / (pp_result['delays'][-1] - pp_result['delays'][0])
        else:
            features['pp_recovery_slope'] = 0.0
        
        features['characterization_success'] = True
        features['error_message'] = None
        
    except Exception as e:
        logger.error(f"Characterization failed: {str(e)}")
        # Fill with NaN on failure
        for key in ['step_rise_time', 'step_saturation_voltage', 'step_oscillation_index',
                    'pp_recovery_ratio_delay_200', 'pp_recovery_ratio_delay_800', 'pp_recovery_ratio_delay_2000',
                    'pp_first_peak_delay_200', 'pp_first_peak_delay_800', 'pp_first_peak_delay_2000',
                    'pp_second_peak_delay_200', 'pp_second_peak_delay_800', 'pp_second_peak_delay_2000',
                    'tri_hysteresis_area', 'pp_avg_recovery_ratio', 'pp_std_recovery_ratio', 'pp_recovery_slope']:
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

def run_systematic_characterization(resume: bool = False):
    """Run the systematic characterization study.
    
    Args:
        resume: If True, attempt to resume from the most recent checkpoint
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
    
    # Main loop: iterate over node counts
    for num_nodes in NODE_COUNTS:
        logger.info("")
        logger.info("="*70)
        logger.info(f"TESTING NODE COUNT: {num_nodes}")
        logger.info("="*70)
        
        # Run multiple trials with different random parameters
        for trial_idx in range(TRIALS_PER_NODE_COUNT):
            current_trial += 1
            
            # Skip if this trial was already completed
            if (num_nodes, trial_idx) in completed_trials:
                logger.info(f"Skipping completed trial {current_trial}/{total_trials}: num_nodes={num_nodes}, trial={trial_idx+1}/{TRIALS_PER_NODE_COUNT}")
                continue
            
            # Generate random seed for this trial
            random_state = np.random.randint(0, 1000000)
            rng = np.random.RandomState(random_state)
            
            # Sample fungal parameters
            params = sample_fungal_parameters(rng)
            
            logger.info("")
            logger.info(f"Trial {current_trial}/{total_trials}: num_nodes={num_nodes}, trial={trial_idx+1}/{TRIALS_PER_NODE_COUNT}, seed={random_state}")
            logger.debug(f"Parameters: tau_v={params['tau_v']:.1f}, tau_w={params['tau_w']:.1f}, a={params['a']:.2f}, b={params['b']:.2f}")
            logger.info("-"*70)
            
            trial_start_time = time.time()
            
            try:
                # Create environment with sampled parameters
                env = RealisticFungalComputer(num_nodes=num_nodes, random_seed=random_state)
                apply_parameters_to_env(env, params)
                
                # Run characterization protocols
                features = run_characterization(env)
                
                # Compile trial data
                trial_duration = time.time() - trial_start_time
                record = extract_trial_data(env, params, features, num_nodes, 
                                          trial_idx, random_state, trial_duration)
                
                all_results.append(record)
                
                if features['characterization_success']:
                    logger.info(f"Trial completed successfully in {trial_duration:.1f}s")
                    logger.info(f"Features: rise_time={features['step_rise_time']:.1f}ms, "
                              f"saturation={features['step_saturation_voltage']:.3f}V, "
                              f"hysteresis={features['tri_hysteresis_area']:.4f}")
                else:
                    logger.warning(f"Trial completed with errors in {trial_duration:.1f}s")
                
            except Exception as e:
                logger.error(f"Trial failed with error: {str(e)}")
                # Store error information
                record = {
                    'num_nodes': num_nodes,
                    'trial_idx': trial_idx,
                    'random_state': random_state,
                    'characterization_success': False,
                    'error_message': str(e),
                    'trial_duration_seconds': time.time() - trial_start_time,
                }
                # Add NaN for all parameters and features
                for param in PARAM_RANGES.keys():
                    record[param] = np.nan
                all_results.append(record)
            
            # Save checkpoint every 10 trials
            if len(all_results) % 10 == 0:
                results_df = pd.DataFrame(all_results)
                save_checkpoint(results_df, checkpoint_file)
            
            # Progress update every 50 trials
            if current_trial % 50 == 0:
                elapsed_time = time.time() - study_start_time
                avg_time_per_trial = elapsed_time / current_trial
                estimated_remaining = avg_time_per_trial * (total_trials - current_trial)
                logger.info(f"Progress: {current_trial}/{total_trials} ({100*current_trial/total_trials:.1f}%)")
                logger.info(f"Elapsed: {elapsed_time/60:.1f}min, Estimated remaining: {estimated_remaining/60:.1f}min")
    
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
        logger.info(f"  Step rise time: {successful_trials['step_rise_time'].mean():.1f} ± {successful_trials['step_rise_time'].std():.1f} ms")
        logger.info(f"  Step saturation: {successful_trials['step_saturation_voltage'].mean():.3f} ± {successful_trials['step_saturation_voltage'].std():.3f} V")
        logger.info(f"  Oscillation index: {successful_trials['step_oscillation_index'].mean():.4f} ± {successful_trials['step_oscillation_index'].std():.4f}")
        logger.info(f"  Avg recovery ratio: {successful_trials['pp_avg_recovery_ratio'].mean():.3f} ± {successful_trials['pp_avg_recovery_ratio'].std():.3f}")
        logger.info(f"  Hysteresis area: {successful_trials['tri_hysteresis_area'].mean():.4f} ± {successful_trials['tri_hysteresis_area'].std():.4f}")
    
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
    print("="*70)
    
    if args.yes:
        proceed = True
    else:
        response = input("\nProceed with study? (yes/no): ")
        proceed = response.lower() in ['yes', 'y']
    
    if proceed:
        results_df = run_systematic_characterization(resume=args.resume)
        print("\nStudy completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"Dataset size: {len(results_df[results_df['characterization_success'] == True])} successful samples")
    else:
        print("Study cancelled.")
