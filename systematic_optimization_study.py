"""Systematic study of XOR gate optimization across different network configurations.

This script runs optimize_xor_gate multiple times with varying:
- num_nodes: Systematically varied to study node-density effects
- random_state: Multiple trials for statistical robustness
- tune_physics: Enabled to find optimal fungal characteristics

Results are saved to CSV and JSON for later analysis.
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from realistic_sim import optimize_xor_gate
import networkx as nx
from networkx.algorithms import community
import logging
import argparse
import glob
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

# Node density study: systematic progression
# Start sparse, gradually increase density
# NOTE: 30 and 50 are included to align with sensitivity_analysis.py
# (which uses rng.choice([30, 50, 80])) and batch_rediscovery_study.py.
NODE_COUNTS = [
    # Sparse networks
    20, 30,
    # Medium density
    40, 50,
    # Higher density
    60, 80,
    # Dense networks
    100, 120
]

# Number of random trials per node count
# 20 trials x 8 node counts = 160 total trials.
# At ~32% viability, expect ~51 specimens in the top 25% —
# sufficient for a statistically robust sensitivity analysis.
TRIALS_PER_CONFIG = 20

# Optimization iterations per trial
N_CALLS = 60  # Can be adjusted based on time constraints

# Enable physics tuning
TUNE_PHYSICS = True

# Output directory
OUTPUT_DIR = Path("optimization_study_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# Downstream Column Contract
# ==========================================
# The CSV produced by this study is consumed by:
#   - define_viable_range.py       (reads tuned_score, tuned_* params)
#   - sensitivity_analysis.py      (reads num_nodes, random_state, tuned_* params,
#                                   x_A/y_A, x_B/y_B, x_out/y_out, voltage,
#                                   duration, delay, tuned_score, success)
#   - analyze_optimization_results.py (reads all columns)
#
# Specimen reconstruction contract:
#   RealisticFungalComputer(num_nodes=<num_nodes>, random_seed=<random_state>)
#   reproduces the exact network that was optimized, because optimize_xor_gate()
#   passes random_state directly to RealisticFungalComputer as random_seed.
#   See: realistic_sim.py -> optimize_xor_gate() line:
#       env = RealisticFungalComputer(num_nodes=num_nodes, random_seed=random_state)
#
# Key columns saved per trial:
#   num_nodes, random_state           -> reconstruct network
#   tuned_tau_v .. tuned_alpha        -> ground-truth physics after tuning
#   tuned_score                       -> filter viable specimens (top 25/75%)
#   x_A, y_A, x_B, y_B               -> optimized electrode positions
#   x_out, y_out                      -> optimized output probe position
#   voltage, duration, delay          -> optimized stimulus parameters
#   score                             -> pre-tuning XOR score
# ==========================================

# ==========================================
# Helper Functions
# ==========================================

def calculate_electrode_distances(params):
    """Calculate distances between electrodes and from electrodes to output."""
    x_A, y_A = params['x_A'], params['y_A']
    x_B, y_B = params['x_B'], params['y_B']
    x_out, y_out = params['x_out'], params['y_out']
    
    dist_AB = np.sqrt((x_A - x_B)**2 + (y_A - y_B)**2)
    dist_A_out = np.sqrt((x_A - x_out)**2 + (y_A - y_out)**2)
    dist_B_out = np.sqrt((x_B - x_out)**2 + (y_B - y_out)**2)
    
    return {
        'dist_AB': dist_AB,
        'dist_A_out': dist_A_out,
        'dist_B_out': dist_B_out,
        'dist_avg_input_to_out': (dist_A_out + dist_B_out) / 2
    }

def calculate_network_properties(G):
    """Calculate network topology properties.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with network properties
    """
    properties = {}
    
    # Clustering coefficient (average)
    properties['clustering_coefficient'] = nx.average_clustering(G)
    
    # Algebraic connectivity (second smallest eigenvalue of Laplacian)
    # Only defined for connected graphs
    if nx.is_connected(G):
        properties['algebraic_connectivity'] = nx.algebraic_connectivity(G)
    else:
        # For disconnected graphs, use the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        properties['algebraic_connectivity'] = nx.algebraic_connectivity(G_connected)
    
    # Average shortest path length
    if nx.is_connected(G):
        properties['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        # For disconnected graphs, compute for largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        properties['avg_path_length'] = nx.average_shortest_path_length(G_connected)
    
    # Modularity (using greedy modularity communities)
    communities_generator = community.greedy_modularity_communities(G)
    properties['modularity'] = community.modularity(G, communities_generator)
    
    return properties

def extract_results(optimization_results, num_nodes, trial_idx, random_state):
    """Extract relevant metrics from optimization results."""
    params = optimization_results['params']
    env = optimization_results['env']
    result = optimization_results['result']
    
    # Basic info
    record = {
        'num_nodes': num_nodes,
        'trial_idx': trial_idx,
        'random_state': random_state,
        'num_edges': len(env.edge_list),
        'network_density': len(env.edge_list) / (num_nodes * (num_nodes - 1) / 2),
        'area_size': env.area_size,
    }
    
    # Network topology properties
    network_props = calculate_network_properties(env.G)
    record.update(network_props)
    
    # Electrode positions
    record.update({
        'x_A': params['x_A'],
        'y_A': params['y_A'],
        'x_B': params['x_B'],
        'y_B': params['y_B'],
        'x_out': params['x_out'],
        'y_out': params['y_out'],
    })
    
    # Distances
    distances = calculate_electrode_distances(params)
    record.update(distances)
    
    # Stimulus parameters
    record.update({
        'voltage': params['voltage'],
        'duration': params['duration'],
        'delay': params['delay'],
        'score': params['score'],
    })
    
    # Default fungal parameters (before tuning)
    record.update({
        'default_tau_v': 50.0,
        'default_tau_w': 800.0,
        'default_a': 0.7,
        'default_b': 0.8,
        'default_v_scale': 5.0,
        'default_R_off': 100.0,
        'default_R_on': 10.0,
        'default_alpha': 0.01,
    })
    
    # Tuned physics parameters (if available)
    if 'tuned_params' in optimization_results:
        tuned = optimization_results['tuned_params']
        record.update({
            'tuned_tau_v': tuned['tau_v'],
            'tuned_tau_w': tuned['tau_w'],
            'tuned_a': tuned['a'],
            'tuned_b': tuned['b'],
            'tuned_v_scale': tuned['v_scale'],
            'tuned_R_off': tuned['R_off'],
            'tuned_R_on': tuned['R_on'],
            'tuned_alpha': tuned['alpha'],
            'tuned_score': tuned['score'],
            'score_improvement': tuned['score'] - params['score'],
        })
    else:
        # Fill with NaN if physics tuning was not performed
        record.update({
            'tuned_tau_v': np.nan,
            'tuned_tau_w': np.nan,
            'tuned_a': np.nan,
            'tuned_b': np.nan,
            'tuned_v_scale': np.nan,
            'tuned_R_off': np.nan,
            'tuned_R_on': np.nan,
            'tuned_alpha': np.nan,
            'tuned_score': np.nan,
            'score_improvement': np.nan,
        })
    
    # Optimization statistics
    all_scores = -np.array(result.func_vals)
    record.update({
        'opt_min_score': np.min(all_scores),
        'opt_max_score': np.max(all_scores),
        'opt_mean_score': np.mean(all_scores),
        'opt_std_score': np.std(all_scores),
        'opt_n_calls': len(all_scores),
    })
    
    return record

def save_checkpoint(results_df, checkpoint_path):
    """Save intermediate results to CSV."""
    results_df.to_csv(checkpoint_path, index=False)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def find_latest_checkpoint():
    """Find the most recent checkpoint file in the output directory.
    
    Returns:
        Path to checkpoint file or None if no checkpoint exists
    """
    checkpoint_files = list(OUTPUT_DIR.glob("checkpoint_*.csv"))
    if not checkpoint_files:
        return None
    
    # Sort by modification time, most recent first
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoint_files[0]

def load_checkpoint(checkpoint_path):
    """Load checkpoint data and determine completed trials.
    
    Args:
        checkpoint_path: Path to checkpoint CSV file
        
    Returns:
        Tuple of (results_df, completed_trials_set)
        where completed_trials_set contains (num_nodes, trial_idx) tuples
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    results_df = pd.read_csv(checkpoint_path)
    
    # Get set of completed trials (only successful ones)
    completed_trials = set()
    for _, row in results_df.iterrows():
        if row.get('success', False):
            completed_trials.add((int(row['num_nodes']), int(row['trial_idx'])))
    
    logger.info(f"Loaded {len(results_df)} previous results ({len(completed_trials)} successful trials)")
    return results_df, completed_trials

# ==========================================
# Main Study Loop
# ==========================================

# ==========================================
# Top-level worker (must be picklable — no lambdas)
# ==========================================

def _run_trial_worker(args):
    """Run a single optimization trial in a worker process.

    Args:
        args: Tuple of (num_nodes, trial_idx, random_state, n_calls, tune_physics)

    Returns:
        Record dict (same schema as extract_results output).
    """
    num_nodes, trial_idx, random_state, n_calls, tune_physics = args
    trial_start = time.time()
    try:
        results = optimize_xor_gate(
            num_nodes=num_nodes,
            n_calls=n_calls,
            random_state=random_state,
            tune_physics=tune_physics,
            minimizer='gp',
        )
        record = extract_results(results, num_nodes, trial_idx, random_state)
        record['trial_duration_seconds'] = time.time() - trial_start
        record['success'] = True
        record['error_message'] = None
    except Exception as e:
        record = {
            'num_nodes': num_nodes,
            'trial_idx': trial_idx,
            'random_state': random_state,
            'success': False,
            'error_message': str(e),
            'trial_duration_seconds': time.time() - trial_start,
        }
    return record


def run_systematic_study(resume=False, n_workers=1):
    """Run the systematic optimization study.

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
                
                # Extract timestamp from checkpoint filename to continue with same session
                checkpoint_name = checkpoint_path.stem  # e.g., "checkpoint_20231230_123456"
                timestamp = checkpoint_name.replace('checkpoint_', '')
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
    
    results_file = OUTPUT_DIR / f"optimization_results_{timestamp}.csv"
    checkpoint_file = OUTPUT_DIR / f"checkpoint_{timestamp}.csv"
    config_file = OUTPUT_DIR / f"study_config_{timestamp}.json"
    
    # Save study configuration (only if starting fresh)
    if not config_file.exists():
        config = {
            'node_counts': NODE_COUNTS,
            'trials_per_config': TRIALS_PER_CONFIG,
            'n_calls': N_CALLS,
            'tune_physics': TUNE_PHYSICS,
            'total_trials': len(NODE_COUNTS) * TRIALS_PER_CONFIG,
            'timestamp': timestamp,
            'resumed': resume and len(completed_trials) > 0,
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Study configuration saved: {config_file}")
    
    # Calculate total number of trials
    total_trials = len(NODE_COUNTS) * TRIALS_PER_CONFIG
    current_trial = 0
    
    logger.info("="*70)
    if completed_trials:
        logger.info("RESUMING SYSTEMATIC OPTIMIZATION STUDY")
        logger.info("="*70)
        logger.info(f"Previously completed trials: {len(completed_trials)}/{total_trials}")
        logger.info(f"Remaining trials: {total_trials - len(completed_trials)}")
    else:
        logger.info("STARTING SYSTEMATIC OPTIMIZATION STUDY")
        logger.info("="*70)
    logger.info(f"Node counts to test: {NODE_COUNTS}")
    logger.info(f"Trials per configuration: {TRIALS_PER_CONFIG}")
    logger.info(f"Total trials: {total_trials}")
    logger.info(f"Optimization calls per trial: {N_CALLS}")
    logger.info(f"Physics tuning enabled: {TUNE_PHYSICS}")
    logger.info("="*70)
    
    study_start_time = time.time()

    # Build the list of pending tasks (skip already-completed)
    pending_tasks = []
    for num_nodes in NODE_COUNTS:
        for trial_idx in range(TRIALS_PER_CONFIG):
            if (num_nodes, trial_idx) not in completed_trials:
                random_state = int(np.random.randint(0, 100000))
                pending_tasks.append((num_nodes, trial_idx, random_state, N_CALLS, TUNE_PHYSICS))

    logger.info(f"Pending trials: {len(pending_tasks)} / {total_trials}")
    logger.info(f"Workers: {n_workers}")

    def _handle_record(record):
        """Append record, checkpoint, and log progress."""
        nonlocal current_trial
        current_trial += 1
        all_results.append(record)
        if all_results:
            results_df = pd.DataFrame(all_results)
            save_checkpoint(results_df, checkpoint_file)
        elapsed = time.time() - study_start_time
        done = len(all_results)
        avg = elapsed / done if done else 0
        remaining = avg * (total_trials - done)
        status = 'OK' if record.get('success') else f"FAILED: {record.get('error_message', '')[:60]}"
        logger.info(
            f"[{done}/{total_trials}] nodes={record['num_nodes']} trial={record['trial_idx']} "
            f"seed={record['random_state']} | {status} | "
            f"elapsed={elapsed/60:.1f}min ETA={remaining/60:.1f}min"
        )
        if record.get('success'):
            logger.info(f"  Score: {record.get('score', float('nan')):.4f}  "
                        f"Tuned: {record.get('tuned_score', float('nan')):.4f}")

    if n_workers <= 1:
        # ---- Serial execution (original behaviour) ----
        for task in pending_tasks:
            record = _run_trial_worker(task)
            _handle_record(record)
    else:
        # ---- Parallel execution ----
        logger.info(f"Launching ProcessPoolExecutor with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {executor.submit(_run_trial_worker, t): t for t in pending_tasks}
            for future in as_completed(future_to_task):
                try:
                    record = future.result()
                except Exception as e:
                    task = future_to_task[future]
                    record = {
                        'num_nodes': task[0],
                        'trial_idx': task[1],
                        'random_state': task[2],
                        'success': False,
                        'error_message': str(e),
                        'trial_duration_seconds': 0,
                    }
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
    successful_trials = results_df[results_df['success'] == True]
    if len(successful_trials) > 0:
        logger.info("")
        logger.info("SUMMARY STATISTICS:")
        logger.info(f"Successful trials: {len(successful_trials)}/{len(results_df)}")
        logger.info(f"Score range: [{successful_trials['score'].min():.4f}, {successful_trials['score'].max():.4f}]")
        logger.info(f"Mean score: {successful_trials['score'].mean():.4f} ± {successful_trials['score'].std():.4f}")
        
        if TUNE_PHYSICS:
            tuned_trials = successful_trials[successful_trials['tuned_score'].notna()]
            if len(tuned_trials) > 0:
                logger.info(f"Tuned score range: [{tuned_trials['tuned_score'].min():.4f}, {tuned_trials['tuned_score'].max():.4f}]")
                logger.info(f"Mean tuned score: {tuned_trials['tuned_score'].mean():.4f} ± {tuned_trials['tuned_score'].std():.4f}")
                logger.info(f"Mean improvement: {tuned_trials['score_improvement'].mean():.4f}")
        
        logger.info("")
        logger.info("ELECTRODE DISTANCE STATISTICS:")
        logger.info(f"Mean distance A-B: {successful_trials['dist_AB'].mean():.2f} ± {successful_trials['dist_AB'].std():.2f} mm")
        logger.info(f"Mean distance inputs-output: {successful_trials['dist_avg_input_to_out'].mean():.2f} ± {successful_trials['dist_avg_input_to_out'].std():.2f} mm")
        
        logger.info("")
        logger.info("STIMULUS PARAMETER STATISTICS:")
        logger.info(f"Voltage: {successful_trials['voltage'].mean():.2f} ± {successful_trials['voltage'].std():.2f} V")
        logger.info(f"Duration: {successful_trials['duration'].mean():.1f} ± {successful_trials['duration'].std():.1f} ms")
        logger.info(f"Delay: {successful_trials['delay'].mean():.1f} ± {successful_trials['delay'].std():.1f} ms")
    
    logger.info("="*70)
    
    return results_df

# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Systematic XOR gate optimization study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Start a new study
  python systematic_optimization_study.py
  
  # Resume from the most recent checkpoint
  python systematic_optimization_study.py --resume
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
        '--n-workers', type=int, default=1,
        help='Number of parallel worker processes (default: 1 = serial). '
             'Set to os.cpu_count() or a specific value, e.g. --n-workers 4'
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SYSTEMATIC XOR GATE OPTIMIZATION STUDY")
    print("="*70)
    
    # Check for existing checkpoint if resume is requested
    if args.resume:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            print(f"Found checkpoint: {checkpoint_path.name}")
            print("Will resume from this checkpoint.")
        else:
            print("No checkpoint found. Will start a new study.")
    
    print(f"This study will run {len(NODE_COUNTS) * TRIALS_PER_CONFIG} optimization trials")
    print(f"Node counts: {NODE_COUNTS}")
    print(f"Trials per configuration: {TRIALS_PER_CONFIG}")
    print(f"Physics tuning: {'ENABLED' if TUNE_PHYSICS else 'DISABLED'}")
    print(f"Workers: {args.n_workers} ({'parallel' if args.n_workers > 1 else 'serial'})")
    print("="*70)
    
    if args.yes:
        proceed = True
    else:
        response = input("\nProceed with study? (yes/no): ")
        proceed = response.lower() in ['yes', 'y']
    
    if proceed:
        results_df = run_systematic_study(resume=args.resume, n_workers=args.n_workers)
        print("\nStudy completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR}")
    else:
        print("Study cancelled.")
