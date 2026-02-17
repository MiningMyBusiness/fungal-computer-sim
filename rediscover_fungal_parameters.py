"""Rediscover fungal parameters using ML models and optimization.

This script:
1. Instantiates a random RealisticFungalComputer specimen with unknown parameters
2. Runs characterization protocols to collect response features
3. Uses trained ML models to predict fungal parameters from features
4. Creates a Twin with inferred parameters
5. Refines parameters using optimization to match response waveforms
6. Uses the digital twin to optimize an XOR gate, then validates it on the specimen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from realistic_sim import RealisticFungalComputer, optimize_xor_gate
from systematic_characterization_study import (
    run_characterization, 
    STEP_RESPONSE_PARAMS,
    PAIRED_PULSE_PARAMS,
    TRIANGLE_SWEEP_PARAMS
)
import joblib
import json
import argparse
import logging
from typing import Dict, Tuple, List
from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo, basinhopping
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ==========================================
# Configuration
# ==========================================

OUTPUT_DIR = Path("parameter_rediscovery_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameters to predict
FUNGAL_PARAMS = ['tau_v', 'tau_w', 'a', 'b', 'v_scale', 'R_off', 'R_on', 'alpha']

# Feature columns (must match training)
FEATURE_COLUMNS = [
    # Step response features (13 features)
    'step_baseline',
    'step_peak_amplitude',
    'step_time_to_peak',
    'step_peak_to_baseline_ratio',
    'step_half_decay_time',
    'step_decay_rate',
    'step_area_under_curve',
    'step_response_duration',
    'step_activation_speed',
    'step_latency',
    'step_initial_slope',
    'step_settling_deviation',
    'step_asymmetry_index',
    
    # Paired-pulse features (18 features per delay: 200, 800, 2000)
    # Delay 200ms
    'pp_delay_200_peak1_amplitude',
    'pp_delay_200_peak2_amplitude',
    'pp_delay_200_time_to_peak1',
    'pp_delay_200_time_to_peak2',
    'pp_delay_200_auc1',
    'pp_delay_200_auc2',
    'pp_delay_200_peak_width1',
    'pp_delay_200_peak_width2',
    'pp_delay_200_peak_ratio',
    'pp_delay_200_auc_ratio',
    'pp_delay_200_latency_change',
    'pp_delay_200_width_ratio',
    'pp_delay_200_baseline_shift',
    'pp_delay_200_recovery_fraction',
    'pp_delay_200_effective_ipi',
    'pp_delay_200_waveform_correlation',
    'pp_delay_200_total_response',
    'pp_delay_200_facilitation_index',
    
    # Delay 800ms
    'pp_delay_800_peak1_amplitude',
    'pp_delay_800_peak2_amplitude',
    'pp_delay_800_time_to_peak1',
    'pp_delay_800_time_to_peak2',
    'pp_delay_800_auc1',
    'pp_delay_800_auc2',
    'pp_delay_800_peak_width1',
    'pp_delay_800_peak_width2',
    'pp_delay_800_peak_ratio',
    'pp_delay_800_auc_ratio',
    'pp_delay_800_latency_change',
    'pp_delay_800_width_ratio',
    'pp_delay_800_baseline_shift',
    'pp_delay_800_recovery_fraction',
    'pp_delay_800_effective_ipi',
    'pp_delay_800_waveform_correlation',
    'pp_delay_800_total_response',
    'pp_delay_800_facilitation_index',
    
    # Delay 2000ms
    'pp_delay_2000_peak1_amplitude',
    'pp_delay_2000_peak2_amplitude',
    'pp_delay_2000_time_to_peak1',
    'pp_delay_2000_time_to_peak2',
    'pp_delay_2000_auc1',
    'pp_delay_2000_auc2',
    'pp_delay_2000_peak_width1',
    'pp_delay_2000_peak_width2',
    'pp_delay_2000_peak_ratio',
    'pp_delay_2000_auc_ratio',
    'pp_delay_2000_latency_change',
    'pp_delay_2000_width_ratio',
    'pp_delay_2000_baseline_shift',
    'pp_delay_2000_recovery_fraction',
    'pp_delay_2000_effective_ipi',
    'pp_delay_2000_waveform_correlation',
    'pp_delay_2000_total_response',
    'pp_delay_2000_facilitation_index',
    
    # Aggregate paired-pulse features (4 features)
    'pp_avg_peak_ratio',
    'pp_std_peak_ratio',
    'pp_avg_facilitation_index',
    'pp_recovery_slope',
    
    # Triangle sweep features (23 features)
    'tri_total_hysteresis_area',
    'tri_pos_hysteresis',
    'tri_neg_hysteresis',
    'tri_max_hysteresis_width',
    'tri_response_at_pos_max',
    'tri_response_at_neg_max',
    'tri_pos_neg_ratio',
    'tri_rectification_index',
    'tri_linearity_deviation',
    'tri_slope_variation',
    'tri_num_inflection_points',
    'tri_response_amplitude',
    'tri_voltage_gain',
    'tri_phase1_gain',
    'tri_phase2_gain',
    'tri_phase3_gain',
    'tri_return_point_deviation',
    'tri_loop_closure_error',
    'tri_loop_eccentricity',
    'tri_centroid_v_applied',
    'tri_centroid_v_response',
    'tri_smoothness_index',
    'tri_oscillation_count',
]

# Parameter bounds for optimization
PARAM_BOUNDS = {
    'tau_v': (30.0, 150.0),
    'tau_w': (300.0, 1600.0),
    'a': (0.5, 0.8),
    'b': (0.7, 1.0),
    'v_scale': (0.5, 10.0),
    'R_off': (50.0, 300.0),
    'R_on': (2.0, 50.0),
    'alpha': (0.0001, 0.02)
}

# ==========================================
# Model Loading
# ==========================================

def find_latest_models(model_dir: Path, model_type: str = 'random_forest') -> Dict:
    """Find the most recent trained models.
    
    Args:
        model_dir: Directory containing model files
        model_type: Type of model ('random_forest' or 'mlp')
        
    Returns:
        Dictionary of {param_name: model_path}
    """
    logger.info(f"Searching for {model_type} models in {model_dir}")
    
    model_files = {}
    for param in FUNGAL_PARAMS + ['num_nodes', 'num_edges', 'network_density']:
        pattern = f"{model_type}_{param}_*.pkl"
        matching_files = list(model_dir.glob(pattern))
        
        if matching_files:
            # Get most recent file
            latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
            model_files[param] = latest_file
            logger.debug(f"  Found {param}: {latest_file.name}")
        else:
            logger.warning(f"  No model found for {param}")
    
    return model_files

def load_models(model_dir: Path, model_type: str = 'random_forest') -> Tuple[Dict, object]:
    """Load trained models and scaler.
    
    Args:
        model_dir: Directory containing model files
        model_type: Type of model to load
        
    Returns:
        Tuple of (models_dict, scaler)
    """
    logger.info(f"Loading {model_type} models...")
    
    model_files = find_latest_models(model_dir, model_type)
    
    if not model_files:
        raise FileNotFoundError(f"No {model_type} models found in {model_dir}")
    
    models = {}
    for param, model_path in model_files.items():
        models[param] = joblib.load(model_path)
        logger.debug(f"  Loaded {param}")
    
    # Load scaler (for MLP)
    scaler_files = list(model_dir.glob("scaler_*.pkl"))
    if scaler_files:
        scaler_path = max(scaler_files, key=lambda p: p.stat().st_mtime)
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler: {scaler_path.name}")
    else:
        scaler = None
        logger.warning("No scaler found")
    
    return models, scaler

# ==========================================
# Specimen Creation and Characterization
# ==========================================

def create_random_specimen(num_nodes: int = None, random_seed: int = None) -> Tuple[RealisticFungalComputer, Dict]:
    """Create a RealisticFungalComputer with random parameters.
    
    Args:
        num_nodes: Number of nodes (if None, randomly chosen)
        random_seed: Random seed (if None, randomly chosen)
        
    Returns:
        Tuple of (specimen, true_params)
    """
    if random_seed is None:
        random_seed = np.random.randint(0, 1000000)
    
    if num_nodes is None:
        num_nodes = np.random.choice([30, 40, 50, 60, 80])
    
    logger.info(f"Creating specimen: num_nodes={num_nodes}, seed={random_seed}")
    
    # Create specimen
    specimen = RealisticFungalComputer(num_nodes=num_nodes, random_seed=random_seed)
    
    # Randomize fungal parameters
    rng = np.random.RandomState(random_seed + 1)
    
    true_params = {}
    for param, (low, high) in PARAM_BOUNDS.items():
        if param == 'alpha':
            # Log-uniform for alpha
            log_low = np.log10(low)
            log_high = np.log10(high)
            true_params[param] = 10 ** rng.uniform(log_low, log_high)
        else:
            true_params[param] = rng.uniform(low, high)
    
    # Apply safety constraints
    if true_params['R_off'] < 1.5 * true_params['R_on']:
        true_params['R_off'] = 1.5 * true_params['R_on']
    
    if true_params['b'] < true_params['a']:
        true_params['b'] = true_params['a'] + rng.uniform(0.05, 0.2)
    
    # Apply parameters to specimen
    specimen.tau_v = true_params['tau_v']
    specimen.tau_w = true_params['tau_w']
    specimen.a = true_params['a']
    specimen.b = true_params['b']
    specimen.v_scale = true_params['v_scale']
    specimen.R_off = true_params['R_off']
    specimen.R_on = true_params['R_on']
    specimen.alpha = true_params['alpha']
    
    logger.info(f"True parameters: tau_v={true_params['tau_v']:.1f}, tau_w={true_params['tau_w']:.1f}, "
                f"a={true_params['a']:.2f}, b={true_params['b']:.2f}, alpha={true_params['alpha']:.4f}")
    
    return specimen, true_params

def characterize_specimen(specimen: RealisticFungalComputer) -> Dict:
    """Run characterization protocols on specimen and extract features.
    
    Args:
        specimen: RealisticFungalComputer instance
        
    Returns:
        Dictionary of response features
    """
    logger.info("Running characterization protocols on specimen...")
    
    features = run_characterization(specimen)
    
    if not features['characterization_success']:
        raise RuntimeError(f"Characterization failed: {features['error_message']}")
    
    logger.info(f"Characterization complete: time_to_peak={features['step_time_to_peak']:.1f}ms, "
                f"peak_amplitude={features['step_peak_amplitude']:.3f}V, "
                f"hysteresis={features['tri_total_hysteresis_area']:.4f}")
    
    return features

def collect_response_waveforms(specimen: RealisticFungalComputer) -> Dict:
    """Collect full response waveforms for optimization.
    
    Args:
        specimen: RealisticFungalComputer instance
        
    Returns:
        Dictionary containing time arrays and response waveforms
    """
    logger.info("Collecting response waveforms...")
    
    waveforms = {}
    
    # Step response
    step_result = specimen.step_response_protocol(**STEP_RESPONSE_PARAMS)
    waveforms['step_time'] = step_result['time']
    waveforms['step_response'] = step_result['response']
    
    # Paired-pulse (collect all three delays)
    pp_responses = []
    for delay in PAIRED_PULSE_PARAMS['delays']:
        pp_result = specimen.paired_pulse_protocol(
            voltage=PAIRED_PULSE_PARAMS['voltage'],
            pulse_width=PAIRED_PULSE_PARAMS['pulse_width'],
            probe_distance=PAIRED_PULSE_PARAMS['probe_distance'],
            delays=[delay]
        )
        # Run single delay and collect waveform
        # We need to re-run to get the waveform
        center = (specimen.area_size / 2, specimen.area_size / 2)
        probe = (specimen.area_size / 2 + PAIRED_PULSE_PARAMS['probe_distance'], specimen.area_size / 2)
        
        pulse_start = 100.0
        pulse_width = PAIRED_PULSE_PARAMS['pulse_width']
        voltage = PAIRED_PULSE_PARAMS['voltage']
        coupling_map = specimen.calculate_stimulation_coupling(center, voltage)
        
        def stim(t: float) -> np.ndarray:
            if pulse_start < t < (pulse_start + pulse_width):
                return coupling_map
            if (pulse_start + pulse_width + delay) < t < (pulse_start + 2 * pulse_width + delay):
                return coupling_map
            return np.zeros(specimen.num_nodes)
        
        sim_time = pulse_start + 2 * pulse_width + delay + 1000.0
        t, sol = specimen.run_experiment_custom_stim(sim_time, stim)
        v_out = np.array([specimen.read_output_voltage(probe, sol[i, :]) for i in range(len(sol))])
        
        pp_responses.append({
            'time': t,
            'response': v_out,
            'delay': delay
        })
    
    waveforms['paired_pulse'] = pp_responses
    
    # Triangle sweep
    tri_result = specimen.triangle_sweep_protocol(**TRIANGLE_SWEEP_PARAMS)
    waveforms['triangle_time'] = tri_result['time']
    waveforms['triangle_voltage_applied'] = tri_result['voltage_applied']
    waveforms['triangle_response'] = tri_result['voltage_response']
    
    logger.info("Waveforms collected")
    
    return waveforms

# ==========================================
# Parameter Prediction
# ==========================================

def predict_parameters(features: Dict, models: Dict, scaler: object, 
                      model_type: str = 'random_forest') -> Tuple[Dict, Dict]:
    """Predict fungal parameters from response features.
    
    Args:
        features: Dictionary of response features
        models: Dictionary of trained models
        scaler: Scaler for features (used with MLP)
        model_type: Type of model ('random_forest' or 'mlp')
        
    Returns:
        Tuple of (predicted_params, prediction_metadata)
    """
    logger.info(f"Predicting parameters using {model_type} models...")
    
    # Prepare feature vector
    feature_vector = []
    for feat_name in FEATURE_COLUMNS:
        if feat_name in features:
            feature_vector.append(features[feat_name])
        else:
            logger.warning(f"Feature {feat_name} not found, using 0.0")
            feature_vector.append(0.0)
    
    X = np.array(feature_vector).reshape(1, -1)
    
    # Scale if using MLP
    if model_type == 'mlp' and scaler is not None:
        X = scaler.transform(X)
    
    # Predict each parameter
    predicted_params = {}
    for param in FUNGAL_PARAMS:
        if param in models:
            pred = models[param].predict(X)[0]
            predicted_params[param] = pred
            logger.debug(f"  {param}: {pred:.4f}")
        else:
            logger.warning(f"No model for {param}, skipping")
    
    # Also predict network properties for context
    network_props = {}
    for prop in ['num_nodes', 'num_edges', 'network_density']:
        if prop in models:
            network_props[prop] = models[prop].predict(X)[0]
    
    logger.info(f"Predicted: tau_v={predicted_params.get('tau_v', 0):.1f}, "
                f"tau_w={predicted_params.get('tau_w', 0):.1f}, "
                f"a={predicted_params.get('a', 0):.2f}, "
                f"b={predicted_params.get('b', 0):.2f}")
    
    metadata = {
        'model_type': model_type,
        'network_predictions': network_props,
        'feature_vector': feature_vector
    }
    
    return predicted_params, metadata

# ==========================================
# Twin Creation and Refinement
# ==========================================

def create_twin(specimen: RealisticFungalComputer, predicted_params: Dict,
                network_predictions: Dict = None, use_inferred_network: bool = False,
                network_seed: int = None) -> RealisticFungalComputer:
    """Create a Twin with predicted parameters.
    
    Args:
        specimen: Original specimen
        predicted_params: Predicted fungal parameters
        network_predictions: Predicted network properties (num_nodes, num_edges, density)
        use_inferred_network: If True, use ML-inferred network instead of copying specimen
        network_seed: Random seed for network generation (if using inferred network)
        
    Returns:
        Twin RealisticFungalComputer instance
    """
    if use_inferred_network:
        logger.info("Creating Twin with ML-inferred network structure...")
        
        if network_predictions is None:
            raise ValueError("network_predictions required when use_inferred_network=True")
        
        # Use predicted network properties
        predicted_num_nodes = int(round(network_predictions.get('num_nodes', specimen.num_nodes)))
        
        # Ensure reasonable bounds
        predicted_num_nodes = max(20, min(100, predicted_num_nodes))
        
        if network_seed is None:
            network_seed = np.random.randint(0, 1000000)
        
        logger.info(f"  Predicted num_nodes: {predicted_num_nodes} (true: {specimen.num_nodes})")
        logger.info(f"  Using network seed: {network_seed}")
        
        # Create twin with new random network
        twin = RealisticFungalComputer(
            num_nodes=predicted_num_nodes,
            area_size=specimen.area_size,
            random_seed=network_seed
        )
    else:
        logger.info("Creating Twin with exact network structure (copied from specimen)...")
        
        # Create twin with same network structure
        twin = RealisticFungalComputer(
            num_nodes=specimen.num_nodes,
            area_size=specimen.area_size,
            random_seed=42  # Use a fixed seed to get reproducible network
        )
        
        # Copy the exact network structure from specimen
        twin.G = specimen.G.copy()
        twin.pos = specimen.pos.copy()
        twin.node_coords = specimen.node_coords.copy()
        twin.edge_list = specimen.edge_list.copy()
        twin.adj_matrix = specimen.adj_matrix.copy()
    
    # Apply predicted parameters
    twin.tau_v = predicted_params['tau_v']
    twin.tau_w = predicted_params['tau_w']
    twin.a = predicted_params['a']
    twin.b = predicted_params['b']
    twin.v_scale = predicted_params['v_scale']
    twin.R_off = predicted_params['R_off']
    twin.R_on = predicted_params['R_on']
    twin.alpha = predicted_params['alpha']
    
    logger.info("Twin created with predicted parameters")
    
    return twin

def compute_waveform_mismatch(specimen_waveforms: Dict, twin_waveforms: Dict) -> float:
    """Compute mismatch between specimen and twin response waveforms.
    
    Args:
        specimen_waveforms: Waveforms from specimen
        twin_waveforms: Waveforms from twin
        
    Returns:
        Total mismatch score (lower is better)
    """
    total_mismatch = 0.0
    
    # Step response mismatch
    step_mismatch = mean_squared_error(
        specimen_waveforms['step_response'],
        twin_waveforms['step_response']
    )
    total_mismatch += step_mismatch
    
    # Paired-pulse mismatch (all delays)
    for spec_pp, twin_pp in zip(specimen_waveforms['paired_pulse'], 
                                 twin_waveforms['paired_pulse']):
        # Ensure same length
        min_len = min(len(spec_pp['response']), len(twin_pp['response']))
        pp_mismatch = mean_squared_error(
            spec_pp['response'][:min_len],
            twin_pp['response'][:min_len]
        )
        total_mismatch += pp_mismatch
    
    # Triangle sweep mismatch
    tri_mismatch = mean_squared_error(
        specimen_waveforms['triangle_response'],
        twin_waveforms['triangle_response']
    )
    total_mismatch += tri_mismatch
    
    return total_mismatch

def refine_parameters_optimization(specimen: RealisticFungalComputer,
                                   specimen_waveforms: Dict,
                                   initial_params: Dict,
                                   residual_estimates: Dict = None,
                                   method: str = 'dual_annealing',
                                   use_full_bounds: bool = True) -> Tuple[Dict, Dict]:
    """Refine parameters using optimization to match waveforms.
    
    Args:
        specimen: Original specimen
        specimen_waveforms: Response waveforms from specimen
        initial_params: Initial parameter estimates from ML models
        residual_estimates: Estimated residuals for local search bounds (ignored if use_full_bounds=True)
        method: Optimization method ('dual_annealing', 'multi_start', 'shgo', 'basinhopping', 
                'differential_evolution', 'gp_minimize', 'L-BFGS-B')
        use_full_bounds: If True, use full parameter bounds instead of local search window
        
    Returns:
        Tuple of (refined_params, optimization_info)
    """
    logger.info(f"Refining parameters using {method} optimization...")
    
    # Define bounds for optimization
    bounds = []
    param_order = []
    
    if use_full_bounds:
        logger.info("Using FULL parameter bounds (global search)")
        for param in FUNGAL_PARAMS:
            bounds.append(PARAM_BOUNDS[param])
            param_order.append(param)
            logger.debug(f"  {param}: [{PARAM_BOUNDS[param][0]:.4f}, {PARAM_BOUNDS[param][1]:.4f}]")
    else:
        logger.info("Using LOCAL search bounds around ML prediction")
        # Set up local search bounds around initial estimates
        if residual_estimates is None:
            # Use default: ±20% of parameter range
            search_widths = {}
            for param in FUNGAL_PARAMS:
                low, high = PARAM_BOUNDS[param]
                search_widths[param] = 0.2 * (high - low)
        else:
            search_widths = residual_estimates
        
        for param in FUNGAL_PARAMS:
            initial_val = initial_params[param]
            width = search_widths.get(param, 0.2 * (PARAM_BOUNDS[param][1] - PARAM_BOUNDS[param][0]))
            
            # Local bounds around initial estimate
            lower = max(PARAM_BOUNDS[param][0], initial_val - width)
            upper = min(PARAM_BOUNDS[param][1], initial_val + width)
            
            bounds.append((lower, upper))
            param_order.append(param)
            logger.debug(f"  {param}: [{lower:.4f}, {upper:.4f}] (initial: {initial_val:.4f})")
    
    # Objective function
    eval_count = [0]
    best_mismatch = [np.inf]
    
    def objective(param_vector):
        eval_count[0] += 1
        
        # Create parameter dict
        params = {param: param_vector[i] for i, param in enumerate(param_order)}
        
        # Apply safety constraints
        if params['R_off'] < 1.5 * params['R_on']:
            return 1e10  # Penalty
        if params['b'] < params['a']:
            return 1e10  # Penalty
        
        try:
            # Create twin with these parameters
            twin = create_twin(specimen, params)
            
            # Collect twin waveforms
            twin_waveforms = collect_response_waveforms(twin)
            
            # Compute mismatch
            mismatch = compute_waveform_mismatch(specimen_waveforms, twin_waveforms)
            
            if mismatch < best_mismatch[0]:
                best_mismatch[0] = mismatch
                if eval_count[0] % 10 == 0:
                    logger.info(f"  Eval {eval_count[0]}: mismatch={mismatch:.6f}")
            
            return mismatch
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 1e10
    
    # Run optimization
    initial_vector = [initial_params[param] for param in param_order]
    
    logger.info(f"Starting optimization with {method}...")
    
    if method == 'dual_annealing':
        # Dual annealing: Fast global optimizer using simulated annealing + local search
        # Typically 2-3x faster than differential evolution
        result = dual_annealing(
            objective,
            bounds,
            maxiter=300,  # Max iterations for global search
            initial_temp=5230.0,  # Initial temperature
            restart_temp_ratio=2e-5,  # Temperature restart ratio
            visit=2.62,  # Visiting distribution parameter
            accept=-5.0,  # Acceptance parameter
            seed=42,
            no_local_search=False  # Enable local search for refinement
        )
    elif method == 'multi_start':
        # Multi-start L-BFGS-B: Run from multiple random starting points
        # Fastest approach for parallel exploration
        logger.info("Running multi-start L-BFGS-B from 20 random starting points...")
        
        n_starts = 200
        rng = np.random.RandomState(42)
        best_result = None
        best_fun = np.inf
        
        for i in range(n_starts):
            # Generate random starting point within bounds
            x0 = []
            for j, param in enumerate(param_order):
                low, high = bounds[j]
                if param == 'alpha':
                    # Log-uniform for alpha
                    x0.append(10 ** rng.uniform(np.log10(low), np.log10(high)))
                else:
                    x0.append(rng.uniform(low, high))
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_result = res
                    logger.info(f"  Start {i+1}/{n_starts}: New best mismatch={res.fun:.6f}")
                elif (i + 1) % 5 == 0:
                    logger.info(f"  Start {i+1}/{n_starts}: mismatch={res.fun:.6f}, best={best_fun:.6f}")
            except Exception as e:
                logger.warning(f"  Start {i+1} failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All multi-start attempts failed")
        
        result = best_result
        logger.info(f"Multi-start complete: best_mismatch={best_fun:.6f}")
        
    elif method == 'shgo':
        # SHGO: Simplicial Homology Global Optimization
        # Deterministic global optimizer, good for smooth landscapes
        result = shgo(
            objective,
            bounds,
            n=100,  # Number of sampling points
            iters=3,  # Number of iterations
            sampling_method='sobol',  # Use Sobol sequence for better coverage
            options={'ftol': 1e-8}
        )
    elif method == 'basinhopping':
        # Basin hopping: Random perturbation + local minimization
        # Good for rugged landscapes
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds,
            'options': {'ftol': 1e-8}
        }
        result = basinhopping(
            objective,
            initial_vector,
            minimizer_kwargs=minimizer_kwargs,
            niter=50,  # Number of basin hopping iterations
            T=1.0,  # Temperature for acceptance
            stepsize=0.5,  # Step size for random displacement
            seed=42
        )
    elif method == 'differential_evolution':
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42,
            workers=1,
            updating='deferred',
            disp=True
        )
    elif method == 'gp_minimize':
        # Gaussian Process minimization: Bayesian optimization
        # Efficient for expensive objective functions, builds surrogate model
        logger.info("Running Gaussian Process optimization (Bayesian optimization)...")
        
        # Convert bounds to skopt format
        space = [Real(low, high, name=param) for (low, high), param in zip(bounds, param_order)]
        
        result = gp_minimize(
            objective,
            space,
            n_calls=100,  # Number of function evaluations
            n_initial_points=40,  # Number of random initial points
            initial_point_generator='lhs',  # Latin hypercube sampling
            acq_func='EI',  # Expected Improvement acquisition function
            acq_optimizer='lbfgs',  # Optimizer for acquisition function
            x0=initial_vector,  # Start with ML prediction
            random_state=42,
            verbose=True,
            n_jobs=1
        )
    elif method in ['nelder-mead', 'powell', 'BFGS', 'L-BFGS-B']:
        result = minimize(
            objective,
            initial_vector,
            method=method,
            bounds=bounds if method == 'L-BFGS-B' else None,
            options={'maxiter': 100, 'disp': True}
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Extract refined parameters
    refined_params = {param: result.x[i] for i, param in enumerate(param_order)}
    
    # Handle different result object structures
    success = result.success if hasattr(result, 'success') else True
    message = result.message if hasattr(result, 'message') else 'N/A'
    
    optimization_info = {
        'method': method,
        'success': success,
        'final_mismatch': result.fun,
        'n_evaluations': eval_count[0],
        'message': message
    }
    
    logger.info(f"Optimization complete: success={success}, "
                f"final_mismatch={result.fun:.6f}, n_evals={eval_count[0]}")
    logger.info(f"Refined: tau_v={refined_params['tau_v']:.1f}, "
                f"tau_w={refined_params['tau_w']:.1f}, "
                f"a={refined_params['a']:.2f}, "
                f"b={refined_params['b']:.2f}")
    
    return refined_params, optimization_info

def load_ml_model_rmse(model_dir: Path = Path("ml_models")) -> Dict[str, float]:
    """Load RMSE values for each parameter from ML model evaluation results.
    
    Args:
        model_dir: Directory containing evaluation results
        
    Returns:
        Dictionary mapping parameter names to RMSE values
    """
    # Find most recent evaluation results
    eval_files = list(model_dir.glob("evaluation_results_*.csv"))
    if not eval_files:
        logger.warning("No evaluation results found, using default RMSE estimates")
        # Default RMSE estimates based on parameter ranges
        return {
            'tau_v': 15.0,
            'tau_w': 200.0,
            'a': 0.05,
            'b': 0.05,
            'v_scale': 2.7,
            'R_off': 75.0,
            'R_on': 15.0,
            'alpha': 0.005
        }
    
    # Load most recent file
    latest_file = max(eval_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading RMSE values from: {latest_file.name}")
    
    import pandas as pd
    df = pd.read_csv(latest_file)
    
    # Extract RMSE for each parameter (prefer random_forest, fallback to mlp)
    rmse_dict = {}
    for param in FUNGAL_PARAMS:
        # Try random_forest first
        rf_row = df[(df['model'] == 'random_forest') & (df['parameter'] == param)]
        if not rf_row.empty:
            rmse_dict[param] = float(rf_row['rmse'].values[0])
        else:
            # Fallback to mlp
            mlp_row = df[(df['model'] == 'mlp') & (df['parameter'] == param)]
            if not mlp_row.empty:
                rmse_dict[param] = float(mlp_row['rmse'].values[0])
            else:
                logger.warning(f"No RMSE found for {param}, using default")
                rmse_dict[param] = 0.1 * (PARAM_BOUNDS[param][1] - PARAM_BOUNDS[param][0])
    
    logger.info("RMSE values loaded:")
    for param, rmse in rmse_dict.items():
        logger.info(f"  {param}: {rmse:.4f}")
    
    return rmse_dict

def generate_warm_start_population(initial_params: Dict, 
                                   rmse_dict: Dict[str, float],
                                   population_size: int = 20,
                                   diversity_factor: float = 2.0) -> List[Dict]:
    """Generate a diverse initial population around ML predictions using RMSE.
    
    Strategy:
    - Use ML predictions as the mean
    - Use RMSE as a measure of uncertainty to set variance
    - Sample from truncated normal distributions to respect bounds
    - Add diversity by scaling RMSE by diversity_factor
    
    Args:
        initial_params: ML-predicted parameters (population mean)
        rmse_dict: RMSE values for each parameter from ML models
        population_size: Number of individuals in population
        diversity_factor: Multiplier for RMSE to control diversity (higher = more diverse)
        
    Returns:
        List of parameter dictionaries representing the initial population
    """
    logger.info(f"Generating warm-start population: size={population_size}, diversity_factor={diversity_factor}")
    
    rng = np.random.RandomState(42)
    population = []
    
    # First individual is the ML prediction itself
    population.append(initial_params.copy())
    
    # Generate remaining individuals
    for i in range(population_size - 1):
        individual = {}
        
        for param in FUNGAL_PARAMS:
            mean = initial_params[param]
            std = rmse_dict[param] * diversity_factor
            lower, upper = PARAM_BOUNDS[param]
            
            # Sample from truncated normal distribution
            # Use rejection sampling for simplicity
            max_attempts = 100
            for attempt in range(max_attempts):
                if param == 'alpha':
                    # Log-normal for alpha
                    log_mean = np.log(mean)
                    log_std = std / mean  # Approximate log-space std
                    value = np.exp(rng.normal(log_mean, log_std))
                else:
                    value = rng.normal(mean, std)
                
                # Check bounds
                if lower <= value <= upper:
                    individual[param] = value
                    break
            else:
                # If rejection sampling fails, use uniform within bounds
                if param == 'alpha':
                    individual[param] = 10 ** rng.uniform(np.log10(lower), np.log10(upper))
                else:
                    individual[param] = rng.uniform(lower, upper)
        
        # Apply safety constraints
        if individual['R_off'] < 1.5 * individual['R_on']:
            individual['R_off'] = 1.5 * individual['R_on']
        if individual['b'] < individual['a']:
            individual['b'] = individual['a'] + 0.05
        
        population.append(individual)
    
    logger.info(f"Generated population with {len(population)} individuals")
    
    # Log population statistics
    for param in FUNGAL_PARAMS:
        values = [ind[param] for ind in population]
        logger.debug(f"  {param}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                    f"range=[{np.min(values):.4f}, {np.max(values):.4f}]")
    
    return population

def hierarchical_optimization_with_network(specimen: RealisticFungalComputer,
                                           specimen_waveforms: Dict,
                                           initial_params: Dict,
                                           network_predictions: Dict,
                                           method: str = 'gp_minimize',
                                           use_full_bounds: bool = True) -> Tuple[Dict, int, Dict]:
    """Hierarchical optimization that searches both network structure and biophysical parameters.
    
    Strategy:
    1. Stage 1: Optimize network seed (discrete search over multiple random networks)
               with fixed biophysical parameters (ML predictions)
    2. Warm-Start: Generate diverse initial population around ML predictions using RMSE
    3. Stage 2: Optimize biophysical parameters with the best network from Stage 1
                using Gaussian Process optimization (or other methods)
    
    This approach combines:
    - ML predictions as intelligent initialization (warm-start)
    - RMSE-based population diversity for exploration
    - Gaussian Process surrogate modeling for efficient refinement
    - Hierarchical search to avoid ill-posed joint optimization
    
    Args:
        specimen: Original specimen
        specimen_waveforms: Response waveforms from specimen
        initial_params: Initial parameter estimates from ML models
        network_predictions: Predicted network properties
        method: Optimization method for Stage 2:
                - 'gp_minimize' or 'gaussian_process': Gaussian Process with warm-start (recommended)
                - 'dual_annealing': Simulated annealing global optimizer
                - 'multi_start': Multi-start L-BFGS-B local search
                - 'shgo': Simplicial homology global optimization
                - 'basinhopping': Basin hopping with local refinement
                - 'differential_evolution': Differential evolution
        use_full_bounds: If True, use full parameter bounds instead of local search window
        
    Returns:
        Tuple of (refined_params, best_network_seed, optimization_info)
    """
    logger.info("="*70)
    logger.info("HIERARCHICAL OPTIMIZATION: Network + Biophysical Parameters")
    logger.info("="*70)
    
    # ==========================================
    # STAGE 1: Network Structure Optimization
    # ==========================================
    logger.info("\nSTAGE 1: Optimizing Network Structure")
    logger.info("="*70)
    logger.info("Strategy: Adaptive Coarse-to-Fine Network Search")
    logger.info("  Phase 1: Wide exploration (100 samples)")
    logger.info("  Phase 2: Focused refinement (150 samples)")
    logger.info("  Phase 3: Local intensification (150 samples)")
    logger.info("  Phase 4: Final refinement (100 samples)")
    logger.info("="*70)
    
    predicted_num_nodes = int(round(network_predictions.get('num_nodes', specimen.num_nodes)))
    predicted_num_nodes = max(20, min(100, predicted_num_nodes))
    
    rng = np.random.RandomState(42)
    network_results = []
    all_network_configs = []
    
    best_network_seed = None
    best_network_config = None
    best_network_mismatch = np.inf
    
    # Helper function to evaluate a network configuration
    def evaluate_network(config, phase_name, sample_idx, total_samples):
        try:
            temp_network_pred = network_predictions.copy()
            temp_network_pred['num_nodes'] = config['num_nodes']
            
            twin = create_twin(
                specimen, 
                initial_params,
                network_predictions=temp_network_pred,
                use_inferred_network=True,
                network_seed=config['seed']
            )
            
            twin_waveforms = collect_response_waveforms(twin)
            mismatch = compute_waveform_mismatch(specimen_waveforms, twin_waveforms)
            
            result = {
                'seed': config['seed'],
                'num_nodes': config['num_nodes'],
                'actual_num_nodes': twin.num_nodes,
                'num_edges': len(twin.edge_list),
                'mismatch': mismatch,
                'phase': phase_name
            }
            
            return result, mismatch
            
        except Exception as e:
            logger.warning(f"  {phase_name} sample {sample_idx}/{total_samples} failed: {e}")
            return None, np.inf
    
    # ==========================================
    # PHASE 1: Wide Exploration
    # ==========================================
    logger.info("\n" + "-"*70)
    logger.info("PHASE 1: Wide Exploration")
    logger.info("-"*70)
    
    # Initial wide range: ±50% around ML prediction
    phase1_samples = 100
    node_count_jitter = max(20, int(0.5 * predicted_num_nodes))
    min_nodes_p1 = max(10, predicted_num_nodes - node_count_jitter)
    max_nodes_p1 = min(120, predicted_num_nodes + node_count_jitter)
    
    logger.info(f"Sampling {phase1_samples} networks uniformly across [{min_nodes_p1}, {max_nodes_p1}]")
    logger.info(f"ML predicted nodes: {predicted_num_nodes}, True nodes: {specimen.num_nodes}")
    
    phase1_configs = []
    for i in range(phase1_samples):
        num_nodes = rng.randint(min_nodes_p1, max_nodes_p1 + 1)
        seed = 42 + len(all_network_configs) * 1000
        phase1_configs.append({'num_nodes': num_nodes, 'seed': seed})
        all_network_configs.append({'num_nodes': num_nodes, 'seed': seed})
    
    for i, config in enumerate(phase1_configs):
        result, mismatch = evaluate_network(config, "Phase 1", i+1, phase1_samples)
        
        if result is not None:
            network_results.append(result)
            
            if mismatch < best_network_mismatch:
                best_network_mismatch = mismatch
                best_network_seed = config['seed']
                best_network_config = config
                logger.info(f"  Sample {i+1}/{phase1_samples}: nodes={config['num_nodes']}, "
                          f"mismatch={mismatch:.6f} *** NEW BEST ***")
            elif (i + 1) % 20 == 0:
                logger.info(f"  Sample {i+1}/{phase1_samples}: nodes={config['num_nodes']}, "
                          f"mismatch={mismatch:.6f}, best={best_network_mismatch:.6f}")
    
    # Analyze Phase 1 results to identify promising regions
    sorted_results = sorted(network_results, key=lambda x: x['mismatch'])
    top_10_percent = sorted_results[:max(10, len(sorted_results)//10)]
    top_node_counts = [r['num_nodes'] for r in top_10_percent]
    
    # Calculate center and range for Phase 2
    phase2_center = int(np.median(top_node_counts))
    phase2_std = int(np.std(top_node_counts)) if len(top_node_counts) > 1 else 10
    
    logger.info(f"\nPhase 1 Analysis:")
    logger.info(f"  Best mismatch: {best_network_mismatch:.6f} at {best_network_config['num_nodes']} nodes")
    logger.info(f"  Top 10% node counts: {sorted(set(top_node_counts))}")
    logger.info(f"  Median of top performers: {phase2_center}")
    logger.info(f"  Std of top performers: {phase2_std}")
    
    # ==========================================
    # PHASE 2: Focused Refinement
    # ==========================================
    logger.info("\n" + "-"*70)
    logger.info("PHASE 2: Focused Refinement")
    logger.info("-"*70)
    
    # Narrow to ±30% around top performers
    phase2_samples = 100
    phase2_range = max(10, int(0.3 * phase2_std + 0.15 * phase2_center))
    min_nodes_p2 = max(10, phase2_center - phase2_range)
    max_nodes_p2 = min(120, phase2_center + phase2_range)
    
    logger.info(f"Sampling {phase2_samples} networks in refined range [{min_nodes_p2}, {max_nodes_p2}]")
    logger.info(f"Focused around {phase2_center} nodes (median of Phase 1 top performers)")
    
    phase2_configs = []
    for i in range(phase2_samples):
        num_nodes = rng.randint(min_nodes_p2, max_nodes_p2 + 1)
        seed = 42 + len(all_network_configs) * 1000
        phase2_configs.append({'num_nodes': num_nodes, 'seed': seed})
        all_network_configs.append({'num_nodes': num_nodes, 'seed': seed})
    
    for i, config in enumerate(phase2_configs):
        result, mismatch = evaluate_network(config, "Phase 2", i+1, phase2_samples)
        
        if result is not None:
            network_results.append(result)
            
            if mismatch < best_network_mismatch:
                best_network_mismatch = mismatch
                best_network_seed = config['seed']
                best_network_config = config
                logger.info(f"  Sample {i+1}/{phase2_samples}: nodes={config['num_nodes']}, "
                          f"mismatch={mismatch:.6f} *** NEW BEST ***")
            elif (i + 1) % 30 == 0:
                logger.info(f"  Sample {i+1}/{phase2_samples}: nodes={config['num_nodes']}, "
                          f"mismatch={mismatch:.6f}, best={best_network_mismatch:.6f}")
    
    # Analyze Phase 2 results
    sorted_results = sorted(network_results, key=lambda x: x['mismatch'])
    top_5_percent = sorted_results[:max(10, len(sorted_results)//20)]
    top_node_counts = [r['num_nodes'] for r in top_5_percent]
    
    phase3_center = int(np.median(top_node_counts))
    phase3_std = int(np.std(top_node_counts)) if len(top_node_counts) > 1 else 5
    
    logger.info(f"\nPhase 2 Analysis:")
    logger.info(f"  Best mismatch: {best_network_mismatch:.6f} at {best_network_config['num_nodes']} nodes")
    logger.info(f"  Top 5% node counts: {sorted(set(top_node_counts))}")
    logger.info(f"  Median of top performers: {phase3_center}")
    
    # ==========================================
    # PHASE 3: Local Intensification
    # ==========================================
    logger.info("\n" + "-"*70)
    logger.info("PHASE 3: Local Intensification")
    logger.info("-"*70)
    
    # Very narrow range: ±15% with multiple seeds per node count
    phase3_samples = 100
    phase3_range = max(5, int(0.15 * phase3_center))
    min_nodes_p3 = max(10, phase3_center - phase3_range)
    max_nodes_p3 = min(120, phase3_center + phase3_range)
    
    logger.info(f"Sampling {phase3_samples} networks in tight range [{min_nodes_p3}, {max_nodes_p3}]")
    logger.info(f"Multiple network geometries per node count for diversity")
    
    # Generate diverse seeds for geometric variation
    node_count_range = list(range(min_nodes_p3, max_nodes_p3 + 1))
    phase3_configs = []
    
    for i in range(phase3_samples):
        # Bias towards best node count but sample others too
        if i % 3 == 0 and best_network_config['num_nodes'] in node_count_range:
            num_nodes = best_network_config['num_nodes']
        else:
            num_nodes = rng.choice(node_count_range)
        
        seed = 42 + len(all_network_configs) * 1000
        phase3_configs.append({'num_nodes': num_nodes, 'seed': seed})
        all_network_configs.append({'num_nodes': num_nodes, 'seed': seed})
    
    for i, config in enumerate(phase3_configs):
        result, mismatch = evaluate_network(config, "Phase 3", i+1, phase3_samples)
        
        if result is not None:
            network_results.append(result)
            
            if mismatch < best_network_mismatch:
                best_network_mismatch = mismatch
                best_network_seed = config['seed']
                best_network_config = config
                logger.info(f"  Sample {i+1}/{phase3_samples}: nodes={config['num_nodes']}, "
                          f"mismatch={mismatch:.6f} *** NEW BEST ***")
            elif (i + 1) % 30 == 0:
                logger.info(f"  Sample {i+1}/{phase3_samples}: nodes={config['num_nodes']}, "
                          f"mismatch={mismatch:.6f}, best={best_network_mismatch:.6f}")
    
    # Analyze Phase 3 results
    sorted_results = sorted(network_results, key=lambda x: x['mismatch'])
    top_3_percent = sorted_results[:max(10, len(sorted_results)//33)]
    top_node_counts = [r['num_nodes'] for r in top_3_percent]
    
    phase4_center = int(np.median(top_node_counts))
    
    logger.info(f"\nPhase 3 Analysis:")
    logger.info(f"  Best mismatch: {best_network_mismatch:.6f} at {best_network_config['num_nodes']} nodes")
    logger.info(f"  Top 3% node counts: {sorted(set(top_node_counts))}")
    logger.info(f"  Median of top performers: {phase4_center}")
    
    # ==========================================
    # PHASE 4: Final Refinement
    # ==========================================
    logger.info("\n" + "-"*70)
    logger.info("PHASE 4: Final Refinement")
    logger.info("-"*70)
    
    # Ultra-tight range: ±10% with maximum geometric diversity
    phase4_samples = 100
    phase4_range = max(3, int(0.1 * phase4_center))
    min_nodes_p4 = max(10, phase4_center - phase4_range)
    max_nodes_p4 = min(120, phase4_center + phase4_range)
    
    logger.info(f"Sampling {phase4_samples} networks in final range [{min_nodes_p4}, {max_nodes_p4}]")
    logger.info(f"Maximum geometric diversity at optimal node count")
    
    node_count_range = list(range(min_nodes_p4, max_nodes_p4 + 1))
    phase4_configs = []
    
    for i in range(phase4_samples):
        # Heavy bias towards best node count (50% of samples)
        if i % 2 == 0 and best_network_config['num_nodes'] in node_count_range:
            num_nodes = best_network_config['num_nodes']
        else:
            num_nodes = rng.choice(node_count_range)
        
        seed = 42 + len(all_network_configs) * 1000
        phase4_configs.append({'num_nodes': num_nodes, 'seed': seed})
        all_network_configs.append({'num_nodes': num_nodes, 'seed': seed})
    
    for i, config in enumerate(phase4_configs):
        result, mismatch = evaluate_network(config, "Phase 4", i+1, phase4_samples)
        
        if result is not None:
            network_results.append(result)
            
            if mismatch < best_network_mismatch:
                best_network_mismatch = mismatch
                best_network_seed = config['seed']
                best_network_config = config
                logger.info(f"  Sample {i+1}/{phase4_samples}: nodes={config['num_nodes']}, "
                          f"mismatch={mismatch:.6f} *** NEW BEST ***")
            elif (i + 1) % 20 == 0:
                logger.info(f"  Sample {i+1}/{phase4_samples}: nodes={config['num_nodes']}, "
                          f"mismatch={mismatch:.6f}, best={best_network_mismatch:.6f}")
    
    # ==========================================
    # Stage 1 Summary
    # ==========================================
    if best_network_seed is None:
        raise RuntimeError("All network samples failed")
    
    logger.info("\n" + "="*70)
    logger.info("STAGE 1 COMPLETE: Adaptive Network Search Summary")
    logger.info("="*70)
    logger.info(f"Total samples evaluated: {len(network_results)}")
    logger.info(f"Best network: nodes={best_network_config['num_nodes']}, seed={best_network_seed}")
    logger.info(f"Best mismatch: {best_network_mismatch:.6f}")
    
    # Show progression through phases
    phase_best = {}
    for phase_name in ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']:
        phase_results = [r for r in network_results if r['phase'] == phase_name]
        if phase_results:
            best_in_phase = min(phase_results, key=lambda x: x['mismatch'])
            phase_best[phase_name] = best_in_phase['mismatch']
    
    logger.info(f"\nProgression through phases:")
    for phase_name, mismatch in phase_best.items():
        logger.info(f"  {phase_name}: best mismatch = {mismatch:.6f}")
    
    if len(network_results) > 0:
        first_mismatch = network_results[0]['mismatch']
        improvement = (1 - best_network_mismatch/first_mismatch) * 100
        logger.info(f"\nImprovement over first sample: {improvement:.1f}%")
    
    # Node count distribution analysis
    node_count_hist = {}
    for r in network_results:
        nc = r['num_nodes']
        if nc not in node_count_hist:
            node_count_hist[nc] = []
        node_count_hist[nc].append(r['mismatch'])
    
    logger.info(f"\nNode count analysis (top 5 by average mismatch):")
    avg_by_node_count = {nc: np.mean(mismatches) for nc, mismatches in node_count_hist.items()}
    top_node_counts = sorted(avg_by_node_count.items(), key=lambda x: x[1])[:5]
    for nc, avg_mismatch in top_node_counts:
        count = len(node_count_hist[nc])
        best_at_nc = min(node_count_hist[nc])
        logger.info(f"  {nc} nodes: avg={avg_mismatch:.6f}, best={best_at_nc:.6f}, samples={count}")
    
    # ==========================================
    # STAGE 2: Biophysical Parameter Optimization
    # ==========================================
    logger.info("\nSTAGE 2: Optimizing Biophysical Parameters")
    logger.info("-"*70)
    logger.info(f"Using best network (nodes={best_network_config['num_nodes']}, seed={best_network_seed}) from Stage 1")
    logger.info("Refining biophysical parameters to minimize waveform mismatch...")
    
    # Set up bounds for parameter optimization
    bounds = []
    param_order = []
    
    if use_full_bounds:
        logger.info("\nParameter search bounds: FULL BOUNDS (global search)")
        logger.info(f"{'Parameter':<12} {'Initial':<12} {'Lower':<12} {'Upper':<12} {'Width':<12}")
        logger.info("-"*60)
        
        for param in FUNGAL_PARAMS:
            initial_val = initial_params[param]
            lower, upper = PARAM_BOUNDS[param]
            
            bounds.append((lower, upper))
            param_order.append(param)
            
            logger.info(f"{param:<12} {initial_val:<12.4f} {lower:<12.4f} {upper:<12.4f} {upper-lower:<12.4f}")
    else:
        logger.info("\nParameter search bounds: LOCAL SEARCH (±40% around ML prediction)")
        search_widths = {}
        for param in FUNGAL_PARAMS:
            low, high = PARAM_BOUNDS[param]
            search_widths[param] = 0.40 * (high - low)
        
        logger.info(f"{'Parameter':<12} {'Initial':<12} {'Lower':<12} {'Upper':<12} {'Width':<12}")
        logger.info("-"*60)
        
        for param in FUNGAL_PARAMS:
            initial_val = initial_params[param]
            width = search_widths[param]
            
            lower = max(PARAM_BOUNDS[param][0], initial_val - width)
            upper = min(PARAM_BOUNDS[param][1], initial_val + width)
            
            bounds.append((lower, upper))
            param_order.append(param)
            
            logger.info(f"{param:<12} {initial_val:<12.4f} {lower:<12.4f} {upper:<12.4f} {upper-lower:<12.4f}")
    
    # Objective function for Stage 2
    eval_count = [0]
    best_mismatch = [best_network_mismatch]
    
    def objective_stage2(param_vector):
        eval_count[0] += 1
        
        params = {param: param_vector[i] for i, param in enumerate(param_order)}
        
        # Apply safety constraints
        if params['R_off'] < 1.5 * params['R_on']:
            return 1e10
        if params['b'] < params['a']:
            return 1e10
        
        try:
            # Create twin with best network configuration and these parameters
            temp_network_pred = network_predictions.copy()
            temp_network_pred['num_nodes'] = best_network_config['num_nodes']
            
            twin = create_twin(
                specimen,
                params,
                network_predictions=temp_network_pred,
                use_inferred_network=True,
                network_seed=best_network_seed
            )
            
            twin_waveforms = collect_response_waveforms(twin)
            mismatch = compute_waveform_mismatch(specimen_waveforms, twin_waveforms)
            
            if mismatch < best_mismatch[0]:
                best_mismatch[0] = mismatch
                improvement = (1 - mismatch/best_network_mismatch) * 100
                logger.info(f"  Eval {eval_count[0]}: mismatch={mismatch:.6f} (improvement: {improvement:.1f}%)")
            elif eval_count[0] % 20 == 0:
                logger.info(f"  Eval {eval_count[0]}: current_mismatch={mismatch:.6f}, best={best_mismatch[0]:.6f}")
            
            return mismatch
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 1e10
    
    # ==========================================
    # Load RMSE and Generate Warm-Start Population
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("WARM-START POPULATION GENERATION")
    logger.info("="*70)
    
    # Load RMSE values from ML model evaluation
    rmse_dict = load_ml_model_rmse()
    
    # Generate warm-start population around ML predictions
    warm_start_population = generate_warm_start_population(
        initial_params=initial_params,
        rmse_dict=rmse_dict,
        population_size=20,
        diversity_factor=2.0
    )
    
    # Evaluate warm-start population to seed GP
    logger.info("\nEvaluating warm-start population...")
    x0_list = []
    y0_list = []
    
    for i, individual in enumerate(warm_start_population):
        param_vector = [individual[param] for param in param_order]
        mismatch = objective_stage2(param_vector)
        
        if mismatch < 1e9:  # Valid evaluation
            x0_list.append(param_vector)
            y0_list.append(mismatch)
            
            if mismatch < best_mismatch[0]:
                logger.info(f"  Individual {i+1}/{len(warm_start_population)}: mismatch={mismatch:.6f} *** NEW BEST ***")
            elif (i + 1) % 5 == 0:
                logger.info(f"  Individual {i+1}/{len(warm_start_population)}: mismatch={mismatch:.6f}")
    
    logger.info(f"\nWarm-start evaluation complete:")
    logger.info(f"  Valid evaluations: {len(x0_list)}/{len(warm_start_population)}")
    logger.info(f"  Best mismatch: {min(y0_list):.6f}")
    logger.info(f"  Mean mismatch: {np.mean(y0_list):.6f}")
    logger.info(f"  Std mismatch: {np.std(y0_list):.6f}")
    
    # ==========================================
    # Run Stage 2 Optimization
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("STAGE 2 OPTIMIZATION: Gaussian Process Refinement")
    logger.info("="*70)
    
    initial_vector = [initial_params[param] for param in param_order]
    
    if method == 'gp_minimize' or method == 'gaussian_process':
        logger.info("Using Gaussian Process optimization (scikit-optimize)...")
        logger.info(f"  Initial evaluations from warm-start: {len(x0_list)}")
        logger.info(f"  Additional GP iterations: 100")
        logger.info(f"  Total budget: {len(x0_list) + 100} evaluations")
        
        # Define search space for scikit-optimize
        space = []
        for param in param_order:
            lower, upper = bounds[param_order.index(param)]
            if param == 'alpha':
                # Log-uniform for alpha
                space.append(Real(lower, upper, prior='log-uniform', name=param))
            else:
                space.append(Real(lower, upper, name=param))
        
        # Run GP optimization with warm-start
        result = gp_minimize(
            objective_stage2,
            space,
            n_calls=100,  # Additional calls beyond warm-start
            n_initial_points=40,  # Don't generate random points, use warm-start only
            x0=x0_list,  # Warm-start population
            y0=y0_list,  # Corresponding objective values
            acq_func='EI',  # Expected Improvement acquisition function
            acq_optimizer='sampling',  # Use sampling for acquisition optimization
            n_points=10000,  # Number of points to sample when optimizing acquisition
            random_state=42,
            verbose=False,
            n_jobs=1
        )
        
        logger.info(f"\nGaussian Process optimization complete:")
        logger.info(f"  Total function evaluations: {len(x0_list) + 100}")
        logger.info(f"  Best mismatch found: {result.fun:.6f}")
        
    elif method == 'dual_annealing':
        logger.info("Using dual_annealing (fast global optimizer)...")
        result = dual_annealing(
            objective_stage2,
            bounds,
            maxiter=300,
            initial_temp=5230.0,
            restart_temp_ratio=2e-5,
            visit=2.62,
            accept=-5.0,
            seed=42,
            no_local_search=False
        )
    elif method == 'multi_start':
        logger.info("Using multi-start L-BFGS-B (parallel local search)...")
        n_starts = 200
        rng = np.random.RandomState(42)
        best_result = None
        best_fun = best_mismatch[0]
        
        for i in range(n_starts):
            x0 = []
            for j, param in enumerate(param_order):
                low, high = bounds[j]
                if param == 'alpha':
                    x0.append(10 ** rng.uniform(np.log10(low), np.log10(high)))
                else:
                    x0.append(rng.uniform(low, high))
            
            try:
                res = minimize(
                    objective_stage2,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_result = res
                    logger.info(f"  Start {i+1}/{n_starts}: New best={res.fun:.6f}")
                elif (i + 1) % 5 == 0:
                    logger.info(f"  Start {i+1}/{n_starts}: current={res.fun:.6f}, best={best_fun:.6f}")
            except Exception as e:
                logger.warning(f"  Start {i+1} failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All multi-start attempts failed")
        result = best_result
        
    elif method == 'shgo':
        logger.info("Using SHGO (simplicial homology global optimization)...")
        result = shgo(
            objective_stage2,
            bounds,
            n=100,
            iters=3,
            sampling_method='sobol',
            options={'ftol': 1e-8}
        )
    elif method == 'basinhopping':
        logger.info("Using basin hopping...")
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds,
            'options': {'ftol': 1e-8}
        }
        result = basinhopping(
            objective_stage2,
            initial_vector,
            minimizer_kwargs=minimizer_kwargs,
            niter=50,
            T=1.0,
            stepsize=0.5,
            seed=42
        )
    elif method == 'differential_evolution':
        logger.info("Using differential evolution...")
        result = differential_evolution(
            objective_stage2,
            bounds,
            maxiter=40,
            popsize=8,
            seed=42,
            workers=1,
            updating='deferred',
            disp=True
        )
    elif method in ['nelder-mead', 'powell', 'L-BFGS-B']:
        logger.info(f"Using {method}...")
        if method == 'L-BFGS-B':
            options = {
                'maxiter': 200,
                'ftol': 1e-8,
                'gtol': 1e-6,
                'disp': True,
                'maxfun': 300
            }
        else:
            options = {'maxiter': 150, 'disp': True}
        
        result = minimize(
            objective_stage2,
            initial_vector,
            method=method,
            bounds=bounds if method == 'L-BFGS-B' else None,
            options=options
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Extract refined parameters
    refined_params = {param: result.x[i] for i, param in enumerate(param_order)}
    
    # Calculate total network samples from all phases
    total_network_samples = len(network_results)
    
    optimization_info = {
        'method': 'hierarchical_' + method,
        'stage1_samples': total_network_samples,
        'stage1_best_network': best_network_config,
        'stage1_best_mismatch': float(best_network_mismatch),
        'stage2_success': getattr(result, 'success', True),  # GP result may not have success attribute
        'stage2_final_mismatch': float(result.fun),
        'stage2_n_evaluations': eval_count[0],
        'stage2_warm_start_size': len(warm_start_population),
        'stage2_warm_start_best': float(min(y0_list)) if y0_list else float('inf'),
        'total_improvement': float((1 - result.fun/network_results[0]['mismatch'])*100) if network_results else 0.0,
        'network_results': network_results[:10]  # Store top 10 for analysis
    }
    
    logger.info(f"\nStage 2 Complete:")
    logger.info(f"  Success: {optimization_info['stage2_success']}")
    logger.info(f"  Final mismatch: {result.fun:.6f}")
    logger.info(f"  Stage 2 improvement: {(1 - result.fun/best_network_mismatch)*100:.1f}%")
    logger.info(f"  Total improvement: {optimization_info['total_improvement']:.1f}%")
    logger.info(f"  Total evaluations: Stage 1={total_network_samples}, Stage 2={eval_count[0]}")
    
    logger.info(f"\nRefined parameters:")
    logger.info(f"  tau_v={refined_params['tau_v']:.1f}, tau_w={refined_params['tau_w']:.1f}")
    logger.info(f"  a={refined_params['a']:.2f}, b={refined_params['b']:.2f}")
    logger.info(f"  R_off={refined_params['R_off']:.1f}, R_on={refined_params['R_on']:.1f}")
    logger.info(f"  alpha={refined_params['alpha']:.4f}, v_scale={refined_params['v_scale']:.2f}")
    
    return refined_params, best_network_config, optimization_info

# ==========================================
# XOR Gate Validation
# ==========================================

def test_xor_gate_performance(env: RealisticFungalComputer, xor_params: Dict) -> Dict:
    """Test XOR gate performance using optimized parameters.
    
    Args:
        env: RealisticFungalComputer instance
        xor_params: Dictionary with electrode positions and stimulus parameters
        
    Returns:
        Dictionary with XOR gate test results
    """
    logger.info("Testing XOR gate performance...")
    
    # Log the fungal parameters being used
    logger.info(f"Environment parameters: tau_v={env.tau_v:.1f}, tau_w={env.tau_w:.1f}, "
                f"a={env.a:.3f}, b={env.b:.3f}, v_scale={env.v_scale:.2f}")
    logger.info(f"Memristor parameters: R_off={env.R_off:.1f}, R_on={env.R_on:.1f}, alpha={env.alpha:.4f}")
    logger.info(f"Network: {env.num_nodes} nodes, {len(env.edge_list)} edges")
    
    # Extract parameters
    electrode_A = (xor_params['x_A'], xor_params['y_A'])
    electrode_B = (xor_params['x_B'], xor_params['y_B'])
    output_probe = (xor_params['x_out'], xor_params['y_out'])
    voltage = xor_params['voltage']
    duration = xor_params['duration']
    delay = xor_params['delay']
    
    # Test all four input combinations: (0,0), (0,1), (1,0), (1,1)
    test_cases = [
        {'A': 0, 'B': 0, 'expected_xor': 0},
        {'A': 0, 'B': 1, 'expected_xor': 1},
        {'A': 1, 'B': 0, 'expected_xor': 1},
        {'A': 1, 'B': 1, 'expected_xor': 0}
    ]
    
    results = []
    output_voltages = []
    
    for test in test_cases:
        # Create stimulation function
        def stim_func(t: float) -> np.ndarray:
            stim = np.zeros(env.num_nodes)
            
            # Pulse start times
            pulse_start = 100.0
            
            # Input A (always starts at pulse_start if active)
            if test['A'] == 1:
                if pulse_start < t < (pulse_start + duration):
                    coupling_A = env.calculate_stimulation_coupling(electrode_A, voltage)
                    stim += coupling_A
            
            # Input B (starts at pulse_start + delay if active)
            # This matches how electrode_delays=[0.0, delay] works in run_experiment
            if test['B'] == 1:
                pulse_B_start = pulse_start + delay
                if pulse_B_start < t < (pulse_B_start + duration):
                    coupling_B = env.calculate_stimulation_coupling(electrode_B, voltage)
                    stim += coupling_B
            
            return stim
        
        # Run simulation
        sim_time = 100.0 + max(duration, duration + delay) + 1000.0
        t, sol = env.run_experiment_custom_stim(sim_time, stim_func)
        
        # Read output voltage
        v_out = np.array([env.read_output_voltage(output_probe, sol[i, :]) for i in range(len(sol))])
        
        # Measure output using MAX (same as optimize_xor_gate)
        # This is critical: optimization maximizes peak separation, so validation must also use peaks
        output_voltage = np.max(v_out)
        output_voltages.append(output_voltage)
        
        results.append({
            'A': test['A'],
            'B': test['B'],
            'expected_xor': test['expected_xor'],
            'output_voltage': output_voltage,
            'time': t,
            'voltage_trace': v_out
        })
    
    # Determine threshold (midpoint between low and high outputs)
    all_outputs = [r['output_voltage'] for r in results]
    threshold = (np.min(all_outputs) + np.max(all_outputs)) / 2.0
    
    # Classify outputs and compute accuracy
    correct = 0
    for i, result in enumerate(results):
        predicted_xor = 1 if result['output_voltage'] > threshold else 0
        result['predicted_xor'] = predicted_xor
        result['correct'] = (predicted_xor == result['expected_xor'])
        if result['correct']:
            correct += 1
    
    accuracy = correct / len(test_cases)
    
    # Compute XOR score (separation metric)
    xor_0_outputs = [r['output_voltage'] for r in results if r['expected_xor'] == 0]
    xor_1_outputs = [r['output_voltage'] for r in results if r['expected_xor'] == 1]
    
    if len(xor_0_outputs) > 0 and len(xor_1_outputs) > 0:
        separation = np.mean(xor_1_outputs) - np.mean(xor_0_outputs)
    else:
        separation = 0.0
    
    logger.info(f"  XOR Gate Accuracy: {accuracy*100:.1f}% ({correct}/{len(test_cases)})")
    logger.info(f"  Output separation: {separation:.4f}V")
    logger.info(f"  Threshold: {threshold:.4f}V")
    
    return {
        'test_cases': results,
        'accuracy': accuracy,
        'separation': separation,
        'threshold': threshold,
        'output_voltages': output_voltages
    }

def validate_xor_gate_on_specimen(twin: RealisticFungalComputer, 
                                   specimen: RealisticFungalComputer,
                                   xor_params: Dict) -> Dict:
    """Validate XOR gate by testing on both twin and specimen.
    
    Args:
        twin: Digital twin RealisticFungalComputer
        specimen: Original specimen RealisticFungalComputer
        xor_params: Optimized XOR gate parameters from twin
        
    Returns:
        Dictionary with validation results
    """
    logger.info("="*70)
    logger.info("XOR GATE VALIDATION: Twin vs Specimen")
    logger.info("="*70)
    
    # Test on twin
    logger.info("\nTesting XOR gate on Digital Twin...")
    twin_results = test_xor_gate_performance(twin, xor_params)
    
    # Test on specimen
    logger.info("\nTesting XOR gate on Specimen...")
    specimen_results = test_xor_gate_performance(specimen, xor_params)
    
    # Compare results
    logger.info("\n" + "="*70)
    logger.info("COMPARISON: Twin vs Specimen")
    logger.info("="*70)
    
    logger.info(f"\nAccuracy:")
    logger.info(f"  Twin:     {twin_results['accuracy']*100:.1f}%")
    logger.info(f"  Specimen: {specimen_results['accuracy']*100:.1f}%")
    logger.info(f"  Match:    {twin_results['accuracy'] == specimen_results['accuracy']}")
    
    logger.info(f"\nOutput Separation:")
    logger.info(f"  Twin:     {twin_results['separation']:.4f}V")
    logger.info(f"  Specimen: {specimen_results['separation']:.4f}V")
    logger.info(f"  Difference: {abs(twin_results['separation'] - specimen_results['separation']):.4f}V")
    
    # Compare individual test cases
    logger.info(f"\nIndividual Test Cases:")
    logger.info(f"{'Input':<10} {'Twin Out':<12} {'Spec Out':<12} {'Difference':<12}")
    logger.info("-"*50)
    
    voltage_differences = []
    for i in range(len(twin_results['test_cases'])):
        twin_case = twin_results['test_cases'][i]
        spec_case = specimen_results['test_cases'][i]
        
        input_str = f"({twin_case['A']},{twin_case['B']})"
        twin_v = twin_case['output_voltage']
        spec_v = spec_case['output_voltage']
        diff = abs(twin_v - spec_v)
        voltage_differences.append(diff)
        
        logger.info(f"{input_str:<10} {twin_v:<12.4f} {spec_v:<12.4f} {diff:<12.4f}")
    
    avg_voltage_diff = np.mean(voltage_differences)
    max_voltage_diff = np.max(voltage_differences)
    
    logger.info(f"\nVoltage Difference Statistics:")
    logger.info(f"  Average: {avg_voltage_diff:.4f}V")
    logger.info(f"  Maximum: {max_voltage_diff:.4f}V")
    
    # Overall validation score
    accuracy_match = (twin_results['accuracy'] == specimen_results['accuracy'])
    separation_similar = abs(twin_results['separation'] - specimen_results['separation']) < 0.1
    voltage_similar = avg_voltage_diff < 0.2
    
    validation_passed = accuracy_match and separation_similar and voltage_similar
    
    logger.info(f"\nValidation Status:")
    logger.info(f"  Accuracy Match:      {accuracy_match}")
    logger.info(f"  Separation Similar:  {separation_similar}")
    logger.info(f"  Voltage Similar:     {voltage_similar}")
    logger.info(f"  Overall:             {'PASSED' if validation_passed else 'FAILED'}")
    
    return {
        'twin_results': twin_results,
        'specimen_results': specimen_results,
        'avg_voltage_difference': float(avg_voltage_diff),
        'max_voltage_difference': float(max_voltage_diff),
        'accuracy_match': accuracy_match,
        'separation_similar': separation_similar,
        'voltage_similar': voltage_similar,
        'validation_passed': validation_passed
    }

# ==========================================
# Utilities
# ==========================================

def convert_to_json_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ==========================================
# Visualization
# ==========================================

def plot_parameter_comparison(true_params: Dict, predicted_params: Dict, 
                              refined_params: Dict, output_path: Path):
    """Plot comparison of true, predicted, and refined parameters.
    
    Args:
        true_params: True parameters
        predicted_params: ML-predicted parameters
        refined_params: Optimization-refined parameters
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, param in enumerate(FUNGAL_PARAMS):
        ax = axes[idx]
        
        true_val = true_params[param]
        pred_val = predicted_params[param]
        refined_val = refined_params[param]
        
        # Bar plot
        x = ['True', 'ML Predicted', 'Refined']
        y = [true_val, pred_val, refined_val]
        colors = ['green', 'orange', 'blue']
        
        bars = ax.bar(x, y, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Calculate errors
        pred_error = abs(pred_val - true_val) / (abs(true_val) + 1e-10) * 100
        refined_error = abs(refined_val - true_val) / (abs(true_val) + 1e-10) * 100
        
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{param}\nML error: {pred_error:.1f}%, Refined error: {refined_error:.1f}%',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Parameter Rediscovery Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved parameter comparison plot: {output_path}")
    plt.close()

def plot_waveform_comparison(specimen_waveforms: Dict, twin_waveforms: Dict,
                            output_path: Path):
    """Plot comparison of specimen and twin response waveforms.
    
    Args:
        specimen_waveforms: Waveforms from specimen
        twin_waveforms: Waveforms from twin
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Step response
    ax = axes[0, 0]
    ax.plot(specimen_waveforms['step_time'], specimen_waveforms['step_response'],
           'b-', label='Specimen', linewidth=2)
    ax.plot(twin_waveforms['step_time'], twin_waveforms['step_response'],
           'r--', label='Twin', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Voltage (V)', fontsize=11)
    ax.set_title('Step Response', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Paired-pulse (first delay)
    ax = axes[0, 1]
    pp_spec = specimen_waveforms['paired_pulse'][0]
    pp_twin = twin_waveforms['paired_pulse'][0]
    ax.plot(pp_spec['time'], pp_spec['response'],
           'b-', label='Specimen', linewidth=2)
    ax.plot(pp_twin['time'], pp_twin['response'],
           'r--', label='Twin', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Voltage (V)', fontsize=11)
    ax.set_title(f'Paired-Pulse (delay={pp_spec["delay"]:.0f}ms)', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Paired-pulse (last delay)
    ax = axes[1, 0]
    pp_spec = specimen_waveforms['paired_pulse'][-1]
    pp_twin = twin_waveforms['paired_pulse'][-1]
    ax.plot(pp_spec['time'], pp_spec['response'],
           'b-', label='Specimen', linewidth=2)
    ax.plot(pp_twin['time'], pp_twin['response'],
           'r--', label='Twin', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Voltage (V)', fontsize=11)
    ax.set_title(f'Paired-Pulse (delay={pp_spec["delay"]:.0f}ms)', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Triangle sweep
    ax = axes[1, 1]
    ax.plot(specimen_waveforms['triangle_voltage_applied'], 
           specimen_waveforms['triangle_response'],
           'b-', label='Specimen', linewidth=2)
    ax.plot(twin_waveforms['triangle_voltage_applied'], 
           twin_waveforms['triangle_response'],
           'r--', label='Twin', linewidth=2, alpha=0.7)
    ax.set_xlabel('Applied Voltage (V)', fontsize=11)
    ax.set_ylabel('Response Voltage (V)', fontsize=11)
    ax.set_title('Triangle Sweep (V-V curve)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Waveform Matching: Specimen vs Twin', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved waveform comparison plot: {output_path}")
    plt.close()

def plot_xor_gate_comparison(xor_validation_results: Dict, output_path: Path):
    """Plot comparison of XOR gate performance between twin and specimen.
    
    Args:
        xor_validation_results: Results from validate_xor_gate_on_specimen
        output_path: Path to save plot
    """
    twin_results = xor_validation_results['twin_results']
    specimen_results = xor_validation_results['specimen_results']
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ==========================================
    # Top Row: Output Voltage Comparison
    # ==========================================
    ax1 = fig.add_subplot(gs[0, :])
    
    input_labels = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    x = np.arange(len(input_labels))
    width = 0.35
    
    twin_voltages = [case['output_voltage'] for case in twin_results['test_cases']]
    spec_voltages = [case['output_voltage'] for case in specimen_results['test_cases']]
    expected_xor = [case['expected_xor'] for case in twin_results['test_cases']]
    
    # Color bars by expected XOR output
    twin_colors = ['#2ecc71' if xor == 1 else '#e74c3c' for xor in expected_xor]
    spec_colors = ['#27ae60' if xor == 1 else '#c0392b' for xor in expected_xor]
    
    bars1 = ax1.bar(x - width/2, twin_voltages, width, label='Twin', 
                    color=twin_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, spec_voltages, width, label='Specimen',
                    color=spec_colors, alpha=0.6, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}V',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add threshold lines
    ax1.axhline(y=twin_results['threshold'], color='blue', linestyle='--', 
                linewidth=2, alpha=0.5, label=f"Twin Threshold ({twin_results['threshold']:.3f}V)")
    ax1.axhline(y=specimen_results['threshold'], color='red', linestyle='--', 
                linewidth=2, alpha=0.5, label=f"Spec Threshold ({specimen_results['threshold']:.3f}V)")
    
    ax1.set_xlabel('Input (A, B)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Output Voltage (V)', fontsize=12, fontweight='bold')
    ax1.set_title('XOR Gate Output Voltages: Twin vs Specimen', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(input_labels, fontsize=11)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ==========================================
    # Middle Row: Voltage Traces for Each Input
    # ==========================================
    for i in range(4):
        ax = fig.add_subplot(gs[1, i % 3] if i < 3 else gs[2, 0])
        
        twin_case = twin_results['test_cases'][i]
        spec_case = specimen_results['test_cases'][i]
        
        # Plot voltage traces
        ax.plot(twin_case['time'], twin_case['voltage_trace'], 
                'b-', label='Twin', linewidth=2, alpha=0.8)
        ax.plot(spec_case['time'], spec_case['voltage_trace'], 
                'r--', label='Specimen', linewidth=2, alpha=0.7)
        
        # Add shaded region for measurement window
        response_start = 100.0 + 500.0 + 50.0
        response_end = 100.0 + 500.0 + 500.0
        ax.axvspan(response_start, response_end, alpha=0.1, color='green', 
                   label='Measurement Window')
        
        input_str = f"({twin_case['A']},{twin_case['B']})"
        expected = twin_case['expected_xor']
        twin_pred = twin_case['predicted_xor']
        spec_pred = spec_case['predicted_xor']
        
        title_color = 'green' if (twin_pred == expected and spec_pred == expected) else 'red'
        ax.set_title(f"Input {input_str} → Expected XOR={expected}\n"
                    f"Twin={twin_pred}, Spec={spec_pred}", 
                    fontsize=10, fontweight='bold', color=title_color)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_ylabel('Voltage (V)', fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # ==========================================
    # Bottom Right: Summary Statistics
    # ==========================================
    ax_stats = fig.add_subplot(gs[2, 1:])
    ax_stats.axis('off')
    
    # Create summary text
    summary_text = [
        "XOR GATE VALIDATION SUMMARY",
        "=" * 50,
        "",
        f"Accuracy:",
        f"  Twin:     {twin_results['accuracy']*100:.1f}% ({sum(c['correct'] for c in twin_results['test_cases'])}/4)",
        f"  Specimen: {specimen_results['accuracy']*100:.1f}% ({sum(c['correct'] for c in specimen_results['test_cases'])}/4)",
        f"  Match:    {'✓ YES' if xor_validation_results['accuracy_match'] else '✗ NO'}",
        "",
        f"Output Separation (XOR=1 avg - XOR=0 avg):",
        f"  Twin:     {twin_results['separation']:.4f}V",
        f"  Specimen: {specimen_results['separation']:.4f}V",
        f"  Difference: {abs(twin_results['separation'] - specimen_results['separation']):.4f}V",
        f"  Similar:  {'✓ YES' if xor_validation_results['separation_similar'] else '✗ NO'}",
        "",
        f"Voltage Differences:",
        f"  Average:  {xor_validation_results['avg_voltage_difference']:.4f}V",
        f"  Maximum:  {xor_validation_results['max_voltage_difference']:.4f}V",
        f"  Similar:  {'✓ YES' if xor_validation_results['voltage_similar'] else '✗ NO'}",
        "",
        f"Overall Validation: {'✓ PASSED' if xor_validation_results['validation_passed'] else '✗ FAILED'}",
    ]
    
    # Color code the overall result
    overall_color = 'green' if xor_validation_results['validation_passed'] else 'red'
    
    text_y = 0.95
    for i, line in enumerate(summary_text):
        if i == 0:  # Title
            ax_stats.text(0.5, text_y, line, fontsize=12, fontweight='bold', 
                         ha='center', va='top', family='monospace')
        elif line.startswith('Overall'):
            ax_stats.text(0.5, text_y, line, fontsize=11, fontweight='bold',
                         ha='center', va='top', family='monospace', color=overall_color)
        elif '✓' in line or '✗' in line:
            color = 'green' if '✓' in line else 'red'
            ax_stats.text(0.1, text_y, line, fontsize=10, ha='left', va='top', 
                         family='monospace', color=color)
        else:
            ax_stats.text(0.1, text_y, line, fontsize=10, ha='left', va='top', 
                         family='monospace')
        text_y -= 0.05
    
    plt.suptitle('XOR Gate Performance: Digital Twin vs Specimen', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved XOR gate comparison plot: {output_path}")
    plt.close()

# ==========================================
# Main Pipeline
# ==========================================

def rediscover_parameters(model_dir: Path, model_type: str = 'random_forest',
                         num_nodes: int = None, random_seed: int = None,
                         optimization_method: str = 'dual_annealing',
                         skip_optimization: bool = False,
                         use_inferred_network: bool = False,
                         use_full_bounds: bool = True):
    """Main pipeline for parameter rediscovery.
    
    Args:
        model_dir: Directory containing trained models
        model_type: Type of model to use ('random_forest' or 'mlp')
        num_nodes: Number of nodes for specimen (None for random)
        random_seed: Random seed for specimen (None for random)
        optimization_method: Optimization method for refinement
        skip_optimization: If True, skip optimization step
        use_inferred_network: If True, use ML-inferred network structure (harder problem)
        use_full_bounds: If True, use full parameter bounds for global search
    """
    logger.info("="*70)
    logger.info("FUNGAL PARAMETER REDISCOVERY")
    if use_inferred_network:
        logger.info("MODE: ML-Inferred Network (Hierarchical Optimization)")
    else:
        logger.info("MODE: Exact Network Copy (Default)")
    logger.info("="*70)
    
    # Load models
    models, scaler = load_models(model_dir, model_type)
    
    # Step 1: Create random specimen
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Create Random Specimen")
    logger.info("="*70)
    specimen, true_params = create_random_specimen(num_nodes, random_seed)
    
    # Step 2: Characterize specimen
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Characterize Specimen")
    logger.info("="*70)
    features = characterize_specimen(specimen)
    specimen_waveforms = collect_response_waveforms(specimen)
    
    # Step 3: Predict parameters using ML models
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Predict Parameters with ML Models")
    logger.info("="*70)
    predicted_params, pred_metadata = predict_parameters(features, models, scaler, model_type)
    
    # Step 4: Create Twin
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Create Twin with Predicted Parameters")
    logger.info("="*70)
    
    network_predictions = pred_metadata.get('network_predictions', {})
    
    twin_initial = create_twin(
        specimen, 
        predicted_params,
        network_predictions=network_predictions,
        use_inferred_network=use_inferred_network
    )
    twin_waveforms_initial = collect_response_waveforms(twin_initial)
    initial_mismatch = compute_waveform_mismatch(specimen_waveforms, twin_waveforms_initial)
    logger.info(f"Initial waveform mismatch: {initial_mismatch:.6f}")
    
    # Step 5: Refine parameters with optimization
    best_network_config = None
    
    if not skip_optimization:
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Refine Parameters with Optimization")
        logger.info("="*70)
        
        if use_inferred_network:
            # Use hierarchical optimization for network + parameters
            refined_params, best_network_config, opt_info = hierarchical_optimization_with_network(
                specimen, specimen_waveforms, predicted_params,
                network_predictions, method=optimization_method, use_full_bounds=use_full_bounds
            )
            
            # Create final twin with optimized network and parameters
            temp_network_pred = network_predictions.copy()
            temp_network_pred['num_nodes'] = best_network_config['num_nodes']
            
            twin_final = create_twin(
                specimen, 
                refined_params,
                network_predictions=temp_network_pred,
                use_inferred_network=True,
                network_seed=best_network_config['seed']
            )
        else:
            # Standard optimization (parameters only, exact network)
            refined_params, opt_info = refine_parameters_optimization(
                specimen, specimen_waveforms, predicted_params,
                method=optimization_method, use_full_bounds=use_full_bounds
            )
            
            # Create final twin
            twin_final = create_twin(specimen, refined_params)
        
        twin_waveforms_final = collect_response_waveforms(twin_final)
        final_mismatch = compute_waveform_mismatch(specimen_waveforms, twin_waveforms_final)
        logger.info(f"Final waveform mismatch: {final_mismatch:.6f}")
        logger.info(f"Improvement: {(1 - final_mismatch/initial_mismatch)*100:.1f}%")
    else:
        refined_params = predicted_params
        twin_waveforms_final = twin_waveforms_initial
        final_mismatch = initial_mismatch
        opt_info = {'skipped': True}
    
    # Step 6: XOR Gate Optimization and Validation
    xor_validation_results = None
    xor_params = None
    
    if not skip_optimization:
        logger.info("\n" + "="*70)
        logger.info("STEP 6: XOR Gate Optimization and Validation")
        logger.info("="*70)
        
        # Use the digital twin to optimize XOR gate
        logger.info("\nOptimizing XOR gate on Digital Twin...")
        xor_optimization = optimize_xor_gate(
            num_nodes=twin_final.num_nodes,
            n_calls=200,
            random_state=42,
            minimizer='gp',
            tune_physics=False,
            env=twin_final
        )
        
        xor_params = xor_optimization['params']
        
        logger.info(f"\nXOR Gate Optimized on Twin:")
        logger.info(f"  Best Score: {xor_params['score']:.4f}")
        logger.info(f"  Electrode A: ({xor_params['x_A']:.2f}, {xor_params['y_A']:.2f})")
        logger.info(f"  Electrode B: ({xor_params['x_B']:.2f}, {xor_params['y_B']:.2f})")
        logger.info(f"  Output Probe: ({xor_params['x_out']:.2f}, {xor_params['y_out']:.2f})")
        logger.info(f"  Voltage: {xor_params['voltage']:.2f}V, Duration: {xor_params['duration']:.2f}ms, Delay: {xor_params['delay']:.2f}ms")
        
        # Validate XOR gate on specimen
        logger.info("\nValidating XOR gate on Specimen...")
        xor_validation_results = validate_xor_gate_on_specimen(twin_final, specimen, xor_params)
        
        logger.info("\n" + "="*70)
        logger.info("XOR GATE VALIDATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Validation Status: {'PASSED' if xor_validation_results['validation_passed'] else 'FAILED'}")
    
    # Calculate errors
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    
    logger.info("\nParameter Errors:")
    logger.info(f"{'Parameter':<12} {'True':<10} {'Predicted':<12} {'Refined':<12} {'ML Error %':<12} {'Final Error %':<12}")
    logger.info("-"*70)
    
    for param in FUNGAL_PARAMS:
        true_val = true_params[param]
        pred_val = predicted_params[param]
        refined_val = refined_params[param]
        
        pred_error = abs(pred_val - true_val) / (abs(true_val) + 1e-10) * 100
        refined_error = abs(refined_val - true_val) / (abs(true_val) + 1e-10) * 100
        
        logger.info(f"{param:<12} {true_val:<10.4f} {pred_val:<12.4f} {refined_val:<12.4f} "
                   f"{pred_error:<12.1f} {refined_error:<12.1f}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'mode': 'inferred_network' if use_inferred_network else 'exact_network',
        'specimen': {
            'num_nodes': specimen.num_nodes,
            'num_edges': len(specimen.edge_list),
            'random_seed': random_seed,
        },
        'twin': {
            'num_nodes': twin_final.num_nodes if not skip_optimization else twin_initial.num_nodes,
            'num_edges': len(twin_final.edge_list) if not skip_optimization else len(twin_initial.edge_list),
            'network_config': best_network_config if (use_inferred_network and best_network_config) else None,
        },
        'true_parameters': true_params,
        'predicted_parameters': predicted_params,
        'refined_parameters': refined_params,
        'network_predictions': network_predictions,
        'model_type': model_type,
        'optimization': opt_info,
        'mismatch': {
            'initial': float(initial_mismatch),
            'final': float(final_mismatch),
            'improvement_percent': float((1 - final_mismatch/initial_mismatch)*100) if not skip_optimization else 0.0
        },
        'xor_gate': {
            'optimized_params': xor_params,
            'validation': xor_validation_results
        } if xor_params else None
    }
    
    # Convert to JSON-serializable format
    results = convert_to_json_serializable(results)
    
    results_path = OUTPUT_DIR / f'rediscovery_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {results_path}")
    
    # Generate plots
    logger.info("\nGenerating visualizations...")
    plot_parameter_comparison(
        true_params, predicted_params, refined_params,
        OUTPUT_DIR / f'parameter_comparison_{timestamp}.png'
    )
    plot_waveform_comparison(
        specimen_waveforms, twin_waveforms_final,
        OUTPUT_DIR / f'waveform_comparison_{timestamp}.png'
    )
    
    # Generate XOR gate comparison plot if validation was performed
    if xor_validation_results is not None:
        plot_xor_gate_comparison(
            xor_validation_results,
            OUTPUT_DIR / f'xor_gate_comparison_{timestamp}.png'
        )
    
    logger.info("\n" + "="*70)
    logger.info("PARAMETER REDISCOVERY COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    
    return results

# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rediscover fungal parameters using ML models and optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Basic usage with dual_annealing (fast global optimizer, default)
  python rediscover_fungal_parameters.py
  
  # Use multi-start L-BFGS-B (fastest, good exploration)
  python rediscover_fungal_parameters.py --optimization multi_start
  
  # Use ML-inferred network (hierarchical optimization, harder problem)
  python rediscover_fungal_parameters.py --infer-network
  
  # Use MLP models instead of Random Forest
  python rediscover_fungal_parameters.py --model-type mlp
  
  # Specify specimen properties
  python rediscover_fungal_parameters.py --num-nodes 50 --seed 12345
  
  # Use local search bounds (faster but may miss global optimum)
  python rediscover_fungal_parameters.py --local-bounds
  
  # Skip optimization (ML prediction only)
  python rediscover_fungal_parameters.py --skip-optimization
  
  # Full challenge mode: inferred network + MLP models + dual annealing
  python rediscover_fungal_parameters.py --infer-network --model-type mlp
        """
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='ml_models',
        help='Directory containing trained models (default: ml_models)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['random_forest', 'mlp'],
        default='random_forest',
        help='Type of ML model to use (default: random_forest)'
    )
    parser.add_argument(
        '--num-nodes',
        type=int,
        default=None,
        help='Number of nodes for specimen (default: random)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for specimen (default: random)'
    )
    parser.add_argument(
        '--optimization',
        type=str,
        choices=['gp_minimize', 'gaussian_process', 'dual_annealing', 'multi_start', 'shgo', 'basinhopping', 
                 'differential_evolution', 'L-BFGS-B', 'nelder-mead', 'powell'],
        default='gp_minimize',
        help='Optimization method for refinement (default: gp_minimize). '
             'Recommended: gp_minimize/gaussian_process (GP with warm-start, efficient), '
             'dual_annealing (fast, global), multi_start (fastest, good exploration), '
             'shgo (deterministic global), basinhopping (rugged landscapes)'
    )
    parser.add_argument(
        '--skip-optimization',
        action='store_true',
        help='Skip optimization step (ML prediction only)'
    )
    parser.add_argument(
        '--infer-network',
        action='store_true',
        help='Use ML-inferred network structure instead of copying specimen network (harder problem, uses hierarchical optimization)'
    )
    parser.add_argument(
        '--local-bounds',
        action='store_true',
        help='Use local search bounds around ML prediction instead of full parameter bounds (faster but may miss global optimum)'
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        exit(1)
    
    results = rediscover_parameters(
        model_dir=model_dir,
        model_type=args.model_type,
        num_nodes=args.num_nodes,
        random_seed=args.seed,
        optimization_method=args.optimization,
        skip_optimization=args.skip_optimization,
        use_inferred_network=args.infer_network,
        use_full_bounds=not args.local_bounds
    )
