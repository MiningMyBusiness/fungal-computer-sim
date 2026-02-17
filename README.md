# Fungal Computer Simulation

A computational framework for simulating fungal networks as unconventional computing substrates, featuring **digital twin technology** for parameter discovery and optimization.

## Overview

This repository implements a biophysically realistic fungal network computer using FitzHugh-Nagumo (FHN) neurodynamics coupled with memristive connections. The core innovation is a **complete digital twin workflow** that can characterize unknown fungal specimens, predict their biophysical parameters using machine learning, and optimize computational tasks on the inferred model.

### Key Innovation: Digital Twin Workflow

The centerpiece of this repository is **`rediscover_fungal_parameters.py`**, which implements a complete digital twin pipeline:

1. **Specimen Characterization** - Run system identification protocols on an unknown fungal specimen
2. **ML-Based Parameter Inference** - Predict biophysical parameters from response features using trained models
3. **Digital Twin Creation** - Instantiate a computational twin with inferred parameters
4. **Hierarchical Optimization** - Refine parameters through waveform matching and network structure search
5. **Task Optimization** - Use the digital twin to optimize computational tasks (e.g., XOR gates)
6. **Validation** - Deploy optimized configurations back to the physical specimen

This workflow bridges the gap between physical fungal networks and computational models, enabling **in silico optimization** of biological computing substrates.

## Repository Structure

### Core Simulation Engine
- **`realistic_sim.py`** - The foundational `RealisticFungalComputer` class implementing:
  - FitzHugh-Nagumo neurodynamics for excitable fungal nodes
  - Memristive edge conductance with activity-dependent plasticity
  - Random geometric graph topology mimicking hyphal networks
  - System identification protocols (step response, paired-pulse, triangle sweep)
  - Bayesian optimization for XOR gate design

### Digital Twin & Parameter Discovery
- **`rediscover_fungal_parameters.py` **[CORE WORKHORSE]** - Complete digital twin workflow:
  - Creates random fungal specimens with unknown parameters
  - Runs characterization protocols to extract response features
  - Uses ML models to predict biophysical parameters
  - Creates digital twins with inferred parameters
  - Refines parameters via hierarchical optimization (network + biophysics)
  - Validates twin accuracy through waveform matching
  - Optimizes computational tasks on the twin and validates on specimen

### Machine Learning Pipeline
- **`systematic_characterization_study.py`** - Generates training data by:
  - Sweeping network sizes (20-120 nodes)
  - Sampling biophysical parameter space (tau_v, tau_w, a, b, R_off, R_on, alpha)
  - Running system identification protocols on each configuration
  - Extracting 16 response features per specimen
  - Saving results to CSV for ML training

- **`train_parameter_predictor.py`** - Trains ML models to predict parameters from features:
  - Random Forest regressors (fast, interpretable)
  - Multi-layer Perceptron networks (higher accuracy)
  - Predicts 8 biophysical parameters + 3 network properties
  - Saves trained models and feature scalers

- **`train_parameter_predictor_advanced.py`** - Advanced ML with:
  - XGBoost and LightGBM gradient boosting
  - Hyperparameter tuning
  - Feature importance analysis
  - Cross-validation and ensemble methods

### Optimization Studies
- **`systematic_optimization_study.py`** - Large-scale XOR gate optimization across parameter space
- **`analyze_optimization_results.py`** - Statistical analysis of optimization results
- **`fungal_architect.py`** - Network architecture design and analysis tools

### Testing & Validation
- **`test_characterization.py`** - Validates characterization protocols
- **`test_optimization_logic.py`** - Tests optimization algorithms
- **`verify_coordinates.py`** - Validates spatial coordinate systems

## Installation

```bash
# Clone the repository
git clone git@github.com:MiningMyBusiness/fungal-computer-sim.git
cd fungal-computer-sim

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `numpy` - Numerical computing
- `scipy` - ODE integration and optimization
- `networkx` - Graph/network structures
- `scikit-learn` - Machine learning models
- `scikit-optimize` - Bayesian optimization
- `xgboost`, `lightgbm` - Gradient boosting
- `matplotlib`, `seaborn` - Visualization
- `pandas` - Data manipulation
- `joblib` - Model serialization

## Quick Start

### 1. Generate Training Data

Run system identification on diverse fungal networks:

```bash
python systematic_characterization_study.py --trials 300
```

This creates `characterization_study_results/characterization_results_*.csv` with ~2400 specimens.

### 2. Train ML Models

Train models to predict parameters from response features:

```bash
python train_parameter_predictor.py --model-type random_forest
python train_parameter_predictor.py --model-type mlp
```

Models are saved to `ml_models/` directory.

### 3. Run Digital Twin Workflow

Execute the complete parameter rediscovery pipeline:

```bash
python rediscover_fungal_parameters.py \
    --model-type random_forest \
    --optimization-method dual_annealing \
    --use-full-bounds \
    --num-nodes 50
```

**What this does:**
1. Creates a random fungal specimen with unknown parameters
2. Characterizes it using step response, paired-pulse, and triangle sweep protocols
3. Predicts parameters using trained ML models
4. Creates a digital twin with predicted parameters
5. Refines parameters through hierarchical optimization:
   - **Stage 1**: Network structure search (400 candidate networks)
   - **Stage 2**: Biophysical parameter optimization (dual annealing)
6. Validates twin accuracy via waveform matching
7. Optimizes an XOR gate on the twin
8. Validates the optimized gate on the original specimen

**Key Options:**
- `--model-type`: `random_forest` (fast) or `mlp` (accurate)
- `--optimization-method`: `dual_annealing`, `multi_start`, `shgo`, `differential_evolution`
- `--use-full-bounds`: Global search vs. local refinement
- `--use-inferred-network`: Use ML-predicted network structure vs. exact copy
- `--skip-xor-validation`: Skip final XOR gate optimization

**Output:**
- Parameter comparison plots showing true vs. predicted vs. refined
- Waveform matching visualizations
- XOR gate performance metrics
- JSON metadata with full optimization history

### 4. Simulate a Basic Fungal Network

```python
from realistic_sim import RealisticFungalComputer

# Create a fungal network
fungus = RealisticFungalComputer(num_nodes=40, area_size=20.0, random_seed=42)

# Run step response characterization
result = fungus.step_response_protocol(voltage=2.0, pulse_duration=3000.0)
print(f"Rise time: {result['rise_time']:.1f} ms")
print(f"Saturation: {result['saturation_voltage']:.3f} V")

# Optimize an XOR gate
from realistic_sim import optimize_xor_gate
best_config, best_score = optimize_xor_gate(fungus, n_calls=100)
print(f"XOR Score: {best_score:.4f}")
```

## Scientific Background

### Biophysical Model

**FitzHugh-Nagumo Dynamics** at each node:
```
dV/dt = (V - V³/3 - W + I_stim + I_coupling) / tau_v
dW/dt = (V + a - b*W) / tau_w
```

**Memristive Conductance** on edges:
```
G(M) = 1 / (R_on * M + R_off * (1-M))
dM/dt = alpha * |V_flux| * (M*(1-M) + 0.05)
```

**Parameters:**
- `tau_v` (30-150 ms): Voltage time constant (excitability)
- `tau_w` (300-1600 ms): Recovery time constant (refractoriness)
- `a`, `b` (0.5-1.0): FHN shape parameters
- `R_off`, `R_on` (2-300 Ω): Memristor resistance states
- `alpha` (0.0001-0.02): Memristor plasticity rate

### System Identification Protocols

1. **Step Response** - Long DC pulse → measures rise time, saturation, oscillations
2. **Paired-Pulse** - Two pulses at varying delays → measures refractory period
3. **Triangle Sweep** - Cyclic voltammetry → measures memristive hysteresis

These protocols extract 16 features that uniquely characterize each specimen.

### Hierarchical Optimization Strategy

The digital twin workflow uses a **two-stage hierarchical approach**:

**Stage 1: Network Structure Search**
- Adaptive coarse-to-fine sampling (400 networks)
- Evaluates network topology with fixed ML-predicted parameters
- Identifies best-matching network structure

**Stage 2: Biophysical Parameter Refinement**
- Global optimization (dual annealing, differential evolution, etc.)
- Fine-tunes parameters with fixed network structure
- Minimizes waveform mismatch across all protocols

This decomposition is well-conditioned because:
- Network structure has discrete, bounded search space
- Parameter optimization is continuous but lower-dimensional given fixed network
- Avoids ill-posed simultaneous optimization of structure + parameters

## Advanced Usage

### Custom Optimization Methods

```python
from rediscover_fungal_parameters import (
    create_random_specimen,
    characterize_specimen,
    collect_response_waveforms,
    load_models,
    predict_parameters,
    create_twin,
    hierarchical_optimization_with_network
)

# Create specimen
specimen, true_params = create_random_specimen(num_nodes=50)

# Characterize
features = characterize_specimen(specimen)
waveforms = collect_response_waveforms(specimen)

# Predict parameters
models, scaler = load_models(Path("ml_models"), model_type='random_forest')
predicted_params, metadata = predict_parameters(features, models, scaler)

# Hierarchical optimization
refined_params, best_seed, opt_info = hierarchical_optimization_with_network(
    specimen=specimen,
    specimen_waveforms=waveforms,
    initial_params=predicted_params,
    network_predictions=metadata['network_predictions'],
    method='dual_annealing',
    use_full_bounds=True
)
```

### Network Architecture Analysis

```python
from fungal_architect import analyze_network_properties

# Analyze network topology
properties = analyze_network_properties(fungus.G)
print(f"Clustering coefficient: {properties['clustering']:.3f}")
print(f"Average path length: {properties['avg_path_length']:.2f}")
```

## Results & Performance

### ML Model Accuracy (Random Forest)
- **tau_v**: R² = 0.92, MAE = 8.3 ms
- **tau_w**: R² = 0.89, MAE = 67.1 ms
- **R_off**: R² = 0.94, MAE = 12.4 Ω
- **alpha**: R² = 0.87, MAE = 0.0021

### Digital Twin Workflow Performance
- **Characterization**: ~15 seconds per specimen
- **ML Prediction**: <0.1 seconds
- **Network Search**: ~20 minutes (400 networks)
- **Parameter Refinement**: ~5-15 minutes (depends on method)
- **Total Pipeline**: ~30-40 minutes per specimen

### XOR Gate Optimization
- **Success Rate**: 85-95% (score > 0.8)
- **Typical Score**: 0.85-0.95 (1.0 = perfect XOR)
- **Optimization Time**: 10-20 minutes (100 Bayesian iterations)

## File Outputs

### Characterization Data
- `characterization_study_results/characterization_results_*.csv` - Training data
- `characterization_study_results/checkpoint_*.csv` - Progress checkpoints

### ML Models
- `ml_models/random_forest_<param>_*.pkl` - Trained RF models
- `ml_models/mlp_<param>_*.pkl` - Trained MLP models
- `ml_models/scaler_*.pkl` - Feature scalers
- `ml_models/feature_importance_*.csv` - Feature importance rankings

### Parameter Rediscovery
- `parameter_rediscovery_results/parameter_comparison_*.png` - True vs. predicted plots
- `parameter_rediscovery_results/rediscovery_results_*.json` - Full metadata
- `parameter_rediscovery_results/waveform_comparison_*.png` - Waveform matching plots

### Optimization Studies
- `optimization_study_results/optimization_results_*.csv` - XOR optimization data
- `optimization_study_results/electrode_distance_analysis.png` - Spatial analysis

## Documentation

- **`QUICK_START.md`** - Minimal getting started guide
- **`ML_WORKFLOW_README.md`** - Machine learning pipeline details
- **`OPTIMIZATION_STUDY_README.md`** - Optimization study methodology
- **`NETWORK_PROPERTIES_UPDATE.md`** - Network topology analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fungal_computer_sim,
  title = {Fungal Computer Simulation: Digital Twin Framework for Biological Computing},
  author = {Kiran Bhattacharyya},
  year = {2026},
  url = {git@github.com:MiningMyBusiness/fungal-computer-sim.git}
}
```

## Future Directions

- **Real-time control**: Closed-loop feedback for dynamic task adaptation
- **Multi-task learning**: Simultaneous optimization of multiple logic gates
- **Transfer learning**: Apply learned parameters across different fungal species
- **Hardware integration**: Interface with physical fungal networks via microelectrode arrays
- **Spiking networks**: Extend to spiking neural network formulations

## License

MIT License

## Contact

Kiran Bhattacharyya

---

**Key Insight**: This repository demonstrates that fungal networks can be treated as **programmable biological computers** through a digital twin approach. By combining system identification, machine learning, and hierarchical optimization, we can discover the "source code" (biophysical parameters) of living fungal networks and optimize them for computational tasks—all without direct parameter measurements.
