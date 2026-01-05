# Machine Learning Workflow for Fungal Parameter Prediction

This document describes the complete workflow for generating training data and building ML models to predict fungal computer parameters from system identification responses.

## Overview

The workflow consists of three main steps:

1. **Data Generation**: Run system characterization experiments with varied parameters
2. **Model Training**: Train ML models to predict parameters from response features
3. **Model Deployment**: Use trained models for parameter inference

## Step 1: Generate Training Data

Run the systematic characterization study to generate a dataset:

```bash
# Quick test (30 samples)
python systematic_characterization_study.py --quick --yes

# Full study (2400+ samples)
python systematic_characterization_study.py --yes

# Resume interrupted study
python systematic_characterization_study.py --resume
```

**Output**: CSV file in `characterization_study_results/` containing:
- Fungal parameters (ground truth): `tau_v`, `tau_w`, `a`, `b`, `v_scale`, `R_off`, `R_on`, `alpha`
- Response features from three protocols:
  - Step response: rise time, saturation voltage, oscillation index
  - Paired-pulse: recovery ratios at different delays
  - Triangle sweep: hysteresis area

**Configuration** (in `systematic_characterization_study.py`):
- `NODE_COUNTS`: Network sizes to test (default: [20, 30, 40, 50, 60, 80, 100, 120])
- `TRIALS_PER_NODE_COUNT`: Random parameter sets per node count (default: 300)
- Parameter ranges based on `optimize_fungal_constants` search space

## Step 2: Train ML Models

Train RandomForest and MLP models to predict parameters:

```bash
# Train on generated data
python train_parameter_predictor.py --data characterization_study_results/characterization_results_TIMESTAMP.csv

# Custom train/test split
python train_parameter_predictor.py --data results.csv --test-size 0.3
```

**Models Trained**:
- **RandomForestRegressor**: Ensemble tree-based model (200 trees, max_depth=20)
- **MLPRegressor**: Neural network (3 hidden layers: 128→64→32 neurons)

**Output** (in `ml_models/`):
- Trained models: `{model_type}_{parameter}_TIMESTAMP.pkl` (16 models total)
- Feature scaler: `scaler_TIMESTAMP.pkl`
- Evaluation results: `evaluation_results_TIMESTAMP.csv`
- Feature importance: `feature_importance_TIMESTAMP.csv`
- Training metadata: `training_metadata_TIMESTAMP.json`
- Visualizations:
  - `random_forest_predictions.png`: Predicted vs actual for all parameters
  - `mlp_predictions.png`: Predicted vs actual for all parameters
  - `feature_importance.png`: Top features for each parameter
  - `model_comparison.png`: RF vs MLP performance comparison

**Evaluation Metrics**:
- R² score (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- MAPE (mean absolute percentage error)

## Step 3: Use Trained Models

Load and use trained models for inference:

```python
import joblib
import numpy as np
import pandas as pd

# Load models
rf_tau_v = joblib.load('ml_models/random_forest_tau_v_TIMESTAMP.pkl')
scaler = joblib.load('ml_models/scaler_TIMESTAMP.pkl')

# Prepare features from a new characterization experiment
features = {
    'step_rise_time': 25.1,
    'step_saturation_voltage': -1.26,
    'step_oscillation_index': 0.39,
    'pp_recovery_ratio_delay_200': 0.0,
    'pp_recovery_ratio_delay_800': 0.0,
    'pp_recovery_ratio_delay_2000': 0.0,
    # ... other features
    'tri_hysteresis_area': 0.026,
    'num_nodes': 60,
    'num_edges': 284,
    'network_density': 0.16,
}

# Convert to array (ensure correct feature order)
X = np.array([list(features.values())])

# For Random Forest (use unscaled features)
tau_v_pred = rf_tau_v.predict(X)[0]

# For MLP (use scaled features)
X_scaled = scaler.transform(X)
mlp_tau_v = joblib.load('ml_models/mlp_tau_v_TIMESTAMP.pkl')
tau_v_pred_mlp = mlp_tau_v.predict(X_scaled)[0]

print(f"Predicted tau_v: {tau_v_pred:.2f} ms (RF), {tau_v_pred_mlp:.2f} ms (MLP)")
```

## System Identification Protocols

### 1. Step Response Protocol
- **Stimulus**: Single long DC pulse (3000ms, 2V)
- **Measurement**: Voltage response at probe 5mm from center
- **Features Extracted**:
  - `step_rise_time`: Time to reach 90% of saturation (ms)
  - `step_saturation_voltage`: Steady-state voltage (V)
  - `step_oscillation_index`: Normalized std dev during settling

### 2. Paired-Pulse Protocol
- **Stimulus**: Two short pulses (50ms, 2V) with variable inter-pulse intervals
- **Delays Tested**: 200ms, 800ms, 2000ms
- **Features Extracted** (per delay):
  - `pp_recovery_ratio_delay_X`: 2nd peak / 1st peak ratio
  - `pp_first_peak_delay_X`: First peak height (V)
  - `pp_second_peak_delay_X`: Second peak height (V)
- **Derived Features**:
  - `pp_avg_recovery_ratio`: Mean recovery across delays
  - `pp_std_recovery_ratio`: Std dev of recovery ratios
  - `pp_recovery_slope`: Rate of recovery improvement

### 3. Triangle Sweep Protocol (Cyclic Voltammetry)
- **Stimulus**: Voltage ramp 0→+5V→-5V→0 at 0.01 V/ms
- **Measurement**: Current-voltage relationship
- **Features Extracted**:
  - `tri_hysteresis_area`: Area enclosed by V-I curve (plasticity measure)

## Expected Performance

With 2000+ training samples, typical model performance:

**Random Forest**:
- R² scores: 0.7-0.95 depending on parameter
- Best for: `tau_v`, `tau_w`, `R_off`, `R_on` (direct physical properties)

**MLP**:
- R² scores: 0.6-0.9 depending on parameter
- Best for: `a`, `b`, `alpha` (complex nonlinear relationships)

**Most Predictable Parameters**:
1. `tau_v` (voltage time constant) - strongly correlated with rise time
2. `tau_w` (recovery time constant) - correlated with oscillation behavior
3. `R_off`, `R_on` (resistance states) - correlated with saturation levels

**Challenging Parameters**:
1. `alpha` (memristor adaptation rate) - subtle effects
2. `v_scale` (voltage scaling) - confounded with other parameters

## Tips for Better Performance

1. **More Data**: Increase `TRIALS_PER_NODE_COUNT` to 500+ for better generalization
2. **Feature Engineering**: Consider adding interaction terms or polynomial features
3. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
4. **Ensemble Methods**: Combine RF and MLP predictions (e.g., weighted average)
5. **Cross-Validation**: Use k-fold CV for more robust evaluation

## File Structure

```
fungal-computer-sim/
├── realistic_sim.py                          # Core simulation with protocols
├── systematic_characterization_study.py      # Data generation script
├── train_parameter_predictor.py              # ML training script
├── characterization_study_results/           # Generated datasets
│   ├── characterization_results_*.csv
│   ├── checkpoint_*.csv
│   └── study_config_*.json
└── ml_models/                                # Trained models & results
    ├── random_forest_*.pkl
    ├── mlp_*.pkl
    ├── scaler_*.pkl
    ├── evaluation_results_*.csv
    ├── feature_importance_*.csv
    ├── training_metadata_*.json
    └── *.png (visualizations)
```

## Troubleshooting

**Issue**: All characterization trials fail
- Check NumPy version compatibility (trapz vs trapezoid)
- Verify network connectivity (increase node count or connection radius)

**Issue**: Poor model performance (R² < 0.5)
- Need more training data (increase TRIALS_PER_NODE_COUNT)
- Check for NaN values in features
- Verify parameter ranges are realistic

**Issue**: Out of memory during training
- Reduce `n_estimators` in RandomForest
- Reduce `hidden_layer_sizes` in MLP
- Process data in batches

## Next Steps

1. **Active Learning**: Use model uncertainty to guide next experiments
2. **Transfer Learning**: Pre-train on simulated data, fine-tune on real fungal networks
3. **Multi-Task Learning**: Train single model to predict all parameters jointly
4. **Inverse Design**: Use models to find optimal parameters for desired behavior
