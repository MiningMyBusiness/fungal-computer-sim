# Systematic XOR Gate Optimization Study

This directory contains scripts for running a comprehensive systematic study of XOR gate optimization across different network configurations and analyzing the results.

## Overview

The study systematically varies:
- **num_nodes**: Network size (20 to 150 nodes) to study node-density effects
- **random_state**: Multiple trials per configuration for statistical robustness
- **tune_physics**: Enabled to identify optimal fungal characteristics

The goal is to identify general principles for:
1. Optimal electrode placement distances
2. Optimal stimulus parameters (voltage, duration, delay)
3. Fungal characteristics that work best
4. Effects of node density on XOR gate performance

## Files

### 1. `systematic_optimization_study.py`
Main script that runs the optimization study.

**Configuration:**
- `NODE_COUNTS`: List of node counts to test (default: 20, 25, 30, ..., 150)
- `TRIALS_PER_CONFIG`: Number of random trials per node count (default: 10)
- `N_CALLS`: Optimization iterations per trial (default: 15)
- `TUNE_PHYSICS`: Enable physics parameter tuning (default: True)

**Output:**
- `optimization_results_<timestamp>.csv`: Complete results for all trials
- `checkpoint_<timestamp>.csv`: Intermediate checkpoints (saved after each trial)
- `study_config_<timestamp>.json`: Study configuration metadata

**Usage:**
```bash
python systematic_optimization_study.py
```

The script will prompt for confirmation before starting. It saves checkpoints after each trial, so you can safely interrupt and resume if needed.

**Estimated Runtime:**
- With default settings: ~130 trials total
- Each trial takes approximately 5-15 minutes (depending on `N_CALLS` and `TUNE_PHYSICS`)
- Total estimated time: 10-30 hours

**Adjusting for Faster Testing:**
To run a quicker pilot study, modify these parameters in the script:
```python
NODE_COUNTS = [20, 30, 40, 50]  # Fewer node counts
TRIALS_PER_CONFIG = 3  # Fewer trials per config
N_CALLS = 10  # Fewer optimization calls
TUNE_PHYSICS = False  # Disable physics tuning
```

### 2. `analyze_optimization_results.py`
Analysis script that processes the results and generates insights.

**Features:**
- Statistical analysis of node density effects
- Identification of optimal electrode placement patterns
- Analysis of stimulus parameter relationships
- Fungal characteristic optimization analysis
- Comprehensive visualizations
- Summary report generation

**Output:**
- `node_density_analysis.png`: Visualizations of network size effects
- `electrode_distance_analysis.png`: Electrode placement patterns
- `stimulus_parameter_analysis.png`: Stimulus parameter relationships
- `fungal_parameter_analysis.png`: Optimal fungal characteristics (if tuning enabled)
- `analysis_summary_report.txt`: Text summary of key findings

**Usage:**
```bash
python analyze_optimization_results.py
```

The script automatically loads the most recent results file from `optimization_study_results/`.

## Workflow

### Step 1: Configure and Run Study
1. Edit `systematic_optimization_study.py` to adjust parameters if needed
2. Run the study:
   ```bash
   python systematic_optimization_study.py
   ```
3. Confirm when prompted
4. Monitor progress in the console output
5. Results are saved incrementally (safe to interrupt)

### Step 2: Analyze Results
Once the study completes (or has enough data):
```bash
python analyze_optimization_results.py
```

This will generate:
- Statistical analyses printed to console
- Visualization plots saved as PNG files
- A comprehensive text report

## Results Interpretation

### Key Metrics

**Score**: The primary performance metric
- Higher is better
- Represents the separation between XOR high outputs (1,0) and (0,1) vs low outputs (0,0) and (1,1)

**Electrode Distances**:
- `dist_AB`: Distance between input electrodes A and B
- `dist_A_out`, `dist_B_out`: Distances from inputs to output
- `dist_avg_input_to_out`: Average distance from inputs to output

**Stimulus Parameters**:
- `voltage`: Stimulation voltage (V)
- `duration`: Pulse duration (ms)
- `delay`: Time delay between electrode A and B for (1,1) case (ms)

**Fungal Parameters** (if tuning enabled):
- `tau_v`, `tau_w`: FitzHugh-Nagumo time constants
- `a`, `b`: FitzHugh-Nagumo parameters
- `v_scale`: Voltage scaling factor
- `R_off`, `R_on`: Memristor resistance states
- `alpha`: Memristor adaptation rate

### What to Look For

1. **Node Density Effects**:
   - Does performance improve/degrade with more nodes?
   - Is there an optimal node density?

2. **Electrode Placement Principles**:
   - What's the optimal distance between input electrodes?
   - How far should the output be from the inputs?
   - Are there consistent geometric patterns?

3. **Stimulus Parameter Patterns**:
   - What voltage range works best?
   - Optimal pulse duration?
   - Is delay between electrodes important?

4. **Fungal Characteristics**:
   - Which parameters deviate most from defaults?
   - Which parameters correlate with improvement?
   - Are there consistent optimal values across trials?

## Data Format

### Results CSV Columns

**Basic Info**:
- `num_nodes`, `trial_idx`, `random_state`
- `num_edges`, `network_density`, `area_size`
- `success`, `error_message`, `trial_duration_seconds`

**Electrode Positions**:
- `x_A`, `y_A`, `x_B`, `y_B`, `x_out`, `y_out`

**Distances**:
- `dist_AB`, `dist_A_out`, `dist_B_out`, `dist_avg_input_to_out`

**Stimulus Parameters**:
- `voltage`, `duration`, `delay`, `score`

**Default Fungal Parameters**:
- `default_tau_v`, `default_tau_w`, etc.

**Tuned Parameters** (if physics tuning enabled):
- `tuned_tau_v`, `tuned_tau_w`, etc.
- `tuned_score`, `score_improvement`

**Optimization Statistics**:
- `opt_min_score`, `opt_max_score`, `opt_mean_score`, `opt_std_score`
- `opt_n_calls`

## Tips

### For Long-Running Studies
- Use `screen` or `tmux` to run in background:
  ```bash
  screen -S optimization
  python systematic_optimization_study.py
  # Detach with Ctrl-A, D
  # Reattach with: screen -r optimization
  ```

### For Debugging
- Check the most recent checkpoint file to see progress
- Load checkpoint CSV in Python/pandas to inspect intermediate results
- Reduce `NODE_COUNTS` and `TRIALS_PER_CONFIG` for quick testing

### For Analysis
- Run analysis script multiple times as more data accumulates
- Compare results across different study runs
- Use the generated visualizations to identify patterns
- Read the summary report for high-level insights

## Customization

### Adding New Metrics
Edit `extract_results()` in `systematic_optimization_study.py` to add custom metrics:
```python
# Example: Add network diameter
record['network_diameter'] = nx.diameter(env.G) if nx.is_connected(env.G) else np.nan
```

### Custom Analysis
Add new analysis functions to `analyze_optimization_results.py`:
```python
def analyze_custom_metric(df):
    # Your analysis code here
    pass
```

### Different Node Count Progressions
Modify `NODE_COUNTS` for different sampling strategies:
```python
# Linear progression
NODE_COUNTS = list(range(20, 151, 10))

# Logarithmic progression
NODE_COUNTS = [int(x) for x in np.logspace(np.log10(20), np.log10(150), 15)]

# Focus on specific range
NODE_COUNTS = list(range(30, 61, 2))
```

## Requirements

All dependencies are in `requirements.txt`:
- numpy
- scipy
- networkx
- matplotlib
- scikit-optimize
- pandas
- seaborn

## Questions or Issues?

Check the console output and log messages for detailed information about what's happening during the study. The scripts include extensive logging to help diagnose any issues.
