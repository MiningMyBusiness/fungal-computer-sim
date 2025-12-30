# Quick Start Guide - Systematic Optimization Study

## What Was Created

I've created a complete framework for systematically studying XOR gate optimization with varying parameters. Here's what you have:

### Scripts

1. **`systematic_optimization_study.py`** - Main study script
   - Runs 130 trials (13 node counts × 10 trials each)
   - Tests node counts: 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150
   - Each trial optimizes electrode placement AND fungal physics
   - Estimated time: 10-30 hours

2. **`pilot_study.py`** - Quick test version
   - Runs 9 trials (3 node counts × 3 trials each)
   - Tests node counts: 25, 40, 60
   - Estimated time: 30-70 minutes
   - **Start here to test everything works!**

3. **`analyze_optimization_results.py`** - Analysis script
   - Loads results and performs statistical analysis
   - Generates visualizations
   - Creates summary report with key findings

4. **`OPTIMIZATION_STUDY_README.md`** - Detailed documentation
   - Complete explanation of all features
   - Data format descriptions
   - Customization guide

## Quick Start (Recommended)

### Step 1: Run Pilot Study
```bash
python pilot_study.py
```
- Takes 30-70 minutes
- Tests the framework
- Gives you initial results to explore

### Step 2: Analyze Pilot Results
```bash
python analyze_optimization_results.py
```
- Generates plots and statistics
- Creates summary report
- Check `pilot_study_results/` directory

### Step 3: Run Full Study (Optional)
If pilot looks good, run the full study:
```bash
python systematic_optimization_study.py
```
- Takes 10-30 hours
- Can run overnight or in background
- Saves checkpoints (safe to interrupt)

### Step 4: Analyze Full Results
```bash
python analyze_optimization_results.py
```
- Automatically loads most recent results
- Generates comprehensive analysis

## What You'll Learn

The study will identify:

### 1. Optimal Electrode Placement
- How far apart should input electrodes A and B be?
- How far should the output probe be from the inputs?
- Are there geometric patterns that work best?

### 2. Optimal Stimulus Parameters
- What voltage works best?
- What pulse duration?
- What delay between electrodes?
- Do these depend on network size?

### 3. Optimal Fungal Characteristics
- Which FitzHugh-Nagumo parameters work best?
- Which memristor parameters work best?
- Are there universal optimal values?
- Or do they depend on network configuration?

### 4. Node Density Effects
- Does more nodes = better performance?
- Is there an optimal node density?
- How does network connectivity affect results?

## Output Files

### During Study
- `checkpoint_<timestamp>.csv` - Saved after each trial (safe to interrupt)
- `study_config_<timestamp>.json` - Study configuration

### After Study
- `optimization_results_<timestamp>.csv` - Complete results
- `node_density_analysis.png` - Network size effects
- `electrode_distance_analysis.png` - Placement patterns
- `stimulus_parameter_analysis.png` - Stimulus relationships
- `fungal_parameter_analysis.png` - Physics optimization
- `analysis_summary_report.txt` - Key findings in text format

## Key Features

### Systematic Node Count Variation
Node counts progress from sparse to dense:
- Sparse: 20, 25, 30
- Medium: 35, 40, 45, 50
- Higher: 60, 70, 80
- Dense: 100, 120, 150

This allows analysis of patterns as density increases.

### Multiple Random Trials
Each node count is tested with 10 different random seeds:
- Accounts for network topology variation
- Provides statistical robustness
- Enables confidence intervals

### Physics Tuning Enabled
Each trial optimizes:
1. **First**: Electrode positions and stimulus (15 iterations)
2. **Then**: Fungal physics parameters (15 iterations)

This identifies both optimal placement AND optimal physics.

### Automatic Checkpointing
- Results saved after each trial
- Safe to interrupt and resume
- No data loss if stopped early

## Tips

### For Quick Testing
Edit `pilot_study.py` to make it even faster:
```python
NODE_COUNTS = [30, 50]  # Just 2 node counts
TRIALS_PER_CONFIG = 2   # Just 2 trials each
N_CALLS = 8             # Fewer optimization calls
TUNE_PHYSICS = False    # Skip physics tuning
```

### For Long Studies
Use `screen` or `tmux`:
```bash
screen -S optimization
python systematic_optimization_study.py
# Press Ctrl-A, then D to detach
# Reattach later with: screen -r optimization
```

### For Custom Studies
Edit the configuration in the scripts:
- `NODE_COUNTS`: Which node counts to test
- `TRIALS_PER_CONFIG`: How many random trials
- `N_CALLS`: Optimization iterations (more = better but slower)
- `TUNE_PHYSICS`: Enable/disable physics tuning

## Example Workflow

```bash
# 1. Test with pilot study
python pilot_study.py
# Wait 30-70 minutes

# 2. Check pilot results
python analyze_optimization_results.py
# Review plots and summary report

# 3. If satisfied, run full study
screen -S full_study
python systematic_optimization_study.py
# Detach with Ctrl-A, D
# Check back in 10-30 hours

# 4. Analyze full results
python analyze_optimization_results.py
# Review comprehensive analysis
```

## Understanding Results

### High Score = Good
- Score measures XOR gate quality
- Higher score = better separation between high/low outputs
- Look for patterns in high-scoring configurations

### Key Metrics to Watch
- **dist_AB**: Distance between input electrodes
- **voltage**: Stimulation voltage
- **duration**: Pulse duration
- **tuned_score**: Performance after physics optimization
- **score_improvement**: How much physics tuning helped

### Statistical Significance
- Correlations with p < 0.05 are significant
- Look for consistent patterns across trials
- High-performing configs (top 25%) show optimal ranges

## Next Steps

1. **Run pilot study** to test everything
2. **Review pilot results** to understand output format
3. **Adjust parameters** if needed for your goals
4. **Run full study** for comprehensive analysis
5. **Analyze results** to extract general principles

## Questions?

- Check `OPTIMIZATION_STUDY_README.md` for detailed documentation
- Look at console output for progress and diagnostics
- Examine checkpoint files to see intermediate results
- All scripts have extensive logging for debugging

Good luck with your systematic study! 🍄🔬
