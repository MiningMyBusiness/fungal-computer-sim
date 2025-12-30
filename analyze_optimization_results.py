"""Analysis script for systematic optimization study results.

This script loads the results from systematic_optimization_study.py and performs
comprehensive analysis to identify general principles for:
- Optimal electrode distances
- Stimulus parameter patterns
- Fungal characteristics that work best
- Effects of node density on performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==========================================
# Analysis Functions
# ==========================================

def load_latest_results(results_dir="optimization_study_results"):
    """Load the most recent results file."""
    results_path = Path(results_dir)
    csv_files = list(results_path.glob("optimization_results_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Load corresponding config if available
    timestamp = latest_file.stem.split('_', 2)[2]
    config_file = results_path / f"study_config_{timestamp}.json"
    config = None
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_file}")
    
    return df, config

def analyze_node_density_effects(df):
    """Analyze how node density affects performance."""
    print("\n" + "="*70)
    print("NODE DENSITY EFFECTS ANALYSIS")
    print("="*70)
    
    successful = df[df['success'] == True].copy()
    
    # Group by node count
    agg_dict = {
        'score': ['mean', 'std', 'min', 'max', 'count'],
        'tuned_score': ['mean', 'std'],
        'score_improvement': ['mean', 'std'],
        'network_density': 'mean',
        'num_edges': 'mean'
    }
    
    # Add network properties if available
    if 'clustering_coefficient' in successful.columns:
        agg_dict['clustering_coefficient'] = 'mean'
    if 'algebraic_connectivity' in successful.columns:
        agg_dict['algebraic_connectivity'] = 'mean'
    if 'avg_path_length' in successful.columns:
        agg_dict['avg_path_length'] = 'mean'
    if 'modularity' in successful.columns:
        agg_dict['modularity'] = 'mean'
    
    grouped = successful.groupby('num_nodes').agg(agg_dict).round(4)
    
    print("\nPerformance by Node Count:")
    print(grouped)
    
    # Statistical test: correlation between node count and score
    if len(successful) > 0:
        corr_nodes_score, p_value = stats.pearsonr(successful['num_nodes'], successful['score'])
        print(f"\nCorrelation (num_nodes vs score): r={corr_nodes_score:.3f}, p={p_value:.4f}")
        
        corr_density_score, p_value = stats.pearsonr(successful['network_density'], successful['score'])
        print(f"Correlation (network_density vs score): r={corr_density_score:.3f}, p={p_value:.4f}")
    
    return grouped

def analyze_electrode_distances(df):
    """Analyze optimal electrode placement patterns."""
    print("\n" + "="*70)
    print("ELECTRODE DISTANCE ANALYSIS")
    print("="*70)
    
    successful = df[df['success'] == True].copy()
    
    # Correlation between distances and performance
    distance_cols = ['dist_AB', 'dist_A_out', 'dist_B_out', 'dist_avg_input_to_out']
    
    print("\nCorrelation with Score:")
    for col in distance_cols:
        if col in successful.columns:
            corr, p_value = stats.pearsonr(successful[col], successful['score'])
            print(f"  {col}: r={corr:.3f}, p={p_value:.4f}")
    
    # Identify high-performing configurations (top 25%)
    threshold = successful['score'].quantile(0.75)
    high_performers = successful[successful['score'] >= threshold]
    
    print(f"\nHigh-Performing Configurations (top 25%, score >= {threshold:.4f}):")
    print(f"  N = {len(high_performers)}")
    print(f"\nOptimal Distance Ranges:")
    for col in distance_cols:
        if col in high_performers.columns:
            mean = high_performers[col].mean()
            std = high_performers[col].std()
            median = high_performers[col].median()
            q25 = high_performers[col].quantile(0.25)
            q75 = high_performers[col].quantile(0.75)
            print(f"  {col}:")
            print(f"    Mean: {mean:.2f} ± {std:.2f} mm")
            print(f"    Median: {median:.2f} mm")
            print(f"    IQR: [{q25:.2f}, {q75:.2f}] mm")
    
    return high_performers

def analyze_stimulus_parameters(df):
    """Analyze optimal stimulus parameter patterns."""
    print("\n" + "="*70)
    print("STIMULUS PARAMETER ANALYSIS")
    print("="*70)
    
    successful = df[df['success'] == True].copy()
    
    # Correlation with performance
    stimulus_cols = ['voltage', 'duration', 'delay']
    
    print("\nCorrelation with Score:")
    for col in stimulus_cols:
        if col in successful.columns:
            corr, p_value = stats.pearsonr(successful[col], successful['score'])
            print(f"  {col}: r={corr:.3f}, p={p_value:.4f}")
    
    # High-performing configurations
    threshold = successful['score'].quantile(0.75)
    high_performers = successful[successful['score'] >= threshold]
    
    print(f"\nOptimal Stimulus Parameter Ranges (top 25%):")
    for col in stimulus_cols:
        if col in high_performers.columns:
            mean = high_performers[col].mean()
            std = high_performers[col].std()
            median = high_performers[col].median()
            q25 = high_performers[col].quantile(0.25)
            q75 = high_performers[col].quantile(0.75)
            unit = 'V' if col == 'voltage' else 'ms'
            print(f"  {col}:")
            print(f"    Mean: {mean:.2f} ± {std:.2f} {unit}")
            print(f"    Median: {median:.2f} {unit}")
            print(f"    IQR: [{q25:.2f}, {q75:.2f}] {unit}")
    
    return high_performers

def analyze_network_topology(df):
    """Analyze network topology properties and their relationship to performance."""
    print("\n" + "="*70)
    print("NETWORK TOPOLOGY ANALYSIS")
    print("="*70)
    
    successful = df[df['success'] == True].copy()
    
    # Check if network properties are available
    network_props = ['clustering_coefficient', 'algebraic_connectivity', 'avg_path_length', 'modularity']
    available_props = [prop for prop in network_props if prop in successful.columns]
    
    if not available_props:
        print("No network topology properties found in data.")
        return None
    
    print(f"\nAvailable network properties: {', '.join(available_props)}")
    
    # Correlation with performance
    print("\nCorrelation with Score:")
    for prop in available_props:
        corr, p_value = stats.pearsonr(successful[prop], successful['score'])
        print(f"  {prop}: r={corr:.3f}, p={p_value:.4f}")
    
    # High-performing configurations (top 25%)
    threshold = successful['score'].quantile(0.75)
    high_performers = successful[successful['score'] >= threshold]
    
    print(f"\nNetwork Properties in High-Performing Configurations (top 25%):")
    print(f"  N = {len(high_performers)}")
    for prop in available_props:
        mean = high_performers[prop].mean()
        std = high_performers[prop].std()
        median = high_performers[prop].median()
        q25 = high_performers[prop].quantile(0.25)
        q75 = high_performers[prop].quantile(0.75)
        
        # Also show overall statistics for comparison
        overall_mean = successful[prop].mean()
        overall_std = successful[prop].std()
        
        print(f"\n  {prop}:")
        print(f"    High performers: {mean:.4f} ± {std:.4f} (median: {median:.4f})")
        print(f"    IQR: [{q25:.4f}, {q75:.4f}]")
        print(f"    Overall: {overall_mean:.4f} ± {overall_std:.4f}")
        print(f"    Difference: {((mean - overall_mean) / overall_std):.2f} std devs")
    
    # Correlation matrix between network properties
    if len(available_props) > 1:
        print("\nCorrelation Matrix (Network Properties):")
        corr_matrix = successful[available_props].corr()
        print(corr_matrix.round(3))
    
    return high_performers

def analyze_fungal_characteristics(df):
    """Analyze which fungal characteristics work best."""
    print("\n" + "="*70)
    print("FUNGAL CHARACTERISTICS ANALYSIS")
    print("="*70)
    
    successful = df[df['success'] == True].copy()
    tuned = successful[successful['tuned_score'].notna()].copy()
    
    if len(tuned) == 0:
        print("No tuned physics data available.")
        return None
    
    print(f"\nPhysics tuning results: {len(tuned)} trials")
    print(f"Mean improvement: {tuned['score_improvement'].mean():.4f} ± {tuned['score_improvement'].std():.4f}")
    print(f"Improvement range: [{tuned['score_improvement'].min():.4f}, {tuned['score_improvement'].max():.4f}]")
    
    # Analyze tuned parameters
    fungal_params = ['tau_v', 'tau_w', 'a', 'b', 'v_scale', 'R_off', 'R_on', 'alpha']
    
    print("\nOptimal Fungal Parameter Ranges:")
    for param in fungal_params:
        tuned_col = f'tuned_{param}'
        default_col = f'default_{param}'
        
        if tuned_col in tuned.columns:
            mean = tuned[tuned_col].mean()
            std = tuned[tuned_col].std()
            median = tuned[tuned_col].median()
            default = tuned[default_col].iloc[0] if default_col in tuned.columns else None
            
            print(f"  {param}:")
            print(f"    Optimized: {mean:.3f} ± {std:.3f} (median: {median:.3f})")
            if default is not None:
                print(f"    Default: {default:.3f}")
                print(f"    Change: {((mean - default) / default * 100):+.1f}%")
    
    # Identify which parameters correlate most with improvement
    print("\nParameter Changes Correlated with Improvement:")
    for param in fungal_params:
        tuned_col = f'tuned_{param}'
        default_col = f'default_{param}'
        
        if tuned_col in tuned.columns and default_col in tuned.columns:
            # Calculate relative change
            tuned['change_' + param] = (tuned[tuned_col] - tuned[default_col]) / tuned[default_col]
            
            corr, p_value = stats.pearsonr(tuned['change_' + param], tuned['score_improvement'])
            if abs(corr) > 0.1:  # Only show meaningful correlations
                print(f"  {param}: r={corr:.3f}, p={p_value:.4f}")
    
    return tuned

def create_visualizations(df, output_dir="optimization_study_results"):
    """Create comprehensive visualizations of the results."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    output_path = Path(output_dir)
    successful = df[df['success'] == True].copy()
    
    # Figure 1: Node density effects
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Score vs num_nodes
    axes[0, 0].scatter(successful['num_nodes'], successful['score'], alpha=0.5)
    axes[0, 0].set_xlabel('Number of Nodes')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance vs Network Size')
    
    # Box plot by node count
    node_counts = sorted(successful['num_nodes'].unique())
    score_by_nodes = [successful[successful['num_nodes'] == n]['score'].values for n in node_counts]
    axes[0, 1].boxplot(score_by_nodes, labels=node_counts)
    axes[0, 1].set_xlabel('Number of Nodes')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Score Distribution by Network Size')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Network density vs score
    axes[1, 0].scatter(successful['network_density'], successful['score'], alpha=0.5)
    axes[1, 0].set_xlabel('Network Density')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance vs Network Density')
    
    # Tuned vs original scores
    tuned = successful[successful['tuned_score'].notna()]
    if len(tuned) > 0:
        axes[1, 1].scatter(tuned['score'], tuned['tuned_score'], alpha=0.5)
        axes[1, 1].plot([tuned['score'].min(), tuned['score'].max()],
                       [tuned['score'].min(), tuned['score'].max()],
                       'r--', label='No improvement')
        axes[1, 1].set_xlabel('Original Score')
        axes[1, 1].set_ylabel('Tuned Score')
        axes[1, 1].set_title('Physics Tuning Effect')
        axes[1, 1].legend()
    
    plt.tight_layout()
    fig_path = output_path / "node_density_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()
    
    # Figure 2: Electrode distances
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    distance_cols = ['dist_AB', 'dist_A_out', 'dist_B_out', 'dist_avg_input_to_out']
    for idx, col in enumerate(distance_cols):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(successful[col], successful['score'], alpha=0.5)
        ax.set_xlabel(f'{col} (mm)')
        ax.set_ylabel('Score')
        ax.set_title(f'Score vs {col}')
    
    plt.tight_layout()
    fig_path = output_path / "electrode_distance_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()
    
    # Figure 3: Stimulus parameters
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(successful['voltage'], successful['score'], alpha=0.5)
    axes[0, 0].set_xlabel('Voltage (V)')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Score vs Voltage')
    
    axes[0, 1].scatter(successful['duration'], successful['score'], alpha=0.5)
    axes[0, 1].set_xlabel('Duration (ms)')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Score vs Pulse Duration')
    
    axes[1, 0].scatter(successful['delay'], successful['score'], alpha=0.5)
    axes[1, 0].set_xlabel('Delay (ms)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Score vs Electrode Delay')
    
    # Voltage vs duration colored by score
    scatter = axes[1, 1].scatter(successful['voltage'], successful['duration'], 
                                 c=successful['score'], cmap='viridis', alpha=0.6)
    axes[1, 1].set_xlabel('Voltage (V)')
    axes[1, 1].set_ylabel('Duration (ms)')
    axes[1, 1].set_title('Voltage-Duration Space (colored by score)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Score')
    
    plt.tight_layout()
    fig_path = output_path / "stimulus_parameter_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()
    
    # Figure 4: Fungal parameters (if available)
    tuned = successful[successful['tuned_score'].notna()]
    if len(tuned) > 0:
        fungal_params = ['tau_v', 'tau_w', 'a', 'b', 'v_scale', 'R_off', 'R_on', 'alpha']
        n_params = len(fungal_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for idx, param in enumerate(fungal_params):
            tuned_col = f'tuned_{param}'
            default_col = f'default_{param}'
            
            if tuned_col in tuned.columns:
                axes[idx].scatter(tuned[tuned_col], tuned['tuned_score'], alpha=0.5)
                
                # Add vertical line for default value
                if default_col in tuned.columns:
                    default_val = tuned[default_col].iloc[0]
                    axes[idx].axvline(default_val, color='r', linestyle='--', 
                                     label=f'Default: {default_val:.3f}')
                
                axes[idx].set_xlabel(f'{param}')
                axes[idx].set_ylabel('Tuned Score')
                axes[idx].set_title(f'Tuned Score vs {param}')
                axes[idx].legend()
        
        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        fig_path = output_path / "fungal_parameter_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.close()
    
    # Figure 5: Network topology properties (if available)
    network_props = ['clustering_coefficient', 'algebraic_connectivity', 'avg_path_length', 'modularity']
    available_props = [prop for prop in network_props if prop in successful.columns]
    
    if available_props:
        n_props = len(available_props)
        n_cols = 2
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, prop in enumerate(available_props):
            axes[idx].scatter(successful[prop], successful['score'], alpha=0.5, color='purple')
            axes[idx].set_xlabel(prop.replace('_', ' ').title())
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'Score vs {prop.replace("_", " ").title()}')
            
            # Add correlation coefficient to plot
            corr, p_value = stats.pearsonr(successful[prop], successful['score'])
            axes[idx].text(0.05, 0.95, f'r={corr:.3f}, p={p_value:.4f}',
                          transform=axes[idx].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(n_props, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        fig_path = output_path / "network_topology_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.close()
    
    print("Visualization complete!")

def generate_summary_report(df, output_dir="optimization_study_results"):
    """Generate a comprehensive text summary report."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    output_path = Path(output_dir)
    report_file = output_path / "analysis_summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SYSTEMATIC XOR GATE OPTIMIZATION STUDY - ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        successful = df[df['success'] == True]
        f.write(f"Total trials: {len(df)}\n")
        f.write(f"Successful trials: {len(successful)}\n")
        f.write(f"Success rate: {len(successful)/len(df)*100:.1f}%\n\n")
        
        # General principles
        f.write("="*70 + "\n")
        f.write("GENERAL PRINCIPLES IDENTIFIED\n")
        f.write("="*70 + "\n\n")
        
        # Electrode distances
        threshold = successful['score'].quantile(0.75)
        high_performers = successful[successful['score'] >= threshold]
        
        f.write("1. OPTIMAL ELECTRODE PLACEMENT:\n")
        f.write(f"   Based on top 25% performers (N={len(high_performers)}):\n\n")
        
        dist_AB_mean = high_performers['dist_AB'].mean()
        dist_AB_std = high_performers['dist_AB'].std()
        f.write(f"   - Distance between input electrodes (A-B):\n")
        f.write(f"     {dist_AB_mean:.2f} ± {dist_AB_std:.2f} mm\n\n")
        
        dist_out_mean = high_performers['dist_avg_input_to_out'].mean()
        dist_out_std = high_performers['dist_avg_input_to_out'].std()
        f.write(f"   - Distance from inputs to output:\n")
        f.write(f"     {dist_out_mean:.2f} ± {dist_out_std:.2f} mm\n\n")
        
        # Stimulus parameters
        f.write("2. OPTIMAL STIMULUS PARAMETERS:\n")
        f.write(f"   Based on top 25% performers:\n\n")
        
        voltage_mean = high_performers['voltage'].mean()
        voltage_std = high_performers['voltage'].std()
        f.write(f"   - Voltage: {voltage_mean:.2f} ± {voltage_std:.2f} V\n")
        
        duration_mean = high_performers['duration'].mean()
        duration_std = high_performers['duration'].std()
        f.write(f"   - Pulse duration: {duration_mean:.1f} ± {duration_std:.1f} ms\n")
        
        delay_mean = high_performers['delay'].mean()
        delay_std = high_performers['delay'].std()
        f.write(f"   - Electrode delay: {delay_mean:.1f} ± {delay_std:.1f} ms\n\n")
        
        # Node density effects
        corr_nodes, p_nodes = stats.pearsonr(successful['num_nodes'], successful['score'])
        f.write("3. NODE DENSITY EFFECTS:\n")
        f.write(f"   Correlation (num_nodes vs score): r={corr_nodes:.3f}, p={p_nodes:.4f}\n")
        
        if abs(corr_nodes) < 0.1:
            f.write("   → Node density has MINIMAL effect on performance\n")
        elif corr_nodes > 0:
            f.write("   → Higher node density IMPROVES performance\n")
        else:
            f.write("   → Lower node density IMPROVES performance\n")
        f.write("\n")
        
        # Network topology properties
        network_props = ['clustering_coefficient', 'algebraic_connectivity', 'avg_path_length', 'modularity']
        available_props = [prop for prop in network_props if prop in successful.columns]
        
        if available_props:
            f.write("4. NETWORK TOPOLOGY EFFECTS:\n")
            f.write(f"   Correlations with performance:\n\n")
            
            for prop in available_props:
                corr, p_value = stats.pearsonr(successful[prop], successful['score'])
                f.write(f"   - {prop}: r={corr:.3f}, p={p_value:.4f}\n")
                
                # Interpretation
                if p_value < 0.05:
                    if abs(corr) > 0.3:
                        direction = "STRONG positive" if corr > 0 else "STRONG negative"
                    elif abs(corr) > 0.1:
                        direction = "Moderate positive" if corr > 0 else "Moderate negative"
                    else:
                        direction = "Weak"
                    f.write(f"     → {direction} correlation (significant)\n")
                else:
                    f.write(f"     → No significant correlation\n")
            
            f.write(f"\n   Optimal ranges (top 25% performers):\n")
            for prop in available_props:
                mean = high_performers[prop].mean()
                std = high_performers[prop].std()
                f.write(f"   - {prop}: {mean:.4f} ± {std:.4f}\n")
            f.write("\n")
        
        # Fungal characteristics
        tuned = successful[successful['tuned_score'].notna()]
        if len(tuned) > 0:
            section_num = "5" if available_props else "4"
            f.write(f"{section_num}. OPTIMAL FUNGAL CHARACTERISTICS:\n")
            f.write(f"   Based on physics tuning (N={len(tuned)}):\n\n")
            
            improvement_mean = tuned['score_improvement'].mean()
            improvement_std = tuned['score_improvement'].std()
            f.write(f"   Mean improvement from tuning: {improvement_mean:.4f} ± {improvement_std:.4f}\n\n")
            
            fungal_params = ['tau_v', 'tau_w', 'a', 'b', 'v_scale', 'R_off', 'R_on', 'alpha']
            for param in fungal_params:
                tuned_col = f'tuned_{param}'
                default_col = f'default_{param}'
                
                if tuned_col in tuned.columns:
                    mean = tuned[tuned_col].mean()
                    std = tuned[tuned_col].std()
                    default = tuned[default_col].iloc[0]
                    change_pct = (mean - default) / default * 100
                    
                    f.write(f"   - {param}:\n")
                    f.write(f"     Optimal: {mean:.3f} ± {std:.3f}\n")
                    f.write(f"     Default: {default:.3f}\n")
                    f.write(f"     Change: {change_pct:+.1f}%\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"Report saved: {report_file}")

# ==========================================
# Main Analysis
# ==========================================

def main():
    """Run complete analysis pipeline."""
    print("\n" + "="*70)
    print("SYSTEMATIC OPTIMIZATION STUDY - ANALYSIS")
    print("="*70)
    
    # Load results
    df, config = load_latest_results()
    
    print(f"\nLoaded {len(df)} trials")
    if config:
        print(f"Study configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Run analyses
    analyze_node_density_effects(df)
    analyze_network_topology(df)
    analyze_electrode_distances(df)
    analyze_stimulus_parameters(df)
    analyze_fungal_characteristics(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate summary report
    generate_summary_report(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
