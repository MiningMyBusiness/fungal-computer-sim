"""Train machine learning models to predict fungal parameters from system identification features.

This script loads characterization data and trains models to predict the fungal constants
(tau_v, tau_w, a, b, v_scale, R_off, R_on, alpha) from the response features extracted
from step response, paired-pulse, and triangle sweep protocols.

Models used:
- RandomForestRegressor: Ensemble tree-based model
- MLPRegressor: Neural network model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==========================================
# Configuration
# ==========================================

# Target parameters to predict
TARGET_PARAMS = ['tau_v', 'tau_w', 'a', 'b', 'v_scale', 'R_off', 'R_on', 'alpha', 'num_nodes',
                 'num_edges', 'network_density']

# Response features to use as input
FEATURE_COLUMNS = [
    # Step response features
    'step_rise_time',
    'step_saturation_voltage',
    'step_oscillation_index',
    
    # Paired-pulse features
    'pp_recovery_ratio_delay_200',
    'pp_recovery_ratio_delay_800',
    'pp_recovery_ratio_delay_2000',
    'pp_first_peak_delay_200',
    'pp_first_peak_delay_800',
    'pp_first_peak_delay_2000',
    'pp_second_peak_delay_200',
    'pp_second_peak_delay_800',
    'pp_second_peak_delay_2000',
    'pp_avg_recovery_ratio',
    'pp_std_recovery_ratio',
    'pp_recovery_slope',
    
    # Triangle sweep features
    'tri_hysteresis_area',
]

# Output directory
OUTPUT_DIR = Path("ml_models")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# Data Loading and Preprocessing
# ==========================================

def load_and_preprocess_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load characterization data and prepare for ML training.
    
    Args:
        csv_path: Path to characterization results CSV
        
    Returns:
        Tuple of (full_df, X, y) where:
            - full_df: Complete dataframe
            - X: Feature matrix
            - y: Target matrix
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter successful trials only
    df_success = df[df['characterization_success'] == True].copy()
    print(f"Loaded {len(df)} total records, {len(df_success)} successful")
    
    if len(df_success) == 0:
        raise ValueError("No successful trials found in dataset!")
    
    # Check for missing features
    missing_features = [f for f in FEATURE_COLUMNS if f not in df_success.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        available_features = [f for f in FEATURE_COLUMNS if f in df_success.columns]
    else:
        available_features = FEATURE_COLUMNS
    
    # Extract features and targets
    X = df_success[available_features].copy()
    y = df_success[TARGET_PARAMS].copy()
    
    # Handle any remaining NaN and inf values
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    
    # Replace inf with NaN for easier handling
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    
    # Check for NaN values (including converted inf values)
    nan_features = X.isna().sum()
    nan_targets = y.isna().sum()
    
    if nan_features.any():
        print("\nNaN/inf values in features:")
        print(nan_features[nan_features > 0])
    
    if nan_targets.any():
        print("\nNaN/inf values in targets:")
        print(nan_targets[nan_targets > 0])
    
    if nan_features.any() or nan_targets.any():
        print("Dropping rows with NaN/inf values...")
        valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        df_success = df_success[valid_idx]
        print(f"Remaining samples: {len(X)}")
    
    return df_success, X, y

# ==========================================
# Model Training
# ==========================================

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                       param_name: str) -> RandomForestRegressor:
    """Train a Random Forest model for a single parameter.
    
    Args:
        X_train: Training features
        y_train: Training target (single parameter)
        param_name: Name of parameter being predicted
        
    Returns:
        Trained RandomForestRegressor
    """
    print(f"  Training Random Forest for {param_name}...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def train_mlp(X_train: np.ndarray, y_train: np.ndarray, 
              param_name: str) -> MLPRegressor:
    """Train a Multi-Layer Perceptron model for a single parameter.
    
    Args:
        X_train: Training features
        y_train: Training target (single parameter)
        param_name: Name of parameter being predicted
        
    Returns:
        Trained MLPRegressor
    """
    print(f"  Training MLP for {param_name}...")
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train: np.ndarray, y_train: pd.DataFrame,
                    X_train_scaled: np.ndarray) -> Dict:
    """Train both RF and MLP models for all target parameters.
    
    Args:
        X_train: Training features (unscaled, for RF)
        y_train: Training targets (all parameters)
        X_train_scaled: Training features (scaled, for MLP)
        
    Returns:
        Dictionary containing all trained models
    """
    models = {
        'random_forest': {},
        'mlp': {}
    }
    
    print("\nTraining Random Forest models...")
    for param in TARGET_PARAMS:
        models['random_forest'][param] = train_random_forest(
            X_train, y_train[param].values, param
        )
    
    print("\nTraining MLP models...")
    for param in TARGET_PARAMS:
        models['mlp'][param] = train_mlp(
            X_train_scaled, y_train[param].values, param
        )
    
    return models

# ==========================================
# Model Evaluation
# ==========================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  param_name: str) -> Dict:
    """Evaluate a single model's performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        param_name: Name of parameter
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    }
    
    return metrics, y_pred

def evaluate_all_models(models: Dict, X_test: np.ndarray, y_test: pd.DataFrame,
                       X_test_scaled: np.ndarray) -> pd.DataFrame:
    """Evaluate all models and compile results.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features (unscaled)
        y_test: Test targets
        X_test_scaled: Test features (scaled)
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    predictions = {}
    
    print("\nEvaluating models...")
    
    for model_type in ['random_forest', 'mlp']:
        print(f"\n{model_type.upper()}:")
        predictions[model_type] = {}
        
        for param in TARGET_PARAMS:
            model = models[model_type][param]
            
            # Use scaled features for MLP, unscaled for RF
            X = X_test_scaled if model_type == 'mlp' else X_test
            
            metrics, y_pred = evaluate_model(
                model, X, y_test[param].values, param
            )
            
            predictions[model_type][param] = y_pred
            
            results.append({
                'model': model_type,
                'parameter': param,
                **metrics
            })
            
            print(f"  {param:12s}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df, predictions

# ==========================================
# Feature Importance Analysis
# ==========================================

def analyze_feature_importance(models: Dict, feature_names: List[str]) -> pd.DataFrame:
    """Extract and analyze feature importance from Random Forest models.
    
    Args:
        models: Dictionary of trained models
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance scores
    """
    importance_data = []
    
    for param in TARGET_PARAMS:
        rf_model = models['random_forest'][param]
        importances = rf_model.feature_importances_
        
        for feat_name, importance in zip(feature_names, importances):
            importance_data.append({
                'parameter': param,
                'feature': feat_name,
                'importance': importance
            })
    
    importance_df = pd.DataFrame(importance_data)
    return importance_df

# ==========================================
# Visualization
# ==========================================

def plot_predictions(y_test: pd.DataFrame, predictions: Dict, 
                    output_dir: Path):
    """Create prediction vs actual plots for all models and parameters.
    
    Args:
        y_test: True test values
        predictions: Dictionary of predictions
        output_dir: Directory to save plots
    """
    print("\nGenerating prediction plots...")
    
    for model_type in ['random_forest', 'mlp']:
        n_params = len(TARGET_PARAMS)
        n_cols = 4
        n_rows = int(np.ceil(n_params / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        for idx, param in enumerate(TARGET_PARAMS):
            ax = axes[idx]
            
            y_true = y_test[param].values
            y_pred = predictions[model_type][param]
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
            
            # Calculate R²
            r2 = r2_score(y_true, y_pred)
            
            ax.set_xlabel(f'True {param}', fontsize=11)
            ax.set_ylabel(f'Predicted {param}', fontsize=11)
            ax.set_title(f'{param} (R²={r2:.3f})', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(TARGET_PARAMS), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{model_type.upper()} Model Predictions', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = output_dir / f'{model_type}_predictions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()

def plot_feature_importance(importance_df: pd.DataFrame, output_dir: Path):
    """Plot feature importance for each parameter.
    
    Args:
        importance_df: DataFrame with feature importance scores
        output_dir: Directory to save plots
    """
    print("\nGenerating feature importance plots...")
    
    n_params = len(TARGET_PARAMS)
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, param in enumerate(TARGET_PARAMS):
        ax = axes[idx]
        
        param_data = importance_df[importance_df['parameter'] == param].copy()
        param_data = param_data.sort_values('importance', ascending=True).tail(10)
        
        ax.barh(param_data['feature'], param_data['importance'], color='steelblue')
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title(f'{param}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3, axis='x')
    
    # Hide unused subplots
    for idx in range(len(TARGET_PARAMS), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Top 10 Feature Importances (Random Forest)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_path = output_dir / 'feature_importance.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

def plot_model_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison plots between RF and MLP models.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
    """
    print("\nGenerating model comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['r2', 'rmse', 'mae']
    titles = ['R² Score (higher is better)', 'RMSE (lower is better)', 'MAE (lower is better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        pivot_data = results_df.pivot(index='parameter', columns='model', values=metric)
        
        x = np.arange(len(TARGET_PARAMS))
        width = 0.35
        
        ax.bar(x - width/2, pivot_data['random_forest'], width, 
               label='Random Forest', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, pivot_data['mlp'], width, 
               label='MLP', color='coral', alpha=0.8)
        
        ax.set_xlabel('Parameter', fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_PARAMS, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# ==========================================
# Main Training Pipeline
# ==========================================

def main(csv_path: Path, test_size: float = 0.2):
    """Main training pipeline.
    
    Args:
        csv_path: Path to characterization results CSV
        test_size: Fraction of data to use for testing
    """
    print("="*70)
    print("FUNGAL PARAMETER PREDICTION - ML MODEL TRAINING")
    print("="*70)
    
    # Load and preprocess data
    df, X, y = load_and_preprocess_data(csv_path)
    
    # Split data
    print(f"\nSplitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features for MLP
    print("\nScaling features for MLP...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = train_all_models(X_train.values, y_train, X_train_scaled)
    
    # Evaluate models
    results_df, predictions = evaluate_all_models(
        models, X_test.values, y_test, X_test_scaled
    )
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    importance_df = analyze_feature_importance(models, X.columns.tolist())
    
    # Save models and results
    print("\nSaving models and results...")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save models
    for model_type in ['random_forest', 'mlp']:
        for param in TARGET_PARAMS:
            model_path = OUTPUT_DIR / f'{model_type}_{param}_{timestamp}.pkl'
            joblib.dump(models[model_type][param], model_path)
    
    # Save scaler
    scaler_path = OUTPUT_DIR / f'scaler_{timestamp}.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler: {scaler_path}")
    
    # Save results
    results_path = OUTPUT_DIR / f'evaluation_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  Saved results: {results_path}")
    
    importance_path = OUTPUT_DIR / f'feature_importance_{timestamp}.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"  Saved importance: {importance_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'data_file': str(csv_path),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'features': X.columns.tolist(),
        'targets': TARGET_PARAMS,
        'test_size': test_size,
    }
    metadata_path = OUTPUT_DIR / f'training_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    # Generate visualizations
    plot_predictions(y_test, predictions, OUTPUT_DIR)
    plot_feature_importance(importance_df, OUTPUT_DIR)
    plot_model_comparison(results_df, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    print("\nBest performing model for each parameter (by R²):")
    for param in TARGET_PARAMS:
        param_results = results_df[results_df['parameter'] == param]
        best_model = param_results.loc[param_results['r2'].idxmax(), 'model']
        best_r2 = param_results['r2'].max()
        print(f"  {param:12s}: {best_model:15s} (R²={best_r2:.4f})")
    
    print("\nOverall performance:")
    for model_type in ['random_forest', 'mlp']:
        model_results = results_df[results_df['model'] == model_type]
        print(f"\n  {model_type.upper()}:")
        print(f"    Mean R²:   {model_results['r2'].mean():.4f} ± {model_results['r2'].std():.4f}")
        print(f"    Mean RMSE: {model_results['rmse'].mean():.4f} ± {model_results['rmse'].std():.4f}")
        print(f"    Mean MAE:  {model_results['mae'].mean():.4f} ± {model_results['mae'].std():.4f}")
    
    print("\n" + "="*70)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*70)

# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train ML models to predict fungal parameters from characterization features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Train on a specific results file
  python train_parameter_predictor.py --data characterization_study_results/characterization_results_20260105_144914.csv
  
  # Use different train/test split
  python train_parameter_predictor.py --data results.csv --test-size 0.3
        """
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to characterization results CSV file'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        exit(1)
    
    main(csv_path, test_size=args.test_size)
