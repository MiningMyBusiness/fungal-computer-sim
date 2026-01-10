"""Advanced ML training with hyperparameter optimization and ensemble methods.

This enhanced version includes:
- Gradient Boosting models (XGBoost, LightGBM)
- Hyperparameter grid search with cross-validation
- Feature engineering (polynomial features, interactions)
- Ensemble stacking methods
- More robust evaluation and model selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==========================================
# Configuration
# ==========================================

TARGET_PARAMS = ['tau_v', 'tau_w', 'a', 'b', 'v_scale', 'R_off', 'R_on', 'alpha', 'num_nodes',
                 'num_edges', 'network_density']

FEATURE_COLUMNS = [
    'step_rise_time',
    'step_saturation_voltage',
    'step_oscillation_index',
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
    'tri_hysteresis_area',
]

OUTPUT_DIR = Path("ml_models_advanced")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# Data Loading and Preprocessing
# ==========================================

def load_and_preprocess_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load characterization data and prepare for ML training."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    df_success = df[df['characterization_success'] == True].copy()
    print(f"Loaded {len(df)} total records, {len(df_success)} successful")
    
    if len(df_success) == 0:
        raise ValueError("No successful trials found in dataset!")
    
    missing_features = [f for f in FEATURE_COLUMNS if f not in df_success.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        available_features = [f for f in FEATURE_COLUMNS if f in df_success.columns]
    else:
        available_features = FEATURE_COLUMNS
    
    X = df_success[available_features].copy()
    y = df_success[TARGET_PARAMS].copy()
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    
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
# Feature Engineering
# ==========================================

def engineer_features(X: pd.DataFrame, degree: int = 2, 
                     include_interactions: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Create polynomial and interaction features.
    
    Args:
        X: Input features
        degree: Polynomial degree
        include_interactions: Whether to include interaction terms
        
    Returns:
        Tuple of (engineered features, feature names)
    """
    print(f"\nEngineering features (degree={degree}, interactions={include_interactions})...")
    
    poly = PolynomialFeatures(
        degree=degree, 
        interaction_only=not include_interactions,
        include_bias=False
    )
    
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    
    print(f"  Original features: {X.shape[1]}")
    print(f"  Engineered features: {X_poly.shape[1]}")
    
    return X_poly, feature_names.tolist(), poly

# ==========================================
# Model Definitions with Hyperparameter Grids
# ==========================================

def get_model_configs(fast_mode: bool = False):
    """Get model configurations with hyperparameter grids.
    
    Args:
        fast_mode: If True, use smaller grids for faster training
        
    Returns:
        Dictionary of model configurations
    """
    configs = {}
    
    # Random Forest
    if fast_mode:
        rf_param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [20, 30],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
        }
    else:
        rf_param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
        }
    
    configs['random_forest'] = {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'param_grid': rf_param_grid,
        'needs_scaling': False
    }
    
    # Gradient Boosting
    if fast_mode:
        gb_param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [5, 7],
            'min_samples_split': [5, 10],
        }
    else:
        gb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0],
        }
    
    configs['gradient_boosting'] = {
        'model': GradientBoostingRegressor(random_state=42),
        'param_grid': gb_param_grid,
        'needs_scaling': False
    }
    
    # XGBoost
    if HAS_XGBOOST:
        if fast_mode:
            xgb_param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 7],
                'subsample': [0.8, 1.0],
            }
        else:
            xgb_param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2],
            }
        
        configs['xgboost'] = {
            'model': xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
            'param_grid': xgb_param_grid,
            'needs_scaling': False
        }
    
    # LightGBM
    if HAS_LIGHTGBM:
        if fast_mode:
            lgb_param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 63],
                'max_depth': [5, 7],
            }
        else:
            lgb_param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127],
                'max_depth': [5, 7, 9, -1],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            }
        
        configs['lightgbm'] = {
            'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'param_grid': lgb_param_grid,
            'needs_scaling': False
        }
    
    # MLP (Neural Network)
    if fast_mode:
        mlp_param_grid = {
            'hidden_layer_sizes': [(128, 64), (128, 64, 32)],
            'alpha': [0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
        }
    else:
        mlp_param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.0001, 0.001, 0.01],
            'activation': ['relu', 'tanh'],
        }
    
    configs['mlp'] = {
        'model': MLPRegressor(
            solver='adam',
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        ),
        'param_grid': mlp_param_grid,
        'needs_scaling': True
    }
    
    return configs

# ==========================================
# Model Training with Grid Search
# ==========================================

def train_model_with_gridsearch(X_train: np.ndarray, y_train: np.ndarray,
                                model_config: Dict, param_name: str,
                                cv: int = 5) -> Tuple:
    """Train a model with grid search hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_config: Model configuration dictionary
        param_name: Name of parameter being predicted
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (best model, best params, best score)
    """
    grid_search = GridSearchCV(
        model_config['model'],
        model_config['param_grid'],
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def train_all_models(X_train: np.ndarray, y_train: pd.DataFrame,
                    X_train_scaled: np.ndarray, model_configs: Dict,
                    cv: int = 5) -> Dict:
    """Train all models with hyperparameter optimization.
    
    Args:
        X_train: Training features (unscaled)
        y_train: Training targets
        X_train_scaled: Training features (scaled)
        model_configs: Dictionary of model configurations
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary containing all trained models and metadata
    """
    models = {}
    best_params = {}
    cv_scores = {}
    
    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} models with grid search...")
        print(f"{'='*60}")
        
        models[model_name] = {}
        best_params[model_name] = {}
        cv_scores[model_name] = {}
        
        for param in TARGET_PARAMS:
            print(f"\n  {param}...")
            
            X = X_train_scaled if config['needs_scaling'] else X_train
            
            best_model, best_param, best_score = train_model_with_gridsearch(
                X, y_train[param].values, config, param, cv
            )
            
            models[model_name][param] = best_model
            best_params[model_name][param] = best_param
            cv_scores[model_name][param] = best_score
            
            print(f"    Best CV R²: {best_score:.4f}")
            print(f"    Best params: {best_param}")
    
    return {
        'models': models,
        'best_params': best_params,
        'cv_scores': cv_scores
    }

# ==========================================
# Ensemble Stacking
# ==========================================

def create_stacking_ensemble(base_models: List, X_train: np.ndarray, 
                             y_train: np.ndarray, param_name: str):
    """Create a stacking ensemble from base models.
    
    Args:
        base_models: List of (name, model) tuples
        X_train: Training features
        y_train: Training target
        param_name: Name of parameter
        
    Returns:
        Trained stacking ensemble
    """
    print(f"  Creating stacking ensemble for {param_name}...")
    
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    return stacking_model

def create_all_ensembles(trained_models: Dict, X_train: np.ndarray,
                        X_train_scaled: np.ndarray, y_train: pd.DataFrame,
                        model_configs: Dict) -> Dict:
    """Create stacking ensembles for all parameters.
    
    Args:
        trained_models: Dictionary of trained models
        X_train: Training features (unscaled)
        X_train_scaled: Training features (scaled)
        y_train: Training targets
        model_configs: Model configurations
        
    Returns:
        Dictionary of ensemble models
    """
    print(f"\n{'='*60}")
    print("Creating Stacking Ensembles...")
    print(f"{'='*60}")
    
    ensembles = {}
    
    for param in TARGET_PARAMS:
        base_models = []
        
        for model_name in trained_models['models'].keys():
            model = trained_models['models'][model_name][param]
            base_models.append((model_name, model))
        
        ensemble = create_stacking_ensemble(
            base_models, X_train, y_train[param].values, param
        )
        ensembles[param] = ensemble
    
    return ensembles

# ==========================================
# Model Evaluation
# ==========================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate a single model's performance."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    }
    
    return metrics, y_pred

def evaluate_all_models(trained_models: Dict, ensembles: Dict,
                       X_test: np.ndarray, y_test: pd.DataFrame,
                       X_test_scaled: np.ndarray, model_configs: Dict) -> pd.DataFrame:
    """Evaluate all models including ensembles."""
    results = []
    predictions = {}
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    for model_name in trained_models['models'].keys():
        print(f"\n{model_name.upper()}:")
        predictions[model_name] = {}
        
        for param in TARGET_PARAMS:
            model = trained_models['models'][model_name][param]
            
            X = X_test_scaled if model_configs[model_name]['needs_scaling'] else X_test
            
            metrics, y_pred = evaluate_model(model, X, y_test[param].values)
            predictions[model_name][param] = y_pred
            
            results.append({
                'model': model_name,
                'parameter': param,
                **metrics
            })
            
            print(f"  {param:15s}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    print(f"\nSTACKING ENSEMBLE:")
    predictions['stacking'] = {}
    
    for param in TARGET_PARAMS:
        ensemble = ensembles[param]
        
        metrics, y_pred = evaluate_model(ensemble, X_test, y_test[param].values)
        predictions['stacking'][param] = y_pred
        
        results.append({
            'model': 'stacking',
            'parameter': param,
            **metrics
        })
        
        print(f"  {param:15s}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df, predictions

# ==========================================
# Visualization
# ==========================================

def plot_predictions(y_test: pd.DataFrame, predictions: Dict, output_dir: Path):
    """Create prediction vs actual plots."""
    print("\nGenerating prediction plots...")
    
    all_models = list(predictions.keys())
    
    for model_type in all_models:
        n_params = len(TARGET_PARAMS)
        n_cols = 4
        n_rows = int(np.ceil(n_params / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        for idx, param in enumerate(TARGET_PARAMS):
            ax = axes[idx]
            
            y_true = y_test[param].values
            y_pred = predictions[model_type][param]
            
            ax.scatter(y_true, y_pred, alpha=0.6, s=50)
            
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
            
            r2 = r2_score(y_true, y_pred)
            
            ax.set_xlabel(f'True {param}', fontsize=11)
            ax.set_ylabel(f'Predicted {param}', fontsize=11)
            ax.set_title(f'{param} (R²={r2:.3f})', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for idx in range(len(TARGET_PARAMS), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{model_type.upper()} Model Predictions', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = output_dir / f'{model_type}_predictions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()

def plot_model_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison plots between all models."""
    print("\nGenerating model comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    metrics = ['r2', 'rmse', 'mae']
    titles = ['R² Score (higher is better)', 'RMSE (lower is better)', 'MAE (lower is better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        pivot_data = results_df.pivot(index='parameter', columns='model', values=metric)
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Parameter', fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticklabels(TARGET_PARAMS, rotation=45, ha='right')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

def plot_best_model_summary(results_df: pd.DataFrame, output_dir: Path):
    """Plot summary of best model for each parameter."""
    print("\nGenerating best model summary...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    best_models = []
    best_r2s = []
    
    for param in TARGET_PARAMS:
        param_results = results_df[results_df['parameter'] == param]
        best_idx = param_results['r2'].idxmax()
        best_model = param_results.loc[best_idx, 'model']
        best_r2 = param_results.loc[best_idx, 'r2']
        
        best_models.append(best_model)
        best_r2s.append(best_r2)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(best_models))))
    model_colors = {model: colors[i] for i, model in enumerate(set(best_models))}
    bar_colors = [model_colors[m] for m in best_models]
    
    bars = ax.barh(TARGET_PARAMS, best_r2s, color=bar_colors)
    
    ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_title('Best Model Performance by Parameter', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (param, model, r2) in enumerate(zip(TARGET_PARAMS, best_models, best_r2s)):
        ax.text(r2 + 0.01, i, f'{model} ({r2:.3f})', 
               va='center', fontsize=9)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model_colors[m], label=m) 
                      for m in sorted(set(best_models))]
    ax.legend(handles=legend_elements, title='Model Type', 
             loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'best_model_summary.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# ==========================================
# Main Training Pipeline
# ==========================================

def main(csv_path: Path, test_size: float = 0.2, fast_mode: bool = False,
         use_feature_engineering: bool = True, poly_degree: int = 2):
    """Main training pipeline with advanced ML techniques."""
    print("="*70)
    print("ADVANCED FUNGAL PARAMETER PREDICTION - ML MODEL TRAINING")
    print("="*70)
    print(f"Fast mode: {fast_mode}")
    print(f"Feature engineering: {use_feature_engineering} (degree={poly_degree})")
    
    df, X, y = load_and_preprocess_data(csv_path)
    
    if use_feature_engineering:
        X_engineered, feature_names, poly_transformer = engineer_features(
            X, degree=poly_degree, include_interactions=True
        )
    else:
        X_engineered = X.values
        feature_names = X.columns.tolist()
        poly_transformer = None
    
    print(f"\nSplitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=test_size, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_configs = get_model_configs(fast_mode=fast_mode)
    print(f"\nModels to train: {list(model_configs.keys())}")
    
    trained_models = train_all_models(X_train, y_train, X_train_scaled, model_configs, cv=5)
    
    ensembles = create_all_ensembles(
        trained_models, X_train, X_train_scaled, y_train, model_configs
    )
    
    results_df, predictions = evaluate_all_models(
        trained_models, ensembles, X_test, y_test, X_test_scaled, model_configs
    )
    
    print("\nSaving models and results...")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name in trained_models['models'].keys():
        for param in TARGET_PARAMS:
            model_path = OUTPUT_DIR / f'{model_name}_{param}_{timestamp}.pkl'
            joblib.dump(trained_models['models'][model_name][param], model_path)
    
    for param in TARGET_PARAMS:
        ensemble_path = OUTPUT_DIR / f'stacking_{param}_{timestamp}.pkl'
        joblib.dump(ensembles[param], ensemble_path)
    
    scaler_path = OUTPUT_DIR / f'scaler_{timestamp}.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler: {scaler_path}")
    
    if poly_transformer is not None:
        poly_path = OUTPUT_DIR / f'poly_transformer_{timestamp}.pkl'
        joblib.dump(poly_transformer, poly_path)
        print(f"  Saved poly transformer: {poly_path}")
    
    results_path = OUTPUT_DIR / f'evaluation_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  Saved results: {results_path}")
    
    best_params_path = OUTPUT_DIR / f'best_hyperparameters_{timestamp}.json'
    with open(best_params_path, 'w') as f:
        json.dump(trained_models['best_params'], f, indent=2, default=str)
    print(f"  Saved hyperparameters: {best_params_path}")
    
    cv_scores_path = OUTPUT_DIR / f'cv_scores_{timestamp}.json'
    with open(cv_scores_path, 'w') as f:
        json.dump(trained_models['cv_scores'], f, indent=2)
    print(f"  Saved CV scores: {cv_scores_path}")
    
    metadata = {
        'timestamp': timestamp,
        'data_file': str(csv_path),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'original_features': X.columns.tolist(),
        'n_engineered_features': X_train.shape[1],
        'targets': TARGET_PARAMS,
        'test_size': test_size,
        'fast_mode': fast_mode,
        'feature_engineering': use_feature_engineering,
        'poly_degree': poly_degree if use_feature_engineering else None,
        'models_trained': list(model_configs.keys()),
    }
    metadata_path = OUTPUT_DIR / f'training_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    plot_predictions(y_test, predictions, OUTPUT_DIR)
    plot_model_comparison(results_df, OUTPUT_DIR)
    plot_best_model_summary(results_df, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    print("\nBest performing model for each parameter (by R²):")
    for param in TARGET_PARAMS:
        param_results = results_df[results_df['parameter'] == param]
        best_idx = param_results['r2'].idxmax()
        best_model = param_results.loc[best_idx, 'model']
        best_r2 = param_results.loc[best_idx, 'r2']
        best_rmse = param_results.loc[best_idx, 'rmse']
        best_mae = param_results.loc[best_idx, 'mae']
        print(f"  {param:15s}: {best_model:15s} (R²={best_r2:.4f}, RMSE={best_rmse:.4f}, MAE={best_mae:.4f})")
    
    print("\nOverall performance by model:")
    for model_name in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model_name]
        print(f"\n  {model_name.upper()}:")
        print(f"    Mean R²:   {model_results['r2'].mean():.4f} ± {model_results['r2'].std():.4f}")
        print(f"    Mean RMSE: {model_results['rmse'].mean():.4f} ± {model_results['rmse'].std():.4f}")
        print(f"    Mean MAE:  {model_results['mae'].mean():.4f} ± {model_results['mae'].std():.4f}")
    
    print("\n" + "="*70)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Advanced ML training with hyperparameter optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Full training with all optimizations
  python train_parameter_predictor_advanced.py --data results.csv
  
  # Fast mode for quick testing
  python train_parameter_predictor_advanced.py --data results.csv --fast
  
  # Without feature engineering
  python train_parameter_predictor_advanced.py --data results.csv --no-feature-engineering
  
  # Custom polynomial degree
  python train_parameter_predictor_advanced.py --data results.csv --poly-degree 3
        """
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Path to characterization results CSV file')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data for testing (default: 0.2)')
    parser.add_argument('--fast', action='store_true',
                       help='Use smaller hyperparameter grids for faster training')
    parser.add_argument('--no-feature-engineering', action='store_true',
                       help='Disable polynomial feature engineering')
    parser.add_argument('--poly-degree', type=int, default=2,
                       help='Polynomial degree for feature engineering (default: 2)')
    
    args = parser.parse_args()
    
    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        exit(1)
    
    main(csv_path, 
         test_size=args.test_size,
         fast_mode=args.fast,
         use_feature_engineering=not args.no_feature_engineering,
         poly_degree=args.poly_degree)
