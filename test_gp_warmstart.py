"""Test script to verify GP optimization with warm-start population works correctly."""

import numpy as np
from pathlib import Path
from rediscover_fungal_parameters import (
    load_ml_model_rmse,
    generate_warm_start_population,
    FUNGAL_PARAMS,
    PARAM_BOUNDS
)

def test_load_rmse():
    """Test loading RMSE values from ML model evaluation results."""
    print("Testing load_ml_model_rmse()...")
    rmse_dict = load_ml_model_rmse()
    
    print(f"\nLoaded RMSE for {len(rmse_dict)} parameters:")
    for param, rmse in rmse_dict.items():
        print(f"  {param}: {rmse:.4f}")
    
    # Verify all parameters are present
    assert all(param in rmse_dict for param in FUNGAL_PARAMS), "Missing parameters in RMSE dict"
    print("✓ All parameters present")
    
    # Verify RMSE values are positive
    assert all(rmse > 0 for rmse in rmse_dict.values()), "RMSE values must be positive"
    print("✓ All RMSE values are positive")
    
    return rmse_dict

def test_generate_population(rmse_dict):
    """Test generating warm-start population."""
    print("\n" + "="*70)
    print("Testing generate_warm_start_population()...")
    
    # Create sample initial parameters (ML predictions)
    initial_params = {
        'tau_v': 90.0,
        'tau_w': 800.0,
        'a': 0.65,
        'b': 0.85,
        'v_scale': 3.0,
        'R_off': 150.0,
        'R_on': 20.0,
        'alpha': 0.005
    }
    
    print(f"\nInitial parameters (ML predictions):")
    for param, value in initial_params.items():
        print(f"  {param}: {value:.4f}")
    
    # Generate population
    population = generate_warm_start_population(
        initial_params=initial_params,
        rmse_dict=rmse_dict,
        population_size=20,
        diversity_factor=2.0
    )
    
    print(f"\nGenerated population with {len(population)} individuals")
    
    # Verify population size
    assert len(population) == 20, f"Expected 20 individuals, got {len(population)}"
    print("✓ Correct population size")
    
    # Verify first individual is the ML prediction
    for param in FUNGAL_PARAMS:
        assert abs(population[0][param] - initial_params[param]) < 1e-6, \
            f"First individual should match ML prediction for {param}"
    print("✓ First individual matches ML prediction")
    
    # Verify all individuals respect bounds
    for i, individual in enumerate(population):
        for param in FUNGAL_PARAMS:
            lower, upper = PARAM_BOUNDS[param]
            value = individual[param]
            assert lower <= value <= upper, \
                f"Individual {i}, param {param}: {value} not in bounds [{lower}, {upper}]"
    print("✓ All individuals respect parameter bounds")
    
    # Verify safety constraints
    for i, individual in enumerate(population):
        assert individual['R_off'] >= 1.5 * individual['R_on'], \
            f"Individual {i}: R_off constraint violated"
        assert individual['b'] >= individual['a'], \
            f"Individual {i}: b >= a constraint violated"
    print("✓ All individuals satisfy safety constraints")
    
    # Analyze population diversity
    print("\nPopulation statistics:")
    for param in FUNGAL_PARAMS:
        values = [ind[param] for ind in population]
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"  {param:12s}: mean={mean:8.4f}, std={std:8.4f}, "
              f"range=[{min_val:8.4f}, {max_val:8.4f}]")
        
        # Verify some diversity (std should be non-zero for most parameters)
        if param not in ['a', 'b']:  # These might have low diversity due to constraints
            assert std > 0, f"No diversity for parameter {param}"
    
    print("✓ Population has diversity")
    
    return population

def main():
    """Run all tests."""
    print("="*70)
    print("Testing GP Optimization with Warm-Start Population")
    print("="*70)
    
    # Test 1: Load RMSE
    rmse_dict = test_load_rmse()
    
    # Test 2: Generate population
    population = test_generate_population(rmse_dict)
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nThe implementation is ready to use.")
    print("You can now run rediscover_fungal_parameters.py with:")
    print("  --optimization gp_minimize")
    print("or:")
    print("  --optimization gaussian_process")

if __name__ == "__main__":
    main()
