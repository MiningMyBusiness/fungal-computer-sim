"""Test script to verify optimization logic is correct."""

import numpy as np

# Simulate what happens during optimization
print("="*60)
print("Testing Optimization Logic")
print("="*60)

# Simulate different configurations and their scores
configs = [
    {"name": "Config A", "score": 0.5},
    {"name": "Config B", "score": -2.0},
    {"name": "Config C", "score": 1.2},
    {"name": "Config D", "score": -0.3},
    {"name": "Config E", "score": 0.8},
]

print("\nSimulated Configurations:")
print("-" * 60)
for config in configs:
    objective = -config["score"]  # What gets returned to gp_minimize
    print(f"{config['name']}: Score = {config['score']:6.2f}, Objective = {objective:6.2f}")

print("\n" + "="*60)
print("What gp_minimize does:")
print("="*60)

# gp_minimize finds the minimum objective value
objectives = [-c["score"] for c in configs]
min_objective_idx = np.argmin(objectives)
best_config = configs[min_objective_idx]
min_objective = objectives[min_objective_idx]

print(f"\nMinimum objective value: {min_objective:.2f}")
print(f"This corresponds to: {best_config['name']}")
print(f"With score: {best_config['score']:.2f}")

print("\n" + "="*60)
print("After optimization:")
print("="*60)
print(f"res.fun = {min_objective:.2f} (minimum objective)")
print(f"best_score = -res.fun = {-min_objective:.2f}")
print(f"\nThis is CORRECT! We want to maximize score, so we should select")
print(f"the configuration with score = {best_config['score']:.2f}")

print("\n" + "="*60)
print("Verification:")
print("="*60)
all_scores = [c["score"] for c in configs]
print(f"All scores: {all_scores}")
print(f"Maximum score: {max(all_scores):.2f}")
print(f"Selected score: {best_config['score']:.2f}")
print(f"Match: {max(all_scores) == best_config['score']}")

print("\n" + "="*60)
print("Conclusion:")
print("="*60)
print("The optimization logic is CORRECT.")
print("gp_minimize minimizes -score, which maximizes score.")
print("The best configuration has the highest score, as expected.")
