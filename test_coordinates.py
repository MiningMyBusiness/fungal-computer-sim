"""Quick test to verify electrode coordinates are within network bounds."""

import numpy as np
import matplotlib.pyplot as plt
from realistic_sim import RealisticFungalComputer

# Create a small network
print("Creating test network...")
env = RealisticFungalComputer(num_nodes=30, area_size=20.0, random_seed=42)

# Print coordinate statistics
x_coords = env.node_coords[:, 0]
y_coords = env.node_coords[:, 1]

print(f"\nNetwork Statistics:")
print(f"  Nodes: {env.num_nodes}")
print(f"  Edges: {len(env.edge_list)}")
print(f"  X range: [{np.min(x_coords):.2f}, {np.max(x_coords):.2f}]")
print(f"  Y range: [{np.min(y_coords):.2f}, {np.max(y_coords):.2f}]")
print(f"  Area size: {env.area_size}")

# Test electrode positions (should be within network bounds)
test_electrodes = [
    (10.0, 10.0),  # Center
    (5.0, 5.0),    # Lower left quadrant
    (15.0, 15.0),  # Upper right quadrant
]

print(f"\nTesting electrode positions:")
for i, pos in enumerate(test_electrodes):
    min_dist = np.min(np.sqrt(np.sum((env.node_coords - pos)**2, axis=1)))
    print(f"  Electrode {i+1} at {pos}: nearest node = {min_dist:.2f}mm")

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Plot network nodes
ax.scatter(x_coords, y_coords, c='lightgray', s=100, alpha=0.7, 
          edgecolors='black', linewidths=1, label='Network nodes')

# Plot edges
for u, v in env.edge_list:
    x_vals = [env.node_coords[u, 0], env.node_coords[v, 0]]
    y_vals = [env.node_coords[u, 1], env.node_coords[v, 1]]
    ax.plot(x_vals, y_vals, 'gray', alpha=0.3, linewidth=0.5)

# Plot test electrodes
for i, pos in enumerate(test_electrodes):
    ax.scatter(pos[0], pos[1], c=['red', 'blue', 'green'][i], s=200, 
              marker='*', edgecolors='black', linewidths=2, 
              label=f'Test electrode {i+1}')

ax.set_xlim(-1, env.area_size + 1)
ax.set_ylim(-1, env.area_size + 1)
ax.set_xlabel('X Position (mm)', fontsize=12)
ax.set_ylabel('Y Position (mm)', fontsize=12)
ax.set_title('Network Topology with Test Electrodes', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('coordinate_test.png', dpi=150)
print(f"\nVisualization saved to 'coordinate_test.png'")
plt.show()
