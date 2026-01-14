import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# FitzHugh-Nagumo parameters
a = 0.7
b = 0.8
epsilon = 0.08  # Time scale separation (controls refractory period)
I_ext = 0.0     # External current (at rest)

# FHN equations
def fhn(state, t, I):
    v, w = state
    dvdt = v - (v**3)/3 - w + I
    dwdt = epsilon * (v + a - b*w)
    return [dvdt, dwdt]

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Define v range for nullclines
v_range = np.linspace(-2.5, 2.5, 500)

# v-nullcline: w = v - v³/3 + I (cubic)
w_v_nullcline = v_range - (v_range**3)/3 + I_ext

# w-nullcline: w = (v + a) / b (linear)
w_w_nullcline = (v_range + a) / b

# Plot nullclines
ax.plot(v_range, w_v_nullcline, 'b-', linewidth=2.5, label='v-nullcline (dv/dt = 0)')
ax.plot(v_range, w_w_nullcline, 'r-', linewidth=2.5, label='w-nullcline (dw/dt = 0)')

# Find fixed point (approximate intersection)
# Solve: v - v³/3 = (v + a)/b
# For these parameters, it's around v ≈ -1.2
v_fp = -1.199
w_fp = (v_fp + a) / b
ax.plot(v_fp, w_fp, 'ko', markersize=12, label='Resting state (fixed point)', zorder=5)

# Simulate a spike trajectory
# Start near fixed point, apply brief stimulus
t_span = np.linspace(0, 100, 2000)

# Trajectory 1: Stimulated spike (starts with elevated v)
initial_stimulated = [v_fp + 0.5, w_fp]  # Stimulus pushes v to the right
trajectory = odeint(fhn, initial_stimulated, t_span, args=(I_ext,))
ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2, label='Spike trajectory', zorder=4)

# Add arrows to show direction on trajectory
arrow_indices = [50, 150, 300, 600]
for idx in arrow_indices:
    if idx < len(trajectory) - 10:
        dx = trajectory[idx+5, 0] - trajectory[idx, 0]
        dy = trajectory[idx+5, 1] - trajectory[idx, 1]
        ax.annotate('', xy=(trajectory[idx+5, 0], trajectory[idx+5, 1]),
                   xytext=(trajectory[idx, 0], trajectory[idx, 1]),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Mark key phases on the trajectory
ax.annotate('1. Stimulus\n(v increases)', xy=(v_fp + 0.5, w_fp), 
            xytext=(v_fp + 1.0, w_fp - 0.3),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax.annotate('2. Fast activation\n(rapid depolarization)', xy=(1.5, 0.3), 
            xytext=(2.0, 0.0),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax.annotate('3. Refractory period\n(slow recovery)', xy=(1.0, 1.2), 
            xytext=(1.5, 1.5),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax.annotate('4. Return to rest', xy=(-0.5, 0.5), 
            xytext=(-1.5, 1.0),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

# Add vector field (direction arrows)
v_grid = np.linspace(-2.2, 2.2, 15)
w_grid = np.linspace(-0.8, 1.8, 12)
V, W = np.meshgrid(v_grid, w_grid)
dV = V - (V**3)/3 - W + I_ext
dW = epsilon * (V + a - b*W)
# Normalize for visibility
magnitude = np.sqrt(dV**2 + dW**2)
dV_norm = dV / magnitude
dW_norm = dW / magnitude
ax.quiver(V, W, dV_norm, dW_norm, magnitude, cmap='gray', alpha=0.4, scale=25)

# Labels and formatting
ax.set_xlabel('v (Fast activation variable / Membrane potential)', fontsize=12)
ax.set_ylabel('w (Slow recovery variable)', fontsize=12)
ax.set_title('FIG. 3: FitzHugh-Nagumo Phase Plane Dynamics\nGoverning Fungal Node Excitability', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.8, 1.8)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.grid(True, alpha=0.3)

# Add parameter annotation
param_text = f'Parameters:\nε = {epsilon} (controls τ_w)\na = {a}\nb = {b}'
ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/FIG_3_FHN_Phase_Plane.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/FIG_3_FHN_Phase_Plane.pdf', bbox_inches='tight')
print("Phase plane diagram saved successfully")