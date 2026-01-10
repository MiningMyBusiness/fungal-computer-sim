import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

# Set up the figure with multiple subplots
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# =============================================================================
# Panel A: Hysteresis Loop (I-V characteristic)
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Simulate a memristor I-V curve with hysteresis
# Apply a sinusoidal voltage and track the current

# Parameters
g_min = 0.1   # Minimum conductance (High Resistance State)
g_max = 1.0   # Maximum conductance (Low Resistance State)
alpha = 2.0   # Potentiation rate
beta = 0.5    # Decay rate
v_threshold = 0.3  # Threshold for plasticity activation

# Threshold function f(v)
def f_threshold(v, v_th=v_threshold):
    """Soft threshold function"""
    return np.where(np.abs(v) > v_th, (np.abs(v) - v_th)**2, 0)

# Simulate with sinusoidal voltage
t = np.linspace(0, 4*np.pi, 1000)
V = 1.5 * np.sin(t)  # Applied voltage

# Integrate the memristor state
dt = t[1] - t[0]
x = np.zeros_like(t)
x[0] = 0.2  # Initial state

for i in range(1, len(t)):
    dxdt = alpha * f_threshold(V[i-1]) * (1 - x[i-1]) - beta * x[i-1]
    x[i] = x[i-1] + dxdt * dt
    x[i] = np.clip(x[i], 0, 1)  # Keep in [0, 1]

# Conductance and current
g = g_min + (g_max - g_min) * x
I = g * V

# Plot the pinched hysteresis loop
# Color by time to show direction
colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
for i in range(len(t)-1):
    ax1.plot(V[i:i+2], I[i:i+2], color=colors[i], linewidth=2)

# Add arrows to show direction
arrow_indices = [100, 350, 600, 850]
for idx in arrow_indices:
    ax1.annotate('', xy=(V[idx+20], I[idx+20]), xytext=(V[idx], I[idx]),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Mark the origin (pinched point)
ax1.plot(0, 0, 'ro', markersize=10, zorder=5, label='Pinched at origin')

# Add region labels
ax1.annotate('High Resistance\nState (HRS)\ng ≈ g_min', 
            xy=(-1.0, -0.3), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax1.annotate('Low Resistance\nState (LRS)\ng ≈ g_max', 
            xy=(1.0, 1.0), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

ax1.set_xlabel('Voltage Difference (v_i - v_j)', fontsize=11)
ax1.set_ylabel('Current (I = g · V)', fontsize=11)
ax1.set_title('(A) Memristive I-V Hysteresis Loop\n(Pinched at Origin)', fontsize=12, fontweight='bold')
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax1.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Add colorbar for time
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=t[-1]))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, label='Time', shrink=0.8)

# =============================================================================
# Panel B: Conductance State Evolution
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Plot voltage and conductance state over time
ax2_twin = ax2.twinx()

ln1 = ax2.plot(t, V, 'b-', linewidth=2, label='Applied Voltage V(t)')
ln2 = ax2_twin.plot(t, x, 'r-', linewidth=2, label='State Variable x(t)')
ln3 = ax2_twin.plot(t, g, 'g--', linewidth=2, label='Conductance g(t)')

ax2.set_xlabel('Time', fontsize=11)
ax2.set_ylabel('Voltage (v_i - v_j)', fontsize=11, color='blue')
ax2_twin.set_ylabel('State x / Conductance g', fontsize=11, color='red')
ax2.set_title('(B) Temporal Evolution of Memristor State', fontsize=12, fontweight='bold')

# Combine legends
lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc='upper right', fontsize=9)

ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax2.grid(True, alpha=0.3)

# Shade regions where |V| > threshold
for i in range(len(t)-1):
    if abs(V[i]) > v_threshold:
        ax2.axvspan(t[i], t[i+1], alpha=0.1, color='yellow')

ax2.text(2, 1.7, 'Yellow: |V| > V_threshold\n(plasticity active)', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# =============================================================================
# Panel C: Conductance vs State Variable
# =============================================================================
ax3 = fig.add_subplot(gs[1, 0])

x_range = np.linspace(0, 1, 100)
g_range = g_min + (g_max - g_min) * x_range

ax3.plot(x_range, g_range, 'k-', linewidth=3)
ax3.fill_between(x_range, g_min, g_range, alpha=0.3, color='blue')

# Mark key points
ax3.plot(0, g_min, 'bo', markersize=12, label=f'x=0: g_min = {g_min}')
ax3.plot(1, g_max, 'ro', markersize=12, label=f'x=1: g_max = {g_max}')
ax3.plot(0.5, g_min + (g_max-g_min)*0.5, 'go', markersize=10, label='x=0.5: intermediate')

# Add equation
ax3.text(0.5, 0.3, r'$g_{ij}(t) = g_{min} + (g_{max} - g_{min}) \cdot x_{ij}(t)$', 
        fontsize=12, ha='center', transform=ax3.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax3.set_xlabel('Internal State Variable x_ij ∈ [0,1]', fontsize=11)
ax3.set_ylabel('Edge Conductance g_ij', fontsize=11)
ax3.set_title('(C) Linear Conductance-State Relationship', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.05, 1.05)
ax3.set_ylim(0, g_max + 0.1)

# =============================================================================
# Panel D: State Dynamics Diagram
# =============================================================================
ax4 = fig.add_subplot(gs[1, 1])

# Show the dynamics equation graphically
# dx/dt = α·f(V)·(1-x) - β·x

x_vals = np.linspace(0, 1, 100)

# For different voltage levels
voltage_levels = [0, 0.5, 1.0, 1.5]
colors_v = ['gray', 'blue', 'orange', 'red']

for V_applied, color in zip(voltage_levels, colors_v):
    f_v = f_threshold(V_applied)
    dxdt = alpha * f_v * (1 - x_vals) - beta * x_vals
    label = f'|V| = {V_applied}'
    if V_applied == 0:
        label += ' (decay only)'
    ax4.plot(x_vals, dxdt, color=color, linewidth=2, label=label)

ax4.axhline(y=0, color='black', linewidth=1, linestyle='-')
ax4.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')

# Mark equilibrium points
ax4.plot(0, 0, 'ko', markersize=8)
ax4.annotate('x=0 stable\n(no activity)', xy=(0, 0), xytext=(0.15, -0.3),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

# Add equation (simplified without cases environment)
eq_text = ('dx/dt = α·f(V)·(1-x) - β·x\n\n'
          'f(V) = (|V| - V_th)²  if |V| > V_th\n'
          'f(V) = 0              if |V| ≤ V_th')
ax4.text(0.52, 0.75, eq_text, fontsize=10, ha='left', transform=ax4.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        family='monospace')

# Add parameter values
param_text = f'α = {alpha} (potentiation)\nβ = {beta} (decay)\nV_th = {v_threshold}'
ax4.text(0.02, 0.98, param_text, transform=ax4.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4.set_xlabel('State Variable x_ij', fontsize=11)
ax4.set_ylabel('Rate of Change dx_ij/dt', fontsize=11)
ax4.set_title('(D) State Dynamics at Different Voltage Levels', fontsize=12, fontweight='bold')
ax4.legend(loc='lower left', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.05, 1.05)

# Add annotations for potentiation vs decay
ax4.annotate('Potentiation\n(x increases)', xy=(0.3, 0.8), fontsize=10, ha='center', color='red')
ax4.annotate('Decay\n(x decreases)', xy=(0.7, -0.4), fontsize=10, ha='center', color='gray')

# Main title
fig.suptitle('FIG. 4: Memristive Coupling Model for Edge Conductances', 
            fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/FIG_4_Memristive_Coupling.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/FIG_4_Memristive_Coupling.pdf', bbox_inches='tight')
print("Memristive coupling diagram saved successfully")