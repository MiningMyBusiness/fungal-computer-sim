import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter

# Set up the figure
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25,
              height_ratios=[1, 1, 1.2])

# =============================================================================
# Simulation Parameters
# =============================================================================
V_step = 5.0      # Step voltage amplitude
T_step = 1000     # Step duration (time units)
tau_v = 50        # Fast excitation time constant
tau_w = 200       # Slow recovery time constant (refractory period)
alpha_mem = 0.5   # Memristive strength

# =============================================================================
# Panel A1: Step Response - Stimulus
# =============================================================================
ax1a = fig.add_subplot(gs[0, 0])

# Time axis
t_step = np.linspace(0, 1500, 1500)

# Step stimulus
V_stim_step = np.where((t_step >= 100) & (t_step < 100 + T_step), V_step, 0)

ax1a.plot(t_step, V_stim_step, 'b-', linewidth=2.5)
ax1a.fill_between(t_step, 0, V_stim_step, alpha=0.3, color='blue')

# Annotations
ax1a.annotate('', xy=(100 + T_step, V_step/2), xytext=(100, V_step/2),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax1a.text(100 + T_step/2, V_step/2 + 0.5, f'T_step = {T_step}', fontsize=10, ha='center', color='red')

ax1a.annotate('', xy=(50, V_step), xytext=(50, 0),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax1a.text(70, V_step/2, f'V_step\n= {V_step}V', fontsize=10, ha='left', color='green')

ax1a.set_ylabel('Stimulus Voltage (V)', fontsize=11)
ax1a.set_title('(A1) Step Response Test - Stimulus', fontsize=12, fontweight='bold')
ax1a.set_xlim(0, 1500)
ax1a.set_ylim(-0.5, 7)
ax1a.grid(True, alpha=0.3)
ax1a.set_xticklabels([])

# =============================================================================
# Panel A2: Step Response - Response with Feature Extraction
# =============================================================================
ax1b = fig.add_subplot(gs[0, 1])

# Simulate response (simplified model)
def step_response(t, t_on, t_off, tau_rise, tau_decay, amplitude):
    response = np.zeros_like(t)
    # Rising phase
    mask_on = (t >= t_on) & (t < t_off)
    response[mask_on] = amplitude * (1 - np.exp(-(t[mask_on] - t_on) / tau_rise))
    # Decay phase
    mask_off = t >= t_off
    steady_state = amplitude * (1 - np.exp(-(t_off - t_on) / tau_rise))
    response[mask_off] = steady_state * np.exp(-(t[mask_off] - t_off) / tau_decay)
    return response

response_step = step_response(t_step, 100, 100 + T_step, tau_v, tau_w, 4.0)

# Add some noise for realism
np.random.seed(42)
response_step_noisy = response_step + np.random.normal(0, 0.1, len(response_step))
response_step_smooth = savgol_filter(response_step_noisy, 51, 3)

ax1b.plot(t_step, response_step_smooth, 'r-', linewidth=2, label='Response')
ax1b.plot(t_step, response_step, 'r--', linewidth=1, alpha=0.5, label='Ideal')

# Feature extraction annotations
# Rise time (10% to 90%)
steady_state_amp = 4.0 * (1 - np.exp(-T_step / tau_v))
t_10 = 100 + tau_v * np.log(1 / 0.9)  # Time to reach 10%
t_90 = 100 + tau_v * np.log(1 / 0.1)  # Time to reach 90%

ax1b.axhline(y=0.1 * steady_state_amp, color='purple', linestyle=':', alpha=0.7)
ax1b.axhline(y=0.9 * steady_state_amp, color='purple', linestyle=':', alpha=0.7)
ax1b.axhline(y=steady_state_amp, color='orange', linestyle='--', alpha=0.7)

# Rise time annotation
ax1b.annotate('', xy=(t_90, 2), xytext=(t_10, 2),
             arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax1b.text((t_10 + t_90)/2, 2.3, f't_rise\n(→ τ_v)', fontsize=9, ha='center', color='purple',
         bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

# Steady state annotation
ax1b.annotate('Steady-state\namplitude\n(→ conductance)', 
             xy=(600, steady_state_amp), xytext=(750, steady_state_amp + 1),
             fontsize=9, ha='left', color='orange',
             arrowprops=dict(arrowstyle='->', color='orange'),
             bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.8))

# Decay time constant
t_decay_point = 100 + T_step + tau_w
decay_value = steady_state_amp * np.exp(-1)  # Value at t = tau_w after stimulus off
ax1b.plot(t_decay_point, decay_value, 'go', markersize=10)
ax1b.annotate('τ_decay\n(→ τ_w)', 
             xy=(t_decay_point, decay_value), xytext=(t_decay_point + 100, decay_value + 0.8),
             fontsize=9, ha='left', color='green',
             arrowprops=dict(arrowstyle='->', color='green'),
             bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))

ax1b.set_xlabel('Time (arbitrary units)', fontsize=11)
ax1b.set_ylabel('Response Amplitude', fontsize=11)
ax1b.set_title('(A2) Step Response - Recorded Response & Feature Extraction', fontsize=12, fontweight='bold')
ax1b.set_xlim(0, 1500)
ax1b.set_ylim(-0.5, 6)
ax1b.grid(True, alpha=0.3)
ax1b.legend(loc='upper right', fontsize=9)

# Feature summary box
feature_box = "Features Extracted:\n• Rise time t_rise → τ_v\n• Steady-state amplitude → g\n• Decay time constant → τ_w"
ax1b.text(0.02, 0.98, feature_box, transform=ax1b.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# =============================================================================
# Panel B1: Paired-Pulse Test - Stimulus
# =============================================================================
ax2a = fig.add_subplot(gs[1, 0])

# Time axis for paired pulse
t_pp = np.linspace(0, 800, 800)

# Multiple paired-pulse stimuli with different intervals
intervals = [50, 100, 150, 200, 300]
colors_pp = plt.cm.viridis(np.linspace(0.2, 0.8, len(intervals)))

pulse_width = 30
pulse_amp = 5.0
t_first = 50

for i, (delta_t, color) in enumerate(zip(intervals, colors_pp)):
    offset = i * 0.6  # Vertical offset for visibility
    
    # First pulse
    V_pp = np.zeros_like(t_pp)
    V_pp[(t_pp >= t_first) & (t_pp < t_first + pulse_width)] = pulse_amp
    # Second pulse
    t_second = t_first + pulse_width + delta_t
    V_pp[(t_pp >= t_second) & (t_pp < t_second + pulse_width)] = pulse_amp
    
    ax2a.plot(t_pp, V_pp + offset, color=color, linewidth=2, 
             label=f'Δt = {delta_t}' if i < 3 else None)
    ax2a.fill_between(t_pp, offset, V_pp + offset, alpha=0.2, color=color)

# Annotation for inter-pulse interval
ax2a.annotate('', xy=(t_first + pulse_width + intervals[2], 3.8), 
             xytext=(t_first + pulse_width, 3.8),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax2a.text(t_first + pulse_width + intervals[2]/2, 4.2, 'Δt_pp\n(variable)', 
         fontsize=10, ha='center', color='red')

ax2a.set_ylabel('Stimulus Voltage (V)', fontsize=11)
ax2a.set_title('(B1) Paired-Pulse Test - Stimulus (Multiple Intervals)', fontsize=12, fontweight='bold')
ax2a.set_xlim(0, 600)
ax2a.set_ylim(-0.5, 9)
ax2a.legend(loc='upper right', fontsize=9)
ax2a.grid(True, alpha=0.3)
ax2a.set_xticklabels([])

# =============================================================================
# Panel B2: Paired-Pulse Test - Response and Recovery Curve
# =============================================================================
ax2b = fig.add_subplot(gs[1, 1])

# Create two subplots within this panel
# Left: Example responses
# Right: Recovery curve

# Simulate paired-pulse responses
def paired_pulse_response(t, t1_on, t1_off, t2_on, t2_off, tau_rise, tau_refractory, amp):
    """Simulate response with refractory period effect"""
    r1 = step_response(t, t1_on, t1_off, tau_rise, tau_rise*2, amp)
    
    # Second pulse response is attenuated based on recovery
    delta_t = t2_on - t1_off
    recovery_fraction = 1 - np.exp(-delta_t / tau_refractory)
    r2 = step_response(t, t2_on, t2_off, tau_rise, tau_rise*2, amp * recovery_fraction)
    
    return r1 + r2, recovery_fraction

# Plot example response for one interval
t_example = np.linspace(0, 400, 400)
response_example, _ = paired_pulse_response(t_example, 50, 80, 180, 210, 20, tau_w, 3.0)
response_example += np.random.normal(0, 0.08, len(response_example))

ax2b.plot(t_example, response_example, 'r-', linewidth=2)

# Mark R1 and R2
R1 = 3.0
delta_t_ex = 100
recovery = 1 - np.exp(-delta_t_ex / tau_w)
R2 = R1 * recovery

ax2b.annotate('R₁', xy=(65, R1), xytext=(30, R1 + 0.8),
             fontsize=11, fontweight='bold', color='blue',
             arrowprops=dict(arrowstyle='->', color='blue'))
ax2b.annotate('R₂', xy=(195, R2), xytext=(230, R2 + 0.5),
             fontsize=11, fontweight='bold', color='blue',
             arrowprops=dict(arrowstyle='->', color='blue'))

# Recovery ratio annotation
ax2b.text(0.5, 0.95, f'Recovery Ratio = R₂/R₁ = {recovery:.2f}', 
         transform=ax2b.transAxes, fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

# Inset: Recovery curve
ax2b_inset = ax2b.inset_axes([0.55, 0.35, 0.42, 0.55])

delta_t_range = np.linspace(0, 600, 100)
recovery_curve = 1 - np.exp(-delta_t_range / tau_w)

ax2b_inset.plot(delta_t_range, recovery_curve, 'b-', linewidth=2.5)
ax2b_inset.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5)
ax2b_inset.axvline(x=tau_w * np.log(2), color='red', linestyle='--', linewidth=1.5)

# Mark tau_w,50
tau_w_50 = tau_w * np.log(2)
ax2b_inset.plot(tau_w_50, 0.5, 'ro', markersize=10)
ax2b_inset.annotate(f'τ_w,50 = {tau_w_50:.0f}\n(50% recovery)', 
                   xy=(tau_w_50, 0.5), xytext=(tau_w_50 + 80, 0.35),
                   fontsize=9, color='red',
                   arrowprops=dict(arrowstyle='->', color='red'))

ax2b_inset.set_xlabel('Δt_pp', fontsize=9)
ax2b_inset.set_ylabel('R₂/R₁', fontsize=9)
ax2b_inset.set_title('Recovery Curve', fontsize=10, fontweight='bold')
ax2b_inset.set_xlim(0, 600)
ax2b_inset.set_ylim(0, 1.1)
ax2b_inset.grid(True, alpha=0.3)

# Formula
ax2b_inset.text(0.95, 0.15, r'$\frac{R_2}{R_1} = 1 - e^{-\Delta t / \tau_w}$', 
               transform=ax2b_inset.transAxes, fontsize=10, ha='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax2b.set_xlabel('Time (arbitrary units)', fontsize=11)
ax2b.set_ylabel('Response Amplitude', fontsize=11)
ax2b.set_title('(B2) Paired-Pulse Response & Recovery Curve → τ_w', fontsize=12, fontweight='bold')
ax2b.set_xlim(0, 400)
ax2b.set_ylim(-0.5, 5)
ax2b.grid(True, alpha=0.3)

# =============================================================================
# Panel C1: Triangle Sweep (Cyclic Voltammetry) - Stimulus
# =============================================================================
ax3a = fig.add_subplot(gs[2, 0])

# Time axis
t_cv = np.linspace(0, 400, 1000)

# Triangle wave parameters
V_cv_min = -2.0
V_cv_max = 5.0
period = 100
V_amplitude = (V_cv_max - V_cv_min) / 2
V_offset = (V_cv_max + V_cv_min) / 2

# Generate triangle wave
from scipy.signal import sawtooth
V_triangle = V_offset + V_amplitude * sawtooth(2 * np.pi * t_cv / period, width=0.5)

ax3a.plot(t_cv, V_triangle, 'b-', linewidth=2.5)
ax3a.fill_between(t_cv, 0, V_triangle, where=V_triangle > 0, alpha=0.2, color='blue')
ax3a.fill_between(t_cv, 0, V_triangle, where=V_triangle < 0, alpha=0.2, color='red')

# Annotations
ax3a.axhline(y=V_cv_max, color='green', linestyle='--', alpha=0.7)
ax3a.axhline(y=V_cv_min, color='green', linestyle='--', alpha=0.7)
ax3a.text(350, V_cv_max + 0.3, f'V_max = {V_cv_max}V', fontsize=10, color='green')
ax3a.text(350, V_cv_min - 0.5, f'V_min = {V_cv_min}V', fontsize=10, color='green')

# Scan rate annotation
ax3a.annotate('', xy=(100, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax3a.text(50, -0.8, 'Period T', fontsize=10, ha='center', color='purple')

ax3a.set_ylabel('Stimulus Voltage (V)', fontsize=11)
ax3a.set_title('(C1) Triangle Sweep (Cyclic Voltammetry) - Stimulus', fontsize=12, fontweight='bold')
ax3a.set_xlim(0, 400)
ax3a.set_ylim(-3.5, 7)
ax3a.grid(True, alpha=0.3)
ax3a.axhline(y=0, color='gray', linewidth=0.5)

# =============================================================================
# Panel C2: Cyclic Voltammetry - I-V Hysteresis
# =============================================================================
ax3b = fig.add_subplot(gs[2, 1])

# Simulate I-V response with hysteresis (memristive behavior)
# Generate current response with hysteresis
def memristive_IV_response(V, alpha_mem, g_base=0.5):
    """Simulate memristive I-V with hysteresis"""
    I = np.zeros_like(V)
    g = g_base  # Initial conductance
    
    for i in range(len(V)):
        # Update conductance based on voltage history (simplified)
        if i > 0:
            dV = V[i] - V[i-1]
            if dV > 0:  # Forward sweep
                g = g_base + alpha_mem * (1 - np.exp(-abs(V[i])/2))
            else:  # Reverse sweep
                g = g_base + alpha_mem * 0.5 * (1 - np.exp(-abs(V[i])/2))
        I[i] = g * V[i]
    
    return I

# Use the triangle wave voltage
I_response = memristive_IV_response(V_triangle, alpha_mem=0.3, g_base=0.4)

# Add noise
I_response_noisy = I_response + np.random.normal(0, 0.05, len(I_response))

# Color by time to show direction
n_points = len(V_triangle)
colors = plt.cm.viridis(np.linspace(0, 1, n_points))

# Plot as colored line segments
for i in range(n_points - 1):
    ax3b.plot(V_triangle[i:i+2], I_response_noisy[i:i+2], color=colors[i], linewidth=2)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=t_cv[-1]))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3b, label='Time', shrink=0.7, pad=0.02)

# Highlight hysteresis loop
ax3b.annotate('Forward\nsweep', xy=(3, 2.5), xytext=(4, 3.5),
             fontsize=10, color='darkblue',
             arrowprops=dict(arrowstyle='->', color='darkblue'))
ax3b.annotate('Reverse\nsweep', xy=(3, 1.5), xytext=(4, 0.5),
             fontsize=10, color='darkorange',
             arrowprops=dict(arrowstyle='->', color='darkorange'))

# Shade hysteresis area
# Find points for one complete cycle
cycle_start = 0
cycle_end = 250  # One full cycle
V_cycle = V_triangle[cycle_start:cycle_end]
I_cycle = I_response_noisy[cycle_start:cycle_end]

# Calculate approximate hysteresis area
# Forward sweep (first half)
mid = len(V_cycle) // 2
V_forward = V_cycle[:mid]
I_forward = I_cycle[:mid]
V_reverse = V_cycle[mid:]
I_reverse = I_cycle[mid:]

# Shade the area between curves (simplified visualization)
ax3b.fill_between([2, 4], [1.2, 2.8], [1.8, 3.2], alpha=0.3, color='purple',
                  label='Hysteresis area → α')

# Hysteresis area annotation
ax3b.annotate('HYSTERESIS\nLOOP AREA\n→ memristive\nstrength (α)', 
             xy=(3, 2), xytext=(-0.5, 1.5),
             fontsize=10, ha='center', color='purple', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='purple', lw=2),
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9))

ax3b.set_xlabel('Voltage (V)', fontsize=11)
ax3b.set_ylabel('Current (I)', fontsize=11)
ax3b.set_title('(C2) I-V Characteristic with Hysteresis → Memristive Strength α', 
              fontsize=12, fontweight='bold')
ax3b.axhline(y=0, color='gray', linewidth=0.5)
ax3b.axvline(x=0, color='gray', linewidth=0.5)
ax3b.grid(True, alpha=0.3)

# Feature summary box
feature_box_cv = "Feature Extracted:\n• Hysteresis loop area → α\n  (memristive plasticity strength)"
ax3b.text(0.02, 0.98, feature_box_cv, transform=ax3b.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# =============================================================================
# Add overall summary box at bottom
# =============================================================================
summary_text = """
STANDARDIZED STIMULUS SUITE - SUMMARY

┌─────────────────┬────────────────────────┬─────────────────────────┬──────────────────────────┐
│     TEST        │    STIMULUS            │   FEATURES EXTRACTED    │   PARAMETER INFERRED     │
├─────────────────┼────────────────────────┼─────────────────────────┼──────────────────────────┤
│ (A) Step        │ Square pulse           │ Rise time, steady-state │ τ_v, conductance g       │
│     Response    │ V=5V, T=1000           │ amplitude, decay time   │ τ_w (decay constant)     │
├─────────────────┼────────────────────────┼─────────────────────────┼──────────────────────────┤
│ (B) Paired-     │ Two pulses at          │ Recovery ratio R₂/R₁    │ τ_w (refractory period)  │
│     Pulse       │ variable Δt_pp         │ at multiple intervals   │ τ_w,50 (50% recovery)    │
├─────────────────┼────────────────────────┼─────────────────────────┼──────────────────────────┤
│ (C) Triangle    │ Cyclic voltage         │ I-V hysteresis loop     │ α (memristive strength)  │
│     Sweep       │ V_min to V_max         │ area                    │ g_min, g_max             │
└─────────────────┴────────────────────────┴─────────────────────────┴──────────────────────────┘

This standardized suite enables systematic characterization absent from prior art (Adamatzky et al.).
"""

fig.text(0.5, -0.15, summary_text, ha='center', va='bottom', fontsize=9,
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                       edgecolor='orange', alpha=0.95))

# Main title
fig.suptitle('FIG. 8: Standardized Stimulus Suite for Calibration Protocol\nThree Tests Probing Distinct Biophysical Parameters', 
            fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.3, 1, 0.97])
plt.savefig('FIG_8_Standardized_Stimulus_Suite.png', dpi=300, bbox_inches='tight')
plt.savefig('FIG_8_Standardized_Stimulus_Suite.pdf', bbox_inches='tight')
print("Standardized stimulus suite diagram saved successfully")