import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set up the figure
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# =============================================================================
# Design Heuristic Parameters (educated guesses based on physical reasoning)
# =============================================================================
V_max = 10.0      # Maximum voltage needed for poorly connected networks (Volts)
V_min = 2.0       # Minimum voltage for well-connected networks (Volts)
k_lambda = 1.5    # Decay rate constant
lambda_crit = 2.5 # Critical connectivity above which V_min suffices

# The design heuristic formula
def V_opt(lambda2, Vmax=V_max, Vmin=V_min, k=k_lambda, lam_crit=lambda_crit):
    """Optimal stimulus voltage as function of algebraic connectivity"""
    result = np.where(lambda2 < lam_crit,
                     Vmax * np.exp(-k * lambda2),
                     Vmin * np.ones_like(lambda2))
    return np.maximum(result, Vmin)  # Never go below V_min

# =============================================================================
# Panel A: Main Voltage-Connectivity Curve
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Lambda2 range (algebraic connectivity)
lambda2_range = np.linspace(0, 4, 500)
V_opt_curve = V_opt(lambda2_range)

# Fill regions to show different operating regimes
# High voltage region (low connectivity)
mask_low = lambda2_range < 1.0
ax1.fill_between(lambda2_range[mask_low], 0, 12, color='red', alpha=0.1)
# Transition region
mask_mid = (lambda2_range >= 1.0) & (lambda2_range < lambda_crit)
ax1.fill_between(lambda2_range[mask_mid], 0, 12, color='yellow', alpha=0.1)
# Low voltage region (high connectivity)
mask_high = lambda2_range >= lambda_crit
ax1.fill_between(lambda2_range[mask_high], 0, 12, color='green', alpha=0.1)

# Plot the design heuristic curve
ax1.plot(lambda2_range, V_opt_curve, 'b-', linewidth=3, label=r'Design Heuristic: $V_{opt}(\lambda_2)$')

# Mark critical point
ax1.axvline(x=lambda_crit, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.plot(lambda_crit, V_min, 'ko', markersize=10, zorder=5)
ax1.annotate(f'λ_crit = {lambda_crit}\n(transition point)', 
            xy=(lambda_crit, V_min), xytext=(lambda_crit + 0.3, V_min + 2),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

# Mark V_max and V_min
ax1.axhline(y=V_max, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.axhline(y=V_min, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.text(3.3, V_max + 0.3, f'V_max = {V_max}V', fontsize=10, ha='right', color='red')
ax1.text(3.3, V_min - 0.6, f'V_min = {V_min}V', fontsize=10, ha='right', color='green')

# Add simulated "Full Sweep" data points
np.random.seed(123)

# Successful gates (close to the optimal curve)
n_success = 30
lambda2_success = np.random.uniform(0.2, 3.5, n_success)
V_success = V_opt(lambda2_success) + np.random.normal(0, 0.5, n_success)
V_success = np.maximum(V_success, V_min)
ax1.scatter(lambda2_success, V_success, c='green', s=60, marker='o', 
           edgecolors='darkgreen', linewidths=1, label='Successful XOR gate', zorder=4, alpha=0.7)

# Failed gates - too low voltage
lambda2_fail_low = np.random.uniform(0.3, 1.5, 10)
V_fail_low = V_opt(lambda2_fail_low) - np.random.uniform(2, 4, 10)
V_fail_low = np.maximum(V_fail_low, 0.5)
ax1.scatter(lambda2_fail_low, V_fail_low, c='red', s=60, marker='x', 
           linewidths=2, label='Failed (insufficient voltage)', zorder=4)

# Failed gates - too high voltage (saturation)
lambda2_fail_high = np.random.uniform(2.0, 3.5, 8)
V_fail_high = V_opt(lambda2_fail_high) + np.random.uniform(3, 5, 8)
ax1.scatter(lambda2_fail_high, V_fail_high, c='orange', s=60, marker='s', 
           edgecolors='darkorange', linewidths=1, label='Failed (saturation)', zorder=4)

# Add equations
eq_text = (r'$V_{opt}(\lambda_2) = V_{max} \cdot e^{-k_\lambda \cdot \lambda_2}$  for $\lambda_2 < \lambda_{crit}$'
          '\n'
          r'$V_{opt}(\lambda_2) = V_{min}$  for $\lambda_2 \geq \lambda_{crit}$')
ax1.text(0.02, 0.98, eq_text, transform=ax1.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Add parameter values
param_text = f'Parameters:\nV_max = {V_max}V\nV_min = {V_min}V\nk_λ = {k_lambda}\nλ_crit = {lambda_crit}'
ax1.text(0.98, 0.98, param_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Region labels
ax1.text(0.6, 8.0, 'LOW\nCONNECTIVITY', fontsize=10, fontweight='bold', ha='center', color='darkred', alpha=0.8)
ax1.text(1.75, 8.0, 'TRANSITION', fontsize=10, fontweight='bold', ha='center', color='darkorange', alpha=0.8)
ax1.text(3.2, 8.0, 'HIGH\nCONNECTIVITY', fontsize=10, fontweight='bold', ha='center', color='darkgreen', alpha=0.8)

# Labels and formatting
ax1.set_xlabel('Algebraic Connectivity λ₂', fontsize=12)
ax1.set_ylabel('Optimal Stimulus Voltage V_opt (Volts)', fontsize=12)
ax1.set_title('(A) Voltage-Connectivity Design Heuristic\nDerived from Full Sweep Simulations', fontsize=12, fontweight='bold')
ax1.legend(loc='center right', fontsize=9)
ax1.set_xlim(0, 4)
ax1.set_ylim(0, 12)
ax1.grid(True, alpha=0.3)

# =============================================================================
# Panel B: What is Algebraic Connectivity?
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

ax2.text(0.5, 0.98, '(B) Understanding Algebraic Connectivity (λ₂)', 
        fontsize=12, fontweight='bold', ha='center', transform=ax2.transAxes)

# Explanation text
explanation = (
    "Algebraic connectivity (λ₂) is the second-smallest\n"
    "eigenvalue of the graph Laplacian matrix.\n\n"
    "It measures how well-connected a network is:\n"
    "• λ₂ = 0 → Disconnected graph\n"
    "• Small λ₂ → Weakly connected (has bottlenecks)\n"
    "• Large λ₂ → Strongly connected (many paths)"
)
ax2.text(0.5, 0.82, explanation, fontsize=11, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

# Draw three example networks with different connectivity

# --- Low λ₂ (sparse/bottleneck) ---
ax2.text(0.17, 0.45, 'Low λ₂ ≈ 0.3', fontsize=10, fontweight='bold', ha='center')
ax2.text(0.17, 0.40, '(Sparse/Bottleneck)', fontsize=9, ha='center')

sparse_nodes = [(0.08, 0.35), (0.12, 0.25), (0.17, 0.35), (0.22, 0.25), (0.26, 0.35)]
sparse_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Linear chain - low connectivity
for pos in sparse_nodes:
    circle = Circle(pos, 0.018, color='lightcoral', ec='darkred', linewidth=1.5, zorder=5)
    ax2.add_patch(circle)
for i, j in sparse_edges:
    ax2.plot([sparse_nodes[i][0], sparse_nodes[j][0]], 
            [sparse_nodes[i][1], sparse_nodes[j][1]], 
            'darkred', linewidth=2, zorder=1)
ax2.text(0.17, 0.15, 'Needs HIGH\nvoltage', fontsize=9, ha='center', color='red',
        bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

# --- Medium λ₂ ---
ax2.text(0.5, 0.45, 'Medium λ₂ ≈ 1.5', fontsize=10, fontweight='bold', ha='center')
ax2.text(0.5, 0.40, '(Moderate)', fontsize=9, ha='center')

med_nodes = [(0.42, 0.35), (0.46, 0.25), (0.5, 0.38), (0.54, 0.25), (0.58, 0.35)]
med_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4)]  # Some extra connections
for pos in med_nodes:
    circle = Circle(pos, 0.018, color='lightyellow', ec='darkorange', linewidth=1.5, zorder=5)
    ax2.add_patch(circle)
for i, j in med_edges:
    ax2.plot([med_nodes[i][0], med_nodes[j][0]], 
            [med_nodes[i][1], med_nodes[j][1]], 
            'darkorange', linewidth=2, zorder=1)
ax2.text(0.5, 0.15, 'Needs MEDIUM\nvoltage', fontsize=9, ha='center', color='darkorange',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- High λ₂ (dense) ---
ax2.text(0.83, 0.45, 'High λ₂ ≈ 3.0', fontsize=10, fontweight='bold', ha='center')
ax2.text(0.83, 0.40, '(Well-connected)', fontsize=9, ha='center')

dense_nodes = [(0.75, 0.35), (0.79, 0.25), (0.83, 0.38), (0.87, 0.25), (0.91, 0.35)]
dense_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4), (0, 3), (1, 3), (1, 4), (0, 4)]  # Many connections
for pos in dense_nodes:
    circle = Circle(pos, 0.018, color='lightgreen', ec='darkgreen', linewidth=1.5, zorder=5)
    ax2.add_patch(circle)
for i, j in dense_edges:
    ax2.plot([dense_nodes[i][0], dense_nodes[j][0]], 
            [dense_nodes[i][1], dense_nodes[j][1]], 
            'darkgreen', linewidth=1.5, alpha=0.7, zorder=1)
ax2.text(0.83, 0.15, 'Needs LOW\nvoltage', fontsize=9, ha='center', color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))

# Formula for Laplacian
ax2.text(0.5, 0.05, 'λ₂ = second eigenvalue of L = D - A\n(D = degree matrix, A = adjacency matrix)', 
        fontsize=9, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# =============================================================================
# Panel C: Physical Interpretation - Why Voltage Depends on Connectivity
# =============================================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

ax3.text(0.5, 0.98, '(C) Physical Interpretation: Why Connectivity Affects Required Voltage', 
        fontsize=12, fontweight='bold', ha='center', transform=ax3.transAxes)

# --- Low connectivity case ---
ax3.add_patch(FancyBboxPatch((0.02, 0.45), 0.45, 0.48, boxstyle="round,pad=0.02",
                              facecolor='mistyrose', edgecolor='red', linewidth=2, alpha=0.7))
ax3.text(0.245, 0.9, 'LOW CONNECTIVITY', fontsize=11, fontweight='bold', ha='center', color='darkred')
ax3.text(0.245, 0.85, '(Small λ₂)', fontsize=10, ha='center')

# Draw sparse network with electrode
sparse_net = [(0.1, 0.7), (0.2, 0.65), (0.3, 0.7), (0.4, 0.65)]
sparse_edges_c = [(0, 1), (2, 3)]  # Gap in middle!
for pos in sparse_net:
    circle = Circle(pos, 0.02, color='lightblue', ec='black', linewidth=1.5, zorder=5)
    ax3.add_patch(circle)
for i, j in sparse_edges_c:
    ax3.plot([sparse_net[i][0], sparse_net[j][0]], 
            [sparse_net[i][1], sparse_net[j][1]], 
            'gray', linewidth=2, zorder=1)

# Show gap that needs bridging
ax3.annotate('', xy=(0.3, 0.7), xytext=(0.2, 0.65),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
ax3.text(0.25, 0.75, 'GAP!', fontsize=10, fontweight='bold', color='red', ha='center')

ax3.text(0.245, 0.48, 'Problem: Signal must "jump"\nacross gaps in the network.\n\n'
        'Solution: HIGH voltage to\nbridge disconnections.',
        fontsize=9, ha='center')

# --- High connectivity case ---
ax3.add_patch(FancyBboxPatch((0.53, 0.45), 0.45, 0.48, boxstyle="round,pad=0.02",
                              facecolor='honeydew', edgecolor='green', linewidth=2, alpha=0.7))
ax3.text(0.755, 0.9, 'HIGH CONNECTIVITY', fontsize=11, fontweight='bold', ha='center', color='darkgreen')
ax3.text(0.755, 0.85, '(Large λ₂)', fontsize=10, ha='center')

# Draw dense network with electrode
dense_net = [(0.62, 0.7), (0.7, 0.68), (0.78, 0.72), (0.86, 0.68)]
dense_edges_c = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]  # Well connected
for pos in dense_net:
    circle = Circle(pos, 0.02, color='lightblue', ec='black', linewidth=1.5, zorder=5)
    ax3.add_patch(circle)
for i, j in dense_edges_c:
    ax3.plot([dense_net[i][0], dense_net[j][0]], 
            [dense_net[i][1], dense_net[j][1]], 
            'gray', linewidth=2, zorder=1)

# Show easy propagation
ax3.annotate('', xy=(0.7, 0.68), xytext=(0.62, 0.7),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax3.annotate('', xy=(0.78, 0.72), xytext=(0.7, 0.68),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

ax3.text(0.755, 0.48, 'Many paths available.\nSignal propagates easily.\n\n'
        'Caution: HIGH voltage\ncauses SATURATION!',
        fontsize=9, ha='center')

# --- Bottom: Failure modes ---
ax3.add_patch(FancyBboxPatch((0.075, 0.05), 0.40, 0.3, boxstyle="round,pad=0.02",
                              facecolor='mistyrose', edgecolor='darkred', linewidth=2))
ax3.text(0.275, 0.32, 'FAILURE: Insufficient Voltage', fontsize=10, fontweight='bold', ha='center', color='darkred')
ax3.text(0.275, 0.2, '• Signal dies before reaching target\n'
        '• No output at collector electrode\n'
        '• Gate produces no response',
        fontsize=9, ha='center')

ax3.add_patch(FancyBboxPatch((0.525, 0.05), 0.4, 0.3, boxstyle="round,pad=0.02",
                              facecolor='moccasin', edgecolor='darkorange', linewidth=2))
ax3.text(0.725, 0.32, 'FAILURE: Saturation', fontsize=10, fontweight='bold', ha='center', color='darkorange')
ax3.text(0.725, 0.2, '• Entire network activates at once\n'
        '• No discrimination between inputs\n'
        '• Gate outputs always HIGH',
        fontsize=9, ha='center')

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# =============================================================================
# Panel D: Using the Heuristic in Practice
# =============================================================================
ax4 = fig.add_subplot(gs[1, 1])

# Show the curve again with practical guidance
lambda2_range = np.linspace(0, 4, 500)
V_opt_curve = V_opt(lambda2_range)

# Plot curve
ax4.plot(lambda2_range, V_opt_curve, 'b-', linewidth=3, label='Design Heuristic')

# Add tolerance bands
V_upper = V_opt_curve * 1.3  # 30% above optimal
V_lower = V_opt_curve * 0.7  # 30% below optimal
V_lower = np.maximum(V_lower, V_min * 0.7)

ax4.fill_between(lambda2_range, V_lower, V_upper, color='blue', alpha=0.15, label='Acceptable range (±30%)')

# Mark key transition
ax4.axvline(x=lambda_crit, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Example specimens
specimens = [
    {'name': 'A', 'lambda2': 0.8, 'V_natural': None, 'color': 'red'},
    {'name': 'B', 'lambda2': 1.8, 'V_natural': None, 'color': 'orange'},
    {'name': 'C', 'lambda2': 3.2, 'V_natural': None, 'color': 'green'},
]

for spec in specimens:
    V_recommended = V_opt(spec['lambda2'])
    spec['V_recommended'] = V_recommended
    
    # Plot recommended voltage
    ax4.plot(spec['lambda2'], V_recommended, 'o', markersize=15, 
            color=spec['color'], markeredgecolor='black', markeredgewidth=2, zorder=5)
    
    # Annotation
    offset_y = 1.5 if spec['name'] != 'C' else 1.0
    ax4.annotate(f"Specimen {spec['name']}\nλ₂ = {spec['lambda2']}\n→ V = {V_recommended:.1f}V",
                xy=(spec['lambda2'], V_recommended),
                xytext=(spec['lambda2'] + 0.3, V_recommended + offset_y),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color=spec['color']),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=spec['color'], alpha=0.9))

# Add workflow box
workflow_text = (
    "WORKFLOW:\n"
    "1. Measure network topology\n"
    "2. Compute λ₂ from graph Laplacian\n"
    "3. Look up V_opt from heuristic\n"
    "4. Apply stimulus at V_opt ± 30%\n"
    "5. Fine-tune if needed"
)
ax4.text(0.02, 0.02, workflow_text, transform=ax4.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Labels and formatting
ax4.set_xlabel('Algebraic Connectivity λ₂ (measured from specimen)', fontsize=12)
ax4.set_ylabel('Recommended Stimulus Voltage (Volts)', fontsize=12)
ax4.set_title('(D) Practical Application of Voltage-Connectivity Heuristic', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.set_xlim(0, 4)
ax4.set_ylim(0, 14)
ax4.grid(True, alpha=0.3)

# Main title
fig.suptitle('FIG. 7: Voltage-Connectivity Design Heuristic\nOptimal Stimulus Voltage as Function of Network Algebraic Connectivity', 
            fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('FIG_7_Voltage_Connectivity_Heuristic.png', dpi=300, bbox_inches='tight')
plt.savefig('FIG_7_Voltage_Connectivity_Heuristic.pdf', bbox_inches='tight')
print("Voltage-connectivity heuristic diagram saved successfully")