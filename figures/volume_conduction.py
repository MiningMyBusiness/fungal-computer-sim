import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Set up the figure with multiple panels
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Parameters
c_0 = 1.0      # Maximum coupling strength
lambda_decay = 0.3  # Characteristic decay length

# Lorentzian coupling function
def lorentzian_coupling(d, c0=c_0, lam=lambda_decay):
    return c0 / (1 + (d/lam)**2)

# =============================================================================
# Panel A: Lorentzian Decay Profile (1D)
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])

d_range = np.linspace(0, 2, 500)
c_lorentzian = lorentzian_coupling(d_range)

# Also show exponential and Gaussian for comparison
c_exponential = c_0 * np.exp(-d_range/lambda_decay)
c_gaussian = c_0 * np.exp(-(d_range/lambda_decay)**2)

ax1.plot(d_range, c_lorentzian, 'b-', linewidth=3, label='Lorentzian (this model)')
ax1.plot(d_range, c_exponential, 'r--', linewidth=2, alpha=0.7, label='Exponential')
ax1.plot(d_range, c_gaussian, 'g:', linewidth=2, alpha=0.7, label='Gaussian')

# Mark key points
ax1.axhline(y=0.5*c_0, color='gray', linestyle='--', linewidth=1)
ax1.axvline(x=lambda_decay, color='blue', linestyle='--', linewidth=1, alpha=0.5)
ax1.plot(lambda_decay, lorentzian_coupling(lambda_decay), 'bo', markersize=10)
ax1.annotate(f'd = λ\nc = c₀/2', xy=(lambda_decay, 0.5*c_0), 
            xytext=(lambda_decay + 0.3, 0.6*c_0),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'))

# Add equation
eq_text = r'$c_{ik} = \frac{c_0}{1 + (d_{ik}/\lambda)^2}$'
ax1.text(0.55, 0.85, eq_text, fontsize=14, transform=ax1.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax1.set_xlabel('Distance from electrode (d)', fontsize=11)
ax1.set_ylabel('Coupling coefficient (c)', fontsize=11)
ax1.set_title('(A) Lorentzian Decay Profile\nvs. Other Decay Functions', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 1.1)

# Add parameter annotation
param_text = f'c₀ = {c_0}\nλ = {lambda_decay}'
ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# =============================================================================
# Panel B: 2D Spatial Decay (Heatmap)
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Create 2D grid
x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)

# Electrode at origin
d_2d = np.sqrt(X**2 + Y**2)
c_2d = lorentzian_coupling(d_2d)

# Plot heatmap
im = ax2.imshow(c_2d, extent=[-1.5, 1.5, -1.5, 1.5], origin='lower', 
               cmap='hot', vmin=0, vmax=1)
plt.colorbar(im, ax=ax2, label='Coupling strength c', shrink=0.8)

# Add contour lines
contour_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
cs = ax2.contour(X, Y, c_2d, levels=contour_levels, colors='white', linewidths=1, alpha=0.7)
ax2.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

# Mark electrode position
ax2.plot(0, 0, 'c*', markersize=20, markeredgecolor='white', markeredgewidth=2, label='Electrode')
ax2.add_patch(Circle((0, 0), lambda_decay, fill=False, color='cyan', linewidth=2, linestyle='--'))
ax2.annotate('λ radius', xy=(lambda_decay, 0), xytext=(lambda_decay + 0.2, 0.3),
            fontsize=10, color='cyan', arrowprops=dict(arrowstyle='->', color='cyan'))

ax2.set_xlabel('x position', fontsize=11)
ax2.set_ylabel('y position', fontsize=11)
ax2.set_title('(B) 2D Coupling Field Around\na Single Electrode', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')
ax2.legend(loc='upper right')

# =============================================================================
# Panel C: Schematic of Electrode-Network Coupling
# =============================================================================
ax3 = fig.add_subplot(gs[0, 2])

# Create a schematic showing electrode coupling to network nodes
np.random.seed(42)

# Electrode position
electrode_pos = np.array([0.5, 0.5])

# Generate random network nodes
n_nodes = 25
node_positions = np.random.rand(n_nodes, 2)

# Calculate distances and coupling strengths
distances = np.sqrt(np.sum((node_positions - electrode_pos)**2, axis=1))
couplings = lorentzian_coupling(distances, lam=0.25)

# Plot nodes with size/color based on coupling strength
scatter = ax3.scatter(node_positions[:, 0], node_positions[:, 1], 
                     c=couplings, s=100 + 300*couplings, cmap='Reds', 
                     edgecolors='black', linewidths=1, vmin=0, vmax=1, zorder=3)
plt.colorbar(scatter, ax=ax3, label='Coupling c_ik', shrink=0.8)

# Draw lines from electrode to nodes (thickness = coupling strength)
for i, (pos, c) in enumerate(zip(node_positions, couplings)):
    if c > 0.1:  # Only draw significant couplings
        ax3.plot([electrode_pos[0], pos[0]], [electrode_pos[1], pos[1]], 
                'b-', linewidth=3*c, alpha=0.5*c + 0.2)

# Plot electrode
ax3.plot(electrode_pos[0], electrode_pos[1], 'b^', markersize=25, 
        markeredgecolor='black', markeredgewidth=2, label='Electrode k', zorder=5)

# Add some network edges between nearby nodes
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        d_ij = np.sqrt(np.sum((node_positions[i] - node_positions[j])**2))
        if d_ij < 0.2:
            ax3.plot([node_positions[i,0], node_positions[j,0]], 
                    [node_positions[i,1], node_positions[j,1]], 
                    'gray', linewidth=1, alpha=0.5, zorder=1)

ax3.set_xlabel('x position', fontsize=11)
ax3.set_ylabel('y position', fontsize=11)
ax3.set_title('(C) Electrode Coupling to\nNetwork Nodes', fontsize=12, fontweight='bold')
ax3.set_xlim(-0.05, 1.05)
ax3.set_ylim(-0.05, 1.05)
ax3.legend(loc='upper right')
ax3.set_aspect('equal')

# Add annotation
ax3.text(0.02, 0.02, 'Node size & color ∝ coupling\nLine width ∝ coupling', 
        transform=ax3.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# =============================================================================
# Panel D: Point Contact vs Volume Conduction Comparison
# =============================================================================
ax4 = fig.add_subplot(gs[1, 0])

# Show the difference between idealized point contact and realistic volume conduction
d_range = np.linspace(0, 1, 500)

# Point contact (delta function approximation - step function for visualization)
c_point = np.where(d_range < 0.05, 1.0, 0.0)

# Volume conduction (Lorentzian)
c_volume = lorentzian_coupling(d_range, lam=0.2)

ax4.fill_between(d_range, 0, c_point, alpha=0.3, color='red', label='Point contact (idealized)')
ax4.plot(d_range, c_point, 'r-', linewidth=2)
ax4.fill_between(d_range, 0, c_volume, alpha=0.3, color='blue', label='Volume conduction (realistic)')
ax4.plot(d_range, c_volume, 'b-', linewidth=2)

ax4.set_xlabel('Distance from electrode', fontsize=11)
ax4.set_ylabel('Coupling strength', fontsize=11)
ax4.set_title('(D) Point Contact vs. Volume Conduction', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)

# Add explanatory text
ax4.text(0.5, 0.7, 'Point contact:\nCouples only to\nnearest node', fontsize=9, 
        color='red', ha='center', transform=ax4.transAxes)
ax4.text(0.75, 0.4, 'Volume conduction:\n"Soft" coupling to\nmultiple nodes', fontsize=9, 
        color='blue', ha='center', transform=ax4.transAxes)

# =============================================================================
# Panel E: Multiple Electrodes with Overlapping Fields
# =============================================================================
ax5 = fig.add_subplot(gs[1, 1])

# Two electrodes
electrode1 = np.array([-0.5, 0])
electrode2 = np.array([0.5, 0])

# Create 2D grid
x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1, 1, 150)
X, Y = np.meshgrid(x, y)

# Distance from each electrode
d1 = np.sqrt((X - electrode1[0])**2 + (Y - electrode1[1])**2)
d2 = np.sqrt((X - electrode2[0])**2 + (Y - electrode2[1])**2)

# Coupling from each (note: could be additive or independent depending on model)
c1 = lorentzian_coupling(d1, lam=0.4)
c2 = lorentzian_coupling(d2, lam=0.4)

# Show combined field (for visualization, show max of the two)
c_combined = np.maximum(c1, c2)

im5 = ax5.imshow(c_combined, extent=[-1.5, 1.5, -1, 1], origin='lower', 
                cmap='hot', vmin=0, vmax=1)
plt.colorbar(im5, ax=ax5, label='Coupling strength', shrink=0.8)

# Add contours
cs5 = ax5.contour(X, Y, c1, levels=[0.5], colors='cyan', linewidths=2, linestyles='--')
cs5b = ax5.contour(X, Y, c2, levels=[0.5], colors='lime', linewidths=2, linestyles='--')

# Mark electrodes
ax5.plot(electrode1[0], electrode1[1], 'c^', markersize=18, markeredgecolor='white', 
        markeredgewidth=2, label='Input electrode 1')
ax5.plot(electrode2[0], electrode2[1], 'g^', markersize=18, markeredgecolor='white', 
        markeredgewidth=2, label='Input electrode 2')

ax5.set_xlabel('x position', fontsize=11)
ax5.set_ylabel('y position', fontsize=11)
ax5.set_title('(E) Two Input Electrodes with\nOverlapping Coupling Fields', fontsize=12, fontweight='bold')
ax5.legend(loc='upper right', fontsize=9)
ax5.set_aspect('equal')

# Mark overlap region
ax5.annotate('Overlap\nregion', xy=(0, 0), xytext=(0, 0.6),
            fontsize=10, ha='center', color='white',
            arrowprops=dict(arrowstyle='->', color='white'))

# =============================================================================
# Panel F: Physical Interpretation
# =============================================================================
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

# Create a schematic illustration
# Draw electrode
electrode_x, electrode_y = 0.2, 0.7
ax6.annotate('', xy=(electrode_x, electrode_y-0.15), xytext=(electrode_x, electrode_y+0.1),
            arrowprops=dict(arrowstyle='simple', fc='blue', ec='black', lw=2))
ax6.text(electrode_x, electrode_y+0.15, 'Metal\nElectrode', fontsize=10, ha='center', fontweight='bold')

# Draw tissue/mycelium region
tissue_rect = mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.45, 
                                       boxstyle="round,pad=0.02",
                                       facecolor='tan', edgecolor='brown', linewidth=2, alpha=0.7)
ax6.add_patch(tissue_rect)
ax6.text(0.5, 0.05, 'Fungal Mycelium Tissue', fontsize=11, ha='center', fontweight='bold')

# Draw current spread lines (radiating from electrode tip)
for angle in np.linspace(-60, 60, 7):
    rad = np.radians(angle)
    length = 0.3 + 0.1 * np.cos(rad)  # Longer in middle
    end_x = electrode_x + length * np.sin(rad)
    end_y = electrode_y - 0.15 - length * np.cos(rad)
    alpha = 1.0 - abs(angle)/90
    ax6.annotate('', xy=(end_x, end_y), xytext=(electrode_x, electrode_y-0.15),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=alpha))

# Draw some hyphal nodes
node_positions_schematic = [
    (0.15, 0.35), (0.25, 0.25), (0.35, 0.4), (0.45, 0.3),
    (0.55, 0.35), (0.65, 0.25), (0.75, 0.4), (0.85, 0.3),
    (0.3, 0.15), (0.5, 0.2), (0.7, 0.15)
]
for pos in node_positions_schematic:
    d = np.sqrt((pos[0] - electrode_x)**2 + (pos[1] - (electrode_y-0.15))**2)
    c = lorentzian_coupling(d, lam=0.2)
    size = 100 + 400*c
    color = plt.cm.Reds(c)
    ax6.scatter(pos[0], pos[1], s=size, c=[color], edgecolors='black', zorder=5)

# Draw hyphal connections
connections = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (1,8), (3,9), (5,10), (8,9), (9,10)]
for i, j in connections:
    ax6.plot([node_positions_schematic[i][0], node_positions_schematic[j][0]],
            [node_positions_schematic[i][1], node_positions_schematic[j][1]],
            'brown', linewidth=2, alpha=0.7, zorder=1)

# Add labels
ax6.text(0.5, 0.85, '(F) Physical Interpretation', fontsize=12, fontweight='bold', ha='center')
ax6.text(0.95, 0.6, 'Current spreads\nthrough tissue\n(volume conduction)', fontsize=9, ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax6.text(0.95, 0.3, 'Nearby nodes receive\nstronger coupling\n(Lorentzian decay)', fontsize=9, ha='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Legend for node colors
ax6.scatter([], [], s=400, c='darkred', edgecolors='black', label='Strong coupling')
ax6.scatter([], [], s=150, c='lightsalmon', edgecolors='black', label='Weak coupling')
ax6.legend(loc='lower left', fontsize=9)

ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

# Main title
fig.suptitle('FIG. 5: Volume Conduction Model for Electrode-Tissue Coupling', 
            fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/FIG_5_Volume_Conduction.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/FIG_5_Volume_Conduction.pdf', bbox_inches='tight')
print("Volume conduction diagram saved successfully")