"""
Generate a comprehensive figure showing the three characterization protocols,
example waveform responses, and extracted features for scientific publication.

This script creates a multi-panel figure demonstrating:
1. Step Response Protocol - with rise time, saturation voltage, and oscillation features
2. Paired-Pulse Protocol - with recovery ratios at different delays
3. Triangle Sweep Protocol - with hysteresis area measurement

Author: Generated for fungal computer characterization study
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec
from realistic_sim import RealisticFungalComputer
from scipy.signal import find_peaks
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def create_characterization_figure(save_path='figures/characterization_protocols_figure.png', 
                                   save_pdf=True, dpi=300):
    """
    Create a comprehensive figure showing all three characterization protocols.
    
    Args:
        save_path: Path to save the PNG figure
        save_pdf: Whether to also save a PDF version
        dpi: Resolution for saved figure
    """
    print("Initializing RealisticFungalComputer...")
    # Create a fungal computer instance with moderate complexity
    env = RealisticFungalComputer(num_nodes=50, area_size=20.0, random_seed=42)
    
    # Set interesting parameters that show clear features
    env.tau_v = 60.0
    env.tau_w = 900.0
    env.a = 0.65
    env.b = 0.82
    env.v_scale = 4.5
    env.R_off = 150.0
    env.R_on = 15.0
    env.alpha = 0.008
    
    print(f"Network: {env.num_nodes} nodes, {len(env.edge_list)} edges")
    
    # Create figure with custom layout - left panel for network, right 2x3 for protocols
    fig = plt.figure(figsize=(18, 10))
    # Create main grid: 1 column for network, 3 columns for protocols
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1, 2.5], 
                       left=0.05, right=0.98, top=0.92, bottom=0.06, wspace=0.25)
    
    # Left panel for network topology
    gs_network = gs_main[0].subgridspec(1, 1)
    
    # Right panel subdivided into 2x3 for protocols (2 rows, 3 columns)
    gs_protocols = gs_main[1].subgridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # ========================================================================
    # NETWORK TOPOLOGY PANEL (LEFT)
    # ========================================================================
    print("\n" + "="*60)
    print("Creating Network Topology Panel...")
    print("="*60)
    
    ax_network = fig.add_subplot(gs_network[0, 0])
    
    # Draw the network
    pos = {i: env.pos[i] for i in range(env.num_nodes)}
    
    # Draw edges with more visible styling
    nx.draw_networkx_edges(env.G, pos, ax=ax_network, 
                          edge_color='#666666', width=1.0, alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(env.G, pos, ax=ax_network,
                          node_size=40, node_color='#87CEEB', 
                          edgecolors='black', linewidths=0.5, alpha=0.8)
    
    # Mark electrode positions for all three protocols
    # Step response electrodes (center and probe)
    center = (env.area_size / 2, env.area_size / 2)
    probe_step = (env.area_size / 2 + 5.0, env.area_size / 2)
    
    # Mark electrodes
    ax_network.plot(center[0], center[1], 'r*', markersize=20, 
                   markeredgecolor='darkred', markeredgewidth=1.5, 
                   label='Stimulus Electrode', zorder=10)
    ax_network.plot(probe_step[0], probe_step[1], 'g^', markersize=15, 
                   markeredgecolor='darkgreen', markeredgewidth=1.5,
                   label='Output Probe', zorder=10)
    
    ax_network.set_xlim([-1, env.area_size + 1])
    ax_network.set_ylim([-1, env.area_size + 1])
    ax_network.set_xlabel('X Position (mm)', fontsize=12)
    ax_network.set_ylabel('Y Position (mm)', fontsize=12)
    ax_network.set_title('Network Topology & Electrode Configuration', 
                        fontweight='bold', fontsize=13, pad=10)
    ax_network.set_box_aspect(1)
    ax_network.legend(loc='upper left', framealpha=0.95, fontsize=9)
    ax_network.grid(True, alpha=0.2)
    
    # Add network statistics text box
    stats_text = f"Network Statistics:\n" \
                f"Nodes: {env.num_nodes}\n" \
                f"Edges: {len(env.edge_list)}\n" \
                f"Density: {nx.density(env.G):.3f}\n" \
                f"Avg Degree: {np.mean([env.G.degree(n) for n in env.G.nodes()]):.1f}\n" \
                f"Area: {env.area_size}×{env.area_size} mm²"
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                edgecolor='black', linewidth=1.5)
    ax_network.text(0.02, 0.02, stats_text, transform=ax_network.transAxes,
                   fontsize=9, verticalalignment='bottom', bbox=props, 
                   family='monospace')
    
    print(f"Network visualization complete: {env.num_nodes} nodes, {len(env.edge_list)} edges")
    
    # ========================================================================
    # PANEL A: STEP RESPONSE PROTOCOL
    # ========================================================================
    print("\n" + "="*60)
    print("Running Step Response Protocol...")
    print("="*60)
    
    ax_step_stim = fig.add_subplot(gs_protocols[0, 0])
    ax_step_response = fig.add_subplot(gs_protocols[1, 0])
    
    # Run step response protocol
    step_results = env.step_response_protocol(
        voltage=2.0,
        pulse_duration=1000.0,
        probe_distance=5.0,
        sim_time=5000.0
    )
    
    t_step = step_results['time']
    v_step = step_results['response']
    rise_time = step_results['activation_speed']
    sat_voltage = step_results['peak_amplitude']
    osc_index = step_results['settling_deviation']
    
    print(f"Step Response Features:")
    print(f"  Activation speed (rise time): {rise_time:.1f} ms")
    print(f"  Peak amplitude: {sat_voltage:.3f} V")
    print(f"  Settling deviation: {osc_index:.4f}")
    
    # Plot stimulus
    pulse_start = 100.0
    pulse_end = pulse_start + 1000.0
    stim_signal = np.zeros_like(t_step)
    stim_signal[(t_step >= pulse_start) & (t_step < pulse_end)] = 2.0
    
    ax_step_stim.plot(t_step, stim_signal, 'k-', linewidth=2)
    ax_step_stim.fill_between(t_step, 0, stim_signal, alpha=0.3, color='red')
    ax_step_stim.set_xlabel('Time (ms)')
    ax_step_stim.set_ylabel('Stimulus (V)')
    ax_step_stim.set_title('A1. Step Response Stimulus', fontweight='bold', loc='left')
    ax_step_stim.set_xlim([0, 5000])
    ax_step_stim.set_ylim([-0.2, 2.5])
    ax_step_stim.grid(True, alpha=0.3)
    ax_step_stim.text(0.02, 0.95, 'DC Pulse\n3000 ms', transform=ax_step_stim.transAxes,
                      va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', 
                      facecolor='wheat', alpha=0.5))
    ax_step_stim.set_box_aspect(1)
    
    # Plot response
    ax_step_response.plot(t_step, v_step, 'b-', linewidth=1.5, label='Response')
    ax_step_response.axhline(sat_voltage, color='green', linestyle='--', 
                            linewidth=1, label=f'Peak: {sat_voltage:.3f}V')
    ax_step_response.axhline(0.9 * sat_voltage, color='orange', linestyle=':', 
                            linewidth=1, label=f'90% Level')
    ax_step_response.axvspan(pulse_start, pulse_end, alpha=0.1, color='red')
    ax_step_response.set_xlabel('Time (ms)')
    ax_step_response.set_ylabel('Output Voltage (V)')
    ax_step_response.set_title('A2. Voltage Response', fontweight='bold', loc='left')
    ax_step_response.set_xlim([0, 5000])
    ax_step_response.legend(loc='lower right', framealpha=0.9)
    ax_step_response.grid(True, alpha=0.3)
    ax_step_response.set_box_aspect(1)
    
    # ========================================================================
    # PANEL B: PAIRED-PULSE PROTOCOL
    # ========================================================================
    print("\n" + "="*60)
    print("Running Paired-Pulse Protocol...")
    print("="*60)
    
    ax_pp_stim = fig.add_subplot(gs_protocols[0, 1])
    ax_pp_response = fig.add_subplot(gs_protocols[1, 1])
    
    # Run paired-pulse protocol
    pp_results = env.paired_pulse_protocol(
        voltage=2.0,
        pulse_width=50.0,
        probe_distance=5.0,
        delays=[200.0, 800.0, 2000.0]
    )
    
    delays = pp_results['delays']
    results = pp_results['results']
    
    # Extract arrays from results list
    recovery_ratios = np.array([r['peak_ratio'] for r in results])
    first_peaks = np.array([r['peak1_amplitude'] for r in results])
    second_peaks = np.array([r['peak2_amplitude'] for r in results])
    
    print(f"Paired-Pulse Features:")
    for i, (delay, ratio) in enumerate(zip(delays, recovery_ratios)):
        print(f"  Delay {delay:.0f}ms: Peak ratio = {ratio:.3f}")
    
    # Plot stimulus for middle delay (800ms) as example
    example_delay = 800.0
    pulse_start = 100.0
    pulse_width = 50.0
    
    # Create time array for stimulus visualization
    t_pp_stim = np.linspace(0, pulse_start + 2*pulse_width + example_delay + 500, 1000)
    stim_pp = np.zeros_like(t_pp_stim)
    # First pulse
    stim_pp[(t_pp_stim >= pulse_start) & (t_pp_stim < pulse_start + pulse_width)] = 2.0
    # Second pulse
    stim_pp[(t_pp_stim >= pulse_start + pulse_width + example_delay) & 
            (t_pp_stim < pulse_start + 2*pulse_width + example_delay)] = 2.0
    
    ax_pp_stim.plot(t_pp_stim, stim_pp, 'k-', linewidth=2)
    ax_pp_stim.fill_between(t_pp_stim, 0, stim_pp, alpha=0.3, color='purple')
    ax_pp_stim.set_xlabel('Time (ms)')
    ax_pp_stim.set_ylabel('Stimulus (V)')
    ax_pp_stim.set_title('B1. Paired-Pulse Stimulus', fontweight='bold', loc='left')
    ax_pp_stim.set_xlim([0, t_pp_stim[-1]])
    ax_pp_stim.set_ylim([-0.2, 2.5])
    ax_pp_stim.grid(True, alpha=0.3)
    ax_pp_stim.set_box_aspect(1)
    
    # Add annotation for delay
    ax_pp_stim.annotate('', xy=(pulse_start + pulse_width + example_delay, 2.3),
                       xytext=(pulse_start + pulse_width, 2.3),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax_pp_stim.text(pulse_start + pulse_width + example_delay/2, 2.4, 
                   f'Δt = {example_delay:.0f} ms',
                   ha='center', fontsize=9, color='red', fontweight='bold')
    
    # For response, run a single paired-pulse experiment at 800ms delay
    center = (env.area_size / 2, env.area_size / 2)
    probe = (env.area_size / 2 + 5.0, env.area_size / 2)
    coupling_map = env.calculate_stimulation_coupling(center, 2.0)
    
    def stim_800(t):
        if pulse_start < t < (pulse_start + pulse_width):
            return coupling_map
        if (pulse_start + pulse_width + 800.0) < t < (pulse_start + 2 * pulse_width + 800.0):
            return coupling_map
        return np.zeros(env.num_nodes)
    
    sim_time = pulse_start + 2 * pulse_width + 800.0 + 1000.0
    t_pp, sol_pp = env.run_experiment_custom_stim(sim_time, stim_800)
    v_pp = np.array([env.read_output_voltage(probe, sol_pp[i, :]) for i in range(len(sol_pp))])
    
    # Find peaks
    peaks, _ = find_peaks(v_pp, height=0.1, distance=int(20.0 / 5.0))
    
    ax_pp_response.plot(t_pp, v_pp, 'b-', linewidth=1.5, label='Response')
    if len(peaks) >= 2:
        sorted_peaks = peaks[np.argsort(v_pp[peaks])[-2:]]
        sorted_peaks = sorted_peaks[np.argsort(sorted_peaks)]
        ax_pp_response.plot(t_pp[sorted_peaks[0]], v_pp[sorted_peaks[0]], 'ro', 
                           markersize=8, label=f'Peak 1: {v_pp[sorted_peaks[0]]:.3f}V')
        ax_pp_response.plot(t_pp[sorted_peaks[1]], v_pp[sorted_peaks[1]], 'go', 
                           markersize=8, label=f'Peak 2: {v_pp[sorted_peaks[1]]:.3f}V')
    
    ax_pp_response.axvspan(pulse_start, pulse_start + pulse_width, alpha=0.1, color='purple')
    ax_pp_response.axvspan(pulse_start + pulse_width + 800.0, 
                          pulse_start + 2*pulse_width + 800.0, alpha=0.1, color='purple')
    ax_pp_response.set_xlabel('Time (ms)')
    ax_pp_response.set_ylabel('Output Voltage (V)')
    ax_pp_response.set_title('B2. Voltage Response (Δt=800ms)', fontweight='bold', loc='left')
    ax_pp_response.legend(loc='best', framealpha=0.9, fontsize=8)
    ax_pp_response.grid(True, alpha=0.3)
    ax_pp_response.set_box_aspect(1)
    
    # ========================================================================
    # PANEL C: TRIANGLE SWEEP PROTOCOL
    # ========================================================================
    print("\n" + "="*60)
    print("Running Triangle Sweep Protocol...")
    print("="*60)
    
    ax_tri_stim = fig.add_subplot(gs_protocols[0, 2])
    ax_tri_response = fig.add_subplot(gs_protocols[1, 2])
    
    # Run triangle sweep protocol
    tri_results = env.triangle_sweep_protocol(
        v_max=5.0,
        sweep_rate=0.01,
        probe_distance=5.0
    )
    
    t_tri = tri_results['time']
    v_applied = tri_results['voltage_applied']
    i_response = tri_results['current_response']
    v_response = tri_results['voltage_response']
    hysteresis_area = tri_results['total_hysteresis_area']
    
    print(f"Triangle Sweep Features:")
    print(f"  Total hysteresis area: {hysteresis_area:.4f}")
    
    # Plot applied voltage (stimulus)
    ax_tri_stim.plot(t_tri, v_applied, 'k-', linewidth=2)
    ax_tri_stim.fill_between(t_tri, 0, v_applied, where=(v_applied >= 0), 
                            alpha=0.3, color='red', label='Positive')
    ax_tri_stim.fill_between(t_tri, 0, v_applied, where=(v_applied < 0), 
                            alpha=0.3, color='blue', label='Negative')
    ax_tri_stim.set_xlabel('Time (ms)')
    ax_tri_stim.set_ylabel('Applied Voltage (V)')
    ax_tri_stim.set_title('C1. Triangle Sweep Stimulus', fontweight='bold', loc='left')
    ax_tri_stim.grid(True, alpha=0.3)
    ax_tri_stim.legend(loc='upper right', framealpha=0.9)
    ax_tri_stim.axhline(0, color='black', linewidth=0.5)
    ax_tri_stim.set_box_aspect(1)
    
    # Plot voltage response over time
    ax_tri_response.plot(t_tri, v_response, 'b-', linewidth=1.5, label='Voltage Response')
    ax_tri_response.plot(t_tri, v_applied * 0.1, 'k--', linewidth=1, alpha=0.5, 
                        label='Stimulus (scaled)')
    ax_tri_response.set_xlabel('Time (ms)')
    ax_tri_response.set_ylabel('Output Voltage (V)')
    ax_tri_response.set_title('C2. Voltage Response', fontweight='bold', loc='left')
    ax_tri_response.legend(loc='upper right', framealpha=0.9)
    ax_tri_response.grid(True, alpha=0.3)
    ax_tri_response.set_box_aspect(1)
    
    # Add main title with adjusted position
    fig.suptitle('Fungal Computer Characterization Protocols: Stimulus and Response',
                fontsize=14, fontweight='bold', y=0.985)
    
    # Save figure
    print("\n" + "="*60)
    print(f"Saving figure to {save_path}...")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"PNG saved successfully!")
    
    if save_pdf:
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"PDF saved to {pdf_path}")
    
    print("="*60)
    print("Figure generation complete!")
    print("="*60)
    
    return fig

if __name__ == '__main__':
    # Create the figure
    fig = create_characterization_figure(
        save_path='figures/characterization_protocols_figure.png',
        save_pdf=True,
        dpi=300
    )
    
    # Display the figure
    plt.show()
    
    print("\nFigure Summary:")
    print("- Panel A: Step Response Protocol (rise time, saturation, oscillation)")
    print("- Panel B: Paired-Pulse Protocol (recovery dynamics)")
    print("- Panel C: Triangle Sweep Protocol (hysteresis/plasticity)")
    print("\nThis figure is ready for publication in your scientific paper.")
