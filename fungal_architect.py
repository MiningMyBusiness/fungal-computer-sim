import numpy as np
import networkx as nx
from scipy.integrate import odeint
from scipy.spatial.distance import cdist, pdist, squareform
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ==========================================
# Configuration
# ==========================================
@dataclass
class SimulationConfig:
    """Configuration parameters for fungal network simulation."""
    # Network parameters
    num_nodes: int = 30
    area_size: float = 20.0
    
    # Physics constants (Pleurotus ostreatus)
    tau_v: float = 50.0
    tau_w: float = 800.0
    r_off: float = 100.0
    r_on: float = 10.0
    alpha: float = 0.01
    coupling_radius: float = 2.0
    
    # Simulation parameters
    time_end: float = 1500.0
    time_steps: int = 300
    stim_start_time: float = 100.0
    
    # Initial conditions
    v_init: float = -1.2
    w_init: float = -0.6
    m_init: float = 0.1
    
    # Optimization parameters
    num_test_graphs: int = 5
    n_optimization_calls: int = 30
    random_state: int = 42
    
    # XOR scoring thresholds
    signal_threshold: float = -0.5

# ==========================================
# 1. The Generator (Creates Random Fungal Environments)
# ==========================================
class FungalEnvironmentGenerator:
    """Creates randomized fungal networks to test generalization."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_nodes = config.num_nodes
        self.area_size = config.area_size
        
    def generate_new_graph(self, seed: Optional[int] = None) -> Dict:
        """Generates a unique random geometric graph (a new petri dish)."""
        if seed is not None:
            np.random.seed(seed)
            
        # Random Geometric Graph mimics hyphal growth
        # Radius represents the max distance for hyphal fusion (anastomosis)
        # Note: nx.random_geometric_graph uses [0,1] x [0,1] coordinate space
        connection_radius = 0.28  # Normalized radius in [0,1] space (roughly equivalent to area_size/3.5 when scaled)
        G = nx.random_geometric_graph(self.num_nodes, radius=connection_radius)
        
        # Validate connectivity
        if not nx.is_connected(G):
            # Add edges to make it connected
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                G.add_edge(node1, node2)
        
        pos = nx.get_node_attributes(G, 'pos')
        # Scale coordinates from [0,1] to [0, area_size] for physical units (mm)
        coords = np.array([pos[i] for i in range(self.num_nodes)]) * self.area_size
        # Update position dictionary with scaled coordinates
        pos = {i: tuple(coords[i]) for i in range(self.num_nodes)}
        
        # Return a lightweight object with the physics data
        env = {
            'G': G,
            'pos': pos,
            'coords': coords,
            'adj': nx.to_numpy_array(G),
            'edges': list(G.edges()),
            'dist_matrix': squareform(pdist(coords))
        }
        return env

    def run_simulation(self, env: Dict, input_indices: List[int], 
                      readout_index: int, voltage: float, duration: float,
                      return_full_solution: bool = False) -> Tuple:
        """Runs the ODE physics on the specific generated environment.
        
        Returns:
            If return_full_solution is False: peak voltage at readout node
            If return_full_solution is True: (time_array, solution_array, peak_voltage)
        """
        # Validate inputs
        n = self.num_nodes
        edge_list = env['edges']
        
        if not all(0 <= idx < n for idx in input_indices):
            raise ValueError(f"Input indices {input_indices} out of range [0, {n})")
        if not (0 <= readout_index < n):
            raise ValueError(f"Readout index {readout_index} out of range [0, {n})")
        
        # --- ODE Definitions (Local to this environment) ---
        def system_derivatives(state, t, stim_func):
            V = state[0:n]
            W = state[n:2*n]
            M = state[2*n:2*n+len(edge_list)]
            
            # Conductance
            conductances = 1.0 / (self.config.r_on * M + self.config.r_off * (1.0 - M))
            
            # Diffusion (Laplacian)
            I_coupling = np.zeros(n)
            for k, (u, v) in enumerate(edge_list):
                g = conductances[k]
                curr = g * (V[v] - V[u])
                I_coupling[u] += curr
                I_coupling[v] -= curr
            
            # Memristor dynamics
            dM_dt = np.zeros(len(edge_list))
            for k, (u, v) in enumerate(edge_list):
                flux = V[u] - V[v]
                dM_dt[k] = self.config.alpha * np.abs(flux) * (M[k] * (1.0 - M[k]) + 0.05)
                
            dV_dt = (V - (V**3)/3.0 - W + stim_func(t) + I_coupling) / self.config.tau_v
            dW_dt = (V + 0.7 - 0.8 * W) / self.config.tau_w
            return np.concatenate([dV_dt, dW_dt, dM_dt])
            
        # --- Protocol ---
        t = np.linspace(0, self.config.time_end, self.config.time_steps)
        
        def stim_protocol(time):
            I = np.zeros(n)
            if self.config.stim_start_time < time < (self.config.stim_start_time + duration):
                for idx in input_indices:
                    I[idx] = voltage
            return I

        state_0 = np.concatenate([
            self.config.v_init * np.ones(n),
            self.config.w_init * np.ones(n),
            self.config.m_init * np.ones(len(edge_list))
        ])
        sol = odeint(system_derivatives, state_0, t, args=(stim_protocol,))
        
        # Get peak voltage at readout node
        peak_voltage = np.max(sol[:, readout_index])
        
        if return_full_solution:
            return t, sol, peak_voltage
        return peak_voltage

# ==========================================
# 2. The Policy Applicator (The "General Algorithm")
# ==========================================
def apply_geometric_policy(env: Dict, target_input_dist: float, 
                          target_output_dist: float) -> Tuple[List[int], int]:
    """
    This represents the 'Algorithm' running on a new graph.
    It scans the new graph to find nodes that match the geometric parameters.
    
    Returns:
        Tuple of (input_node_list, output_node_index)
    """
    coords = env['coords']
    dist_mat = env['dist_matrix']
    num_nodes = len(coords)
    
    if num_nodes < 3:
        raise ValueError("Need at least 3 nodes for XOR gate")
    
    # 1. Find pair of inputs (A, B) closest to 'target_input_dist'
    # Efficient search: Get indices of upper triangle of distance matrix
    iu = np.triu_indices(num_nodes, k=1)
    dists = dist_mat[iu]
    
    # Find index of pair closest to target
    best_idx = np.argmin(np.abs(dists - target_input_dist))
    node_A, node_B = int(iu[0][best_idx]), int(iu[1][best_idx])
    
    # 2. Find output node closest to 'target_output_dist' from the MIDPOINT of A and B
    midpoint = (coords[node_A] + coords[node_B]) / 2.0
    
    # Calculate distance from midpoint to all other nodes
    dists_from_mid = cdist([midpoint], coords)[0]
    
    # Mask inputs so we don't pick them as output
    dists_from_mid[node_A] = float('inf')
    dists_from_mid[node_B] = float('inf')
    
    node_out = int(np.argmin(np.abs(dists_from_mid - target_output_dist)))
    
    return [node_A, node_B], node_out

# ==========================================
# 3. Optimization Results Storage
# ==========================================
class OptimizationResults:
    """Stores and manages optimization results for visualization."""
    
    def __init__(self):
        self.iteration_history = []
        self.score_history = []
        self.param_history = []
        self.test_graph_scores = []
        self.xor_results = []  # Store individual XOR test results
        
    def add_iteration(self, iteration: int, params: List[float], 
                     score: float, individual_scores: List[float],
                     xor_voltages: Optional[List[Dict]] = None):
        """Record results from one optimization iteration."""
        self.iteration_history.append(iteration)
        self.score_history.append(score)
        self.param_history.append(params)
        self.test_graph_scores.append(individual_scores)
        if xor_voltages:
            self.xor_results.append(xor_voltages)

# ==========================================
# 4. Robust Bayesian Optimization
# ==========================================
class FungalXOROptimizer:
    """Optimizes geometric rules for building XOR gates in fungal networks."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.generator = FungalEnvironmentGenerator(config)
        self.results = OptimizationResults()
        self.iteration_count = 0
        
        # The Design Space: We optimize RULES, not Coordinates.
        self.space = [
            Real(2.0, 15.0, name='target_input_sep'),
            Real(2.0, 15.0, name='target_output_dist'),
            Real(1.5, 5.0, name='voltage'),
            Real(50.0, 300.0, name='duration')
        ]
    
    def evaluate_xor_logic(self, env: Dict, input_nodes: List[int], 
                          output_node: int, voltage: float, 
                          duration: float) -> Tuple[float, Dict]:
        """Test XOR logic on a specific configuration.
        
        Returns:
            (score, voltage_dict) where voltage_dict contains v_00, v_01, v_10, v_11
        """
        # Run (1,0)
        v_10 = self.generator.run_simulation(env, [input_nodes[0]], output_node, voltage, duration)
        # Run (0,1)
        v_01 = self.generator.run_simulation(env, [input_nodes[1]], output_node, voltage, duration)
        # Run (1,1)
        v_11 = self.generator.run_simulation(env, input_nodes, output_node, voltage, duration)
        # Run (0,0) - use actual simulation for accuracy
        v_00 = self.generator.run_simulation(env, [], output_node, voltage, duration)
        
        voltages = {'v_00': v_00, 'v_01': v_01, 'v_10': v_10, 'v_11': v_11}
        
        # Score: XOR should have high output for (0,1) and (1,0), low for (0,0) and (1,1)
        min_high = min(v_10, v_01)
        max_low = max(v_00, v_11)
        
        # If signals are too weak, it's a fail
        if min_high < self.config.signal_threshold:
            score = -2.0
        else:
            score = min_high - max_low
            
        return score, voltages
    
    def objective_function(self, target_input_sep: float, target_output_dist: float,
                          voltage: float, duration: float) -> float:
        """Objective function for Bayesian optimization."""
        scores = []
        xor_voltages_list = []
        
        print(f"[{self.iteration_count+1}] Testing: InSep={target_input_sep:.1f}, "
              f"OutDist={target_output_dist:.1f}, V={voltage:.2f}, Dur={duration:.0f}...", end="")
        
        for i in range(self.config.num_test_graphs):
            try:
                # Create a fresh random graph
                env = self.generator.generate_new_graph(seed=np.random.randint(100000))
                
                # Apply the policy to find specific nodes
                input_nodes, output_node = apply_geometric_policy(
                    env, target_input_sep, target_output_dist
                )
                
                # Test XOR logic
                score, voltages = self.evaluate_xor_logic(
                    env, input_nodes, output_node, voltage, duration
                )
                scores.append(score)
                xor_voltages_list.append(voltages)
                
            except Exception as e:
                print(f"\n  Warning: Graph {i} failed: {e}")
                scores.append(-2.0)
                xor_voltages_list.append({'v_00': -1.2, 'v_01': -1.2, 'v_10': -1.2, 'v_11': -1.2})
        
        avg_score = np.mean(scores)
        print(f" → Avg Score: {avg_score:.4f}")
        
        # Store results
        self.results.add_iteration(
            self.iteration_count,
            [target_input_sep, target_output_dist, voltage, duration],
            avg_score,
            scores,
            xor_voltages_list
        )
        self.iteration_count += 1
        
        # Minimize negative score
        return -avg_score
    
    def optimize(self):
        """Run the Bayesian optimization."""
        print("\n" + "="*70)
        print("FUNGAL XOR GATE OPTIMIZATION")
        print("="*70)
        print(f"Configuration:")
        print(f"  - Network nodes: {self.config.num_nodes}")
        print(f"  - Test graphs per iteration: {self.config.num_test_graphs}")
        print(f"  - Optimization calls: {self.config.n_optimization_calls}")
        print(f"\nGoal: Find universal geometric rules for Fungal XOR gates.\n")
        
        @use_named_args(self.space)
        def objective_wrapper(**params):
            return self.objective_function(**params)
        
        res = gp_minimize(
            objective_wrapper,
            self.space,
            n_calls=self.config.n_optimization_calls,
            random_state=self.config.random_state,
            verbose=False
        )
        
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print("\nOptimized Universal Algorithm:")
        print(f"  1. Locate two input nodes ~{res.x[0]:.2f} mm apart")
        print(f"  2. Locate output node ~{res.x[1]:.2f} mm from their center")
        print(f"  3. Stimulate with {res.x[2]:.2f} V for {res.x[3]:.0f} ms")
        print(f"\nExpected Performance (Score): {-res.fun:.4f}")
        print(f"Total evaluations: {len(res.func_vals)}")
        print("="*70 + "\n")
        
        return res

# ==========================================
# 5. Comprehensive Visualization Module
# ==========================================
class FungalVisualization:
    """Comprehensive visualization suite for fungal XOR optimization."""
    
    def __init__(self, optimizer: FungalXOROptimizer, optimization_result):
        self.optimizer = optimizer
        self.result = optimization_result
        self.results = optimizer.results
        self.config = optimizer.config
        
    def plot_optimization_convergence(self, ax=None):
        """Plot optimization convergence over iterations."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = np.array(self.results.score_history)
        iterations = np.array(self.results.iteration_history)
        
        # Plot all scores
        ax.plot(iterations, scores, 'o-', alpha=0.6, label='Iteration Score', color='steelblue')
        
        # Plot best-so-far
        best_so_far = np.maximum.accumulate(scores)
        ax.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best So Far')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('XOR Score', fontsize=12)
        ax.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_parameter_evolution(self, ax=None):
        """Plot how parameters evolved during optimization."""
        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
        else:
            axes = [ax]
        
        param_names = ['Input Separation (mm)', 'Output Distance (mm)', 
                      'Voltage (V)', 'Duration (ms)']
        params_array = np.array(self.results.param_history)
        
        for i, (name, ax_i) in enumerate(zip(param_names, axes)):
            if i < params_array.shape[1]:
                ax_i.scatter(self.results.iteration_history, params_array[:, i], 
                           c=self.results.score_history, cmap='viridis', s=50, alpha=0.7)
                ax_i.axhline(self.result.x[i], color='r', linestyle='--', 
                           linewidth=2, label=f'Best: {self.result.x[i]:.2f}')
                ax_i.set_xlabel('Iteration', fontsize=10)
                ax_i.set_ylabel(name, fontsize=10)
                ax_i.set_title(f'{name} Evolution', fontsize=11, fontweight='bold')
                ax_i.legend()
                ax_i.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return axes
    
    def plot_score_distribution(self, ax=None):
        """Plot distribution of scores across test graphs."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Flatten all test graph scores
        all_scores = [score for scores in self.results.test_graph_scores for score in scores]
        
        ax.hist(all_scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(all_scores), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(all_scores):.3f}')
        ax.axvline(np.median(all_scores), color='orange', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(all_scores):.3f}')
        
        ax.set_xlabel('XOR Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Score Distribution Across All Test Graphs', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_network_example(self, ax=None, seed=42):
        """Visualize an example fungal network with selected nodes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Generate example environment
        env = self.optimizer.generator.generate_new_graph(seed=seed)
        input_nodes, output_node = apply_geometric_policy(
            env, self.result.x[0], self.result.x[1]
        )
        
        G = env['G']
        pos = env['pos']
        
        # Draw network
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        
        # Draw regular nodes
        regular_nodes = [n for n in G.nodes() if n not in input_nodes and n != output_node]
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, 
                              node_color='lightgray', node_size=200, ax=ax)
        
        # Draw input nodes
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes,
                              node_color='green', node_size=500, 
                              label='Inputs', ax=ax)
        
        # Draw output node
        nx.draw_networkx_nodes(G, pos, nodelist=[output_node],
                              node_color='red', node_size=500,
                              label='Output', ax=ax)
        
        # Draw labels for special nodes
        labels = {input_nodes[0]: 'A', input_nodes[1]: 'B', output_node: 'OUT'}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, 
                               font_weight='bold', ax=ax)
        
        ax.set_title('Example Fungal Network with XOR Configuration', 
                    fontsize=14, fontweight='bold')
        ax.legend(scatterpoints=1, fontsize=11)
        ax.axis('off')
        
        return ax
    
    def plot_xor_truth_table(self, ax=None, seed=42):
        """Visualize XOR truth table results."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate example and test
        env = self.optimizer.generator.generate_new_graph(seed=seed)
        input_nodes, output_node = apply_geometric_policy(
            env, self.result.x[0], self.result.x[1]
        )
        
        _, voltages = self.optimizer.evaluate_xor_logic(
            env, input_nodes, output_node, self.result.x[2], self.result.x[3]
        )
        
        # Create truth table visualization
        inputs = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
        outputs = [voltages['v_00'], voltages['v_01'], voltages['v_10'], voltages['v_11']]
        expected = ['LOW', 'HIGH', 'HIGH', 'LOW']
        colors = ['red' if exp == 'LOW' else 'green' for exp in expected]
        
        bars = ax.bar(inputs, outputs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(self.config.signal_threshold, color='orange', linestyle='--',
                  linewidth=2, label=f'Threshold ({self.config.signal_threshold})')
        
        ax.set_xlabel('Input Pattern (A, B)', fontsize=12)
        ax.set_ylabel('Output Voltage', fontsize=12)
        ax.set_title('XOR Truth Table - Voltage Response', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, outputs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        return ax
    
    def plot_voltage_dynamics(self, seed=42):
        """Plot voltage dynamics over time for all XOR inputs."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Generate example
        env = self.optimizer.generator.generate_new_graph(seed=seed)
        input_nodes, output_node = apply_geometric_policy(
            env, self.result.x[0], self.result.x[1]
        )
        
        # Test all input combinations
        test_cases = [
            ([], '(0,0)', 'Expected: LOW'),
            ([input_nodes[0]], '(1,0)', 'Expected: HIGH'),
            ([input_nodes[1]], '(0,1)', 'Expected: HIGH'),
            (input_nodes, '(1,1)', 'Expected: LOW')
        ]
        
        for ax, (inputs, label, expected) in zip(axes, test_cases):
            t, sol, peak = self.optimizer.generator.run_simulation(
                env, inputs, output_node, self.result.x[2], self.result.x[3],
                return_full_solution=True
            )
            
            # Plot output node voltage
            ax.plot(t, sol[:, output_node], linewidth=2, color='steelblue')
            ax.axhline(self.config.signal_threshold, color='orange', 
                      linestyle='--', alpha=0.7, label='Threshold')
            ax.axvline(self.config.stim_start_time, color='green',
                      linestyle=':', alpha=0.5, label='Stim Start')
            ax.axvline(self.config.stim_start_time + self.result.x[3], 
                      color='red', linestyle=':', alpha=0.5, label='Stim End')
            
            ax.set_xlabel('Time (ms)', fontsize=10)
            ax.set_ylabel('Voltage', fontsize=10)
            ax.set_title(f'{label} - {expected}\nPeak: {peak:.3f}', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self, save_path=None):
        """Create a comprehensive visualization dashboard."""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Optimization metrics
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_optimization_convergence(ax1)
        
        # Row 2: Parameter evolution (2x2 grid)
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])
        
        params_array = np.array(self.results.param_history)
        param_names = ['Input Sep (mm)', 'Output Dist (mm)', 'Voltage (V)']
        
        for i, (ax, name) in enumerate(zip([ax2, ax3, ax4], param_names)):
            ax.scatter(self.results.iteration_history, params_array[:, i],
                      c=self.results.score_history, cmap='viridis', s=50, alpha=0.7)
            ax.axhline(self.result.x[i], color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Iteration', fontsize=9)
            ax.set_ylabel(name, fontsize=9)
            ax.set_title(f'{name} Evolution', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Row 3: Network example, score distribution, truth table
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_network_example(ax5)
        
        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_score_distribution(ax6)
        
        ax7 = fig.add_subplot(gs[2, 2])
        self.plot_xor_truth_table(ax7)
        
        fig.suptitle('Fungal XOR Gate Optimization - Comprehensive Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        return fig

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    # Create configuration
    config = SimulationConfig(
        num_nodes=30,
        area_size=20.0,
        num_test_graphs=5,
        n_optimization_calls=30,
        random_state=42
    )
    
    # Run optimization
    optimizer = FungalXOROptimizer(config)
    result = optimizer.optimize()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    viz = FungalVisualization(optimizer, result)
    
    # Create comprehensive dashboard
    dashboard = viz.create_comprehensive_dashboard(save_path='fungal_xor_dashboard.png')
    
    # Create voltage dynamics plot
    dynamics_fig = viz.plot_voltage_dynamics()
    dynamics_fig.savefig('fungal_voltage_dynamics.png', dpi=300, bbox_inches='tight')
    print("Voltage dynamics saved to: fungal_voltage_dynamics.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("All visualizations complete!")
    print("="*70)