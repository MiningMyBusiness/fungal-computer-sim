"""Realistic Fungal Computer Simulation with FitzHugh-Nagumo dynamics and memristive connections."""

import numpy as np
import networkx as nx
from scipy.integrate import odeint
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings
import logging
import time
from typing import List, Tuple, Callable, Optional

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Simulation Constants
DEFAULT_TIME_STEP = 5.0  # ms
DEFAULT_PULSE_START = 100.0  # ms
DEFAULT_INITIAL_V = -1.2  # Initial voltage
DEFAULT_INITIAL_W = -0.6  # Initial recovery variable
DEFAULT_INITIAL_M = 0.1  # Initial memristor state
MIN_ELECTRODE_DISTANCE = 2.0  # mm
ELECTRODE_PROXIMITY_PENALTY = 5.0

class RealisticFungalComputer:
    """Simulates a fungal network computer with FitzHugh-Nagumo dynamics and memristive connections.
    
    The network is modeled as a random geometric graph where nodes represent hyphal junctions
    and edges represent connections with adaptive memristive conductance.
    
    Attributes:
        num_nodes: Number of nodes in the fungal network
        area_size: Size of the 2D area containing the network (mm)
        G: NetworkX graph representing the fungal network
        node_coords: Numpy array of (x, y) coordinates for each node
        edge_list: List of edges in the network
    """
    
    def __init__(self, num_nodes: int = 40, area_size: float = 20.0, random_seed: int = 42):
        """Initialize the fungal computer simulation.
        
        Args:
            num_nodes: Number of nodes in the network
            area_size: Size of the 2D simulation area in mm
            random_seed: Random seed for reproducibility
        """
        logger.info(f"Initializing RealisticFungalComputer with {num_nodes} nodes, area_size={area_size}mm, seed={random_seed}")
        start_time = time.time()
        
        np.random.seed(random_seed)
        self.num_nodes = num_nodes
        self.area_size = area_size
        
        # 1. Geometry: Random Geometric Graph
        # This mimics hyphae fusing only when physically close
        # Note: nx.random_geometric_graph generates positions in [0,1] x [0,1]
        # We need to scale the radius accordingly
        connection_radius = 0.25  # Connection radius in normalized [0,1] space
        logger.debug(f"Creating random geometric graph with radius={connection_radius} (normalized)")
        self.G = nx.random_geometric_graph(num_nodes, radius=connection_radius, seed=random_seed)
        self.pos = nx.get_node_attributes(self.G, 'pos')
        
        # Scale node coordinates from [0,1] to [0, area_size] to match physical dimensions
        self.node_coords = np.array([self.pos[i] for i in range(num_nodes)]) * area_size
        # Update the position dictionary with scaled coordinates
        self.pos = {i: tuple(self.node_coords[i]) for i in range(num_nodes)}
        logger.debug(f"Node coordinates scaled to [{0}, {area_size}]mm space")
        
        # Validate network connectivity
        if len(self.G.edges()) == 0:
            logger.error("Generated network has no edges")
            raise ValueError("Generated network has no edges. Try increasing num_nodes or area_size.")
        
        logger.info(f"Network created: {len(self.G.edges())} edges, connectivity={nx.is_connected(self.G)}")
        
        # Network structure
        self.edge_list = list(self.G.edges())
        self.adj_matrix = nx.to_numpy_array(self.G)
        
        # Calculate network statistics
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        x_min, x_max = np.min(self.node_coords[:, 0]), np.max(self.node_coords[:, 0])
        y_min, y_max = np.min(self.node_coords[:, 1]), np.max(self.node_coords[:, 1])
        logger.info(f"Network topology: avg_degree={np.mean(degrees):.2f}, density={nx.density(self.G):.4f}")
        logger.info(f"Node coordinate ranges: X=[{x_min:.2f}, {x_max:.2f}]mm, Y=[{y_min:.2f}, {y_max:.2f}]mm")
        
        # FHN Parameters (Pleurotus ostreatus dynamics)
        self.tau_v = 50.0  # Voltage time constant (ms)
        self.tau_w = 800.0  # Recovery variable time constant (ms)
        self.a = 0.7  # FHN parameter
        self.b = 0.8  # FHN parameter
        self.v_scale = 5.0  # Voltage scaling factor
        logger.debug(f"FHN parameters: tau_v={self.tau_v}ms, tau_w={self.tau_w}ms, a={self.a}, b={self.b}")

        # Memristor Parameters
        self.R_off = 100.0  # High resistance state (Ohms)
        self.R_on = 10.0  # Low resistance state (Ohms)
        self.alpha = 0.01  # Memristor adaptation rate
        logger.debug(f"Memristor parameters: R_off={self.R_off}Ω, R_on={self.R_on}Ω, alpha={self.alpha}")

        # Substrate Physics
        self.coupling_radius = 2.0  # mm (Effective range of electrode field)
        logger.debug(f"Coupling radius: {self.coupling_radius}mm")
        
        init_time = time.time() - start_time
        logger.info(f"Initialization complete in {init_time:.3f}s")

    def system_derivatives(self, state: np.ndarray, t: float, 
                          stim_currents: Callable[[float], np.ndarray]) -> np.ndarray:
        """Compute derivatives for the coupled FHN-memristor system.
        
        Args:
            state: Current state vector [V, W, M] where:
                   V: node voltages (length n)
                   W: recovery variables (length n)
                   M: memristor states (length e)
            t: Current time (ms)
            stim_currents: Function that returns stimulation currents at time t
            
        Returns:
            Derivative vector [dV/dt, dW/dt, dM/dt]
        """
        n = self.num_nodes
        e = len(self.edge_list)
        
        # Unpack state variables
        V = state[0:n]
        W = state[n:2*n]
        M = state[2*n:2*n+e]
        
        # Edge Conductance (memristor-dependent)
        conductances = 1.0 / (self.R_on * M + self.R_off * (1.0 - M))
        
        # Laplacian Diffusion (electrical coupling between nodes)
        I_coupling = np.zeros(n)
        for k, (u, v) in enumerate(self.edge_list):
            g = conductances[k]
            current = g * (V[v] - V[u])
            I_coupling[u] += current
            I_coupling[v] -= current 
            
        # Memristor State Update (activity-dependent plasticity)
        dM_dt = np.zeros(e)
        for k, (u, v) in enumerate(self.edge_list):
            flux = V[u] - V[v]
            # Memristor changes based on voltage flux and current state
            dM_dt[k] = self.alpha * np.abs(flux) * (M[k] * (1.0 - M[k]) + 0.05)

        # FHN Dynamics (excitable dynamics at each node)
        I_stim = stim_currents(t)
        
        dV_dt = (V - (V**3)/3.0 - W + I_stim + I_coupling) / self.tau_v
        dW_dt = (V + self.a - self.b * W) / self.tau_w
        
        return np.concatenate([dV_dt, dW_dt, dM_dt])

    def calculate_stimulation_coupling(self, electrode_pos: Tuple[float, float], 
                                      voltage_mag: float) -> np.ndarray:
        """Calculate the effective current entering each node from an electrode.
        
        Uses a Lorentzian decay function to model volume conduction through the substrate.
        
        Args:
            electrode_pos: (x, y) position of the electrode in mm
            voltage_mag: Voltage magnitude applied to the electrode
            
        Returns:
            Array of effective currents for each node
        """
        # Validate electrode position
        if not (0 <= electrode_pos[0] <= self.area_size and 0 <= electrode_pos[1] <= self.area_size):
            logger.warning(f"Electrode position {electrode_pos} is outside simulation area [0, {self.area_size}]")
            warnings.warn(f"Electrode position {electrode_pos} is outside simulation area [0, {self.area_size}]")
        
        # Calculate distances from this electrode to ALL nodes
        dists = cdist([electrode_pos], self.node_coords)[0]
        
        # Lorentzian decay function to simulate volume conduction
        # I_i = V_electrode * (1 / (1 + (dist/r)^2))
        coupling_factors = 1.0 / (1.0 + (dists / self.coupling_radius)**2)
        
        # If distance is too far (> 3*radius), coupling is effectively zero (noise floor)
        coupling_factors[dists > (3 * self.coupling_radius)] = 0.0
        
        return coupling_factors * voltage_mag

    def read_output_voltage(self, probe_pos: Tuple[float, float], 
                           voltages: np.ndarray) -> float:
        """Read output voltage using Lorentzian-weighted average of nearby nodes.
        
        Uses the same Lorentzian decay function as electrode coupling to model
        volume conduction from the fungal network to the measurement probe.
        
        Args:
            probe_pos: (x, y) position of the output probe in mm
            voltages: Array of node voltages at a given time point
            
        Returns:
            Weighted voltage measurement at the probe location
        """
        # Validate probe position
        if not (0 <= probe_pos[0] <= self.area_size and 0 <= probe_pos[1] <= self.area_size):
            logger.warning(f"Probe position {probe_pos} is outside simulation area [0, {self.area_size}]")
        
        # Calculate distances from probe to ALL nodes
        dists = cdist([probe_pos], self.node_coords)[0]
        
        # Lorentzian decay function (same as stimulation coupling)
        coupling_factors = 1.0 / (1.0 + (dists / self.coupling_radius)**2)
        
        # If distance is too far (> 3*radius), coupling is effectively zero
        coupling_factors[dists > (3 * self.coupling_radius)] = 0.0
        
        # Normalize weights and compute weighted average
        total_coupling = np.sum(coupling_factors)
        if total_coupling > 0:
            weights = coupling_factors / total_coupling
            return np.dot(weights, voltages)
        else:
            logger.warning(f"Probe at {probe_pos} has no coupling to network (too far from all nodes)")
            return 0.0

    def run_experiment(self, electrodes: List[Tuple[float, float]], 
                      voltage: float = 2.0, 
                      pulse_duration: float = 100.0, 
                      sim_time: float = 4000.0,
                      time_step: float = DEFAULT_TIME_STEP,
                      pulse_start: float = DEFAULT_PULSE_START,
                      electrode_delays: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run a stimulation experiment on the fungal network.
        
        Args:
            electrodes: List of (x, y) tuples for active electrode positions
            voltage: Voltage magnitude applied to electrodes (V)
            pulse_duration: Duration of the stimulation pulse (ms)
            sim_time: Total simulation time (ms)
            time_step: Time step for numerical integration (ms)
            pulse_start: Time when pulse begins (ms)
            electrode_delays: Optional list of delays (ms) for each electrode. If provided,
                            electrode i starts at pulse_start + electrode_delays[i]
            
        Returns:
            Tuple of (time_array, voltage_matrix) where voltage_matrix has shape (time_steps, num_nodes)
        """
        logger.info(f"Starting experiment: {len(electrodes)} electrodes, V={voltage}V, duration={pulse_duration}ms")
        logger.debug(f"Electrode positions: {electrodes}")
        logger.debug(f"Simulation parameters: sim_time={sim_time}ms, time_step={time_step}ms, pulse_start={pulse_start}ms")
        
        start_time = time.time()
        t = np.linspace(0, sim_time, int(sim_time / time_step))
        logger.debug(f"Time array: {len(t)} points")
        
        # 1. Pre-calculate the spatial coupling map for each electrode
        # If no delays provided, default to zero delay for all electrodes
        if electrode_delays is None:
            electrode_delays = [0.0] * len(electrodes)
        
        if len(electrode_delays) != len(electrodes):
            raise ValueError(f"electrode_delays length ({len(electrode_delays)}) must match electrodes length ({len(electrodes)})")
        
        # Store individual coupling maps for each electrode
        individual_coupling_maps = []
        for i, pos in enumerate(electrodes):
            logger.debug(f"Calculating coupling for electrode {i+1}/{len(electrodes)} at {pos} with delay={electrode_delays[i]:.2f}ms")
            
            # Check if electrode is near any network nodes
            min_dist_to_network = np.min(cdist([pos], self.node_coords)[0])
            if min_dist_to_network > 3 * self.coupling_radius:
                logger.warning(f"Electrode {i+1} at {pos} is {min_dist_to_network:.2f}mm from nearest node (>3x coupling radius)")
            
            coupling = self.calculate_stimulation_coupling(pos, voltage)
            individual_coupling_maps.append(coupling)
        
        if len(electrodes) > 0:
            max_coupling = max(np.max(cm) for cm in individual_coupling_maps)
            logger.debug(f"Max coupling strength: {max_coupling:.4f}")
            
        # 2. Define Time-Dependent Stimulation Protocol with individual delays
        def stim_protocol(time: float) -> np.ndarray:
            """Rectangular pulse with individual delays for each electrode."""
            total_current = np.zeros(self.num_nodes)
            for i, (coupling_map, delay) in enumerate(zip(individual_coupling_maps, electrode_delays)):
                start_time = pulse_start + delay
                end_time = start_time + pulse_duration
                if start_time < time < end_time:
                    total_current += coupling_map
            return total_current

        # 3. Set Initial Conditions
        n = self.num_nodes
        state_0 = np.concatenate([
            DEFAULT_INITIAL_V * np.ones(n),  # Initial voltages
            DEFAULT_INITIAL_W * np.ones(n),  # Initial recovery variables
            DEFAULT_INITIAL_M * np.ones(len(self.edge_list))  # Initial memristor states
        ])
        
        # 4. Solve ODE System
        logger.debug("Starting ODE integration...")
        ode_start = time.time()
        solution = odeint(self.system_derivatives, state_0, t, args=(stim_protocol,))
        ode_time = time.time() - ode_start
        logger.debug(f"ODE integration complete in {ode_time:.3f}s")
        
        # Log solution statistics
        voltages = solution[:, :n]
        v_min, v_max = np.min(voltages), np.max(voltages)
        logger.info(f"Experiment complete: V_range=[{v_min:.3f}, {v_max:.3f}], total_time={time.time()-start_time:.3f}s")
        
        # Return time array and voltage traces for all nodes
        return t, solution[:, :n]

# ==========================================
# Bayesian Optimization for XOR Gate
# ==========================================

def create_xor_objective(env: RealisticFungalComputer) -> Callable:
    """Create an objective function for optimizing XOR gate performance.
    
    Args:
        env: The fungal computer environment to optimize
        
    Returns:
        Objective function for Bayesian optimization
    """
    
    @use_named_args([
        Real(0.0, env.area_size, name='x_A'),
        Real(0.0, env.area_size, name='y_A'),
        Real(0.0, env.area_size, name='x_B'),
        Real(0.0, env.area_size, name='y_B'),
        Real(0.0, env.area_size, name='x_out'),
        Real(0.0, env.area_size, name='y_out'),
        Real(1.0, 5.0, name='voltage'),
        Real(50.0, 300.0, name='duration'),
        Real(-200.0, 200.0, name='delay')
    ])
    def objective_function(x_A: float, y_A: float, x_B: float, y_B: float, 
                          x_out: float, y_out: float, voltage: float, duration: float, delay: float) -> float:
        """Evaluate XOR gate performance for given electrode configuration.
        
        Args:
            x_A, y_A: Position of electrode A
            x_B, y_B: Position of electrode B
            x_out, y_out: Position of output probe
            voltage: Stimulation voltage
            duration: Pulse duration
            delay: Time delay (ms) between electrode A and B for the (1,1) case
            
        Returns:
            Negative score (for minimization)
        """
        logger.debug(f"Evaluating configuration: A=({x_A:.2f},{y_A:.2f}), B=({x_B:.2f},{y_B:.2f}), Out=({x_out:.2f},{y_out:.2f})")
        eval_start = time.time()
        
        # Define electrode and probe locations
        loc_A = (x_A, y_A)
        loc_B = (x_B, y_B)
        loc_out = (x_out, y_out)
        
        # Simulate all 4 XOR input cases
        logger.debug("Running XOR case (1,0)...")
        # Case (1, 0): Only A active
        _, sol_10 = env.run_experiment([loc_A], voltage, duration)
        v_10 = np.max([env.read_output_voltage(loc_out, sol_10[i, :]) for i in range(len(sol_10))])
        logger.debug(f"Case (1,0): max_V={v_10:.4f}")
        
        logger.debug("Running XOR case (0,1)...")
        # Case (0, 1): Only B active
        _, sol_01 = env.run_experiment([loc_B], voltage, duration)
        v_01 = np.max([env.read_output_voltage(loc_out, sol_01[i, :]) for i in range(len(sol_01))])
        logger.debug(f"Case (0,1): max_V={v_01:.4f}")
        
        logger.debug(f"Running XOR case (1,1) with delay={delay:.2f}ms...")
        # Case (1, 1): Both A and B active with delay between them
        # Electrode A starts at default time (0 delay), electrode B starts at delay
        _, sol_11 = env.run_experiment([loc_A, loc_B], voltage, duration, electrode_delays=[0.0, delay])
        v_11 = np.max([env.read_output_voltage(loc_out, sol_11[i, :]) for i in range(len(sol_11))])
        logger.debug(f"Case (1,1): max_V={v_11:.4f}")
        
        logger.debug("Running XOR case (0,0)...")
        # Case (0, 0): Neither active
        _, sol_00 = env.run_experiment([], voltage, duration)
        v_00 = np.max([env.read_output_voltage(loc_out, sol_00[i, :]) for i in range(len(sol_00))])
        logger.debug(f"Case (0,0): max_V={v_00:.4f}")

        # XOR Score: maximize separation between high (10, 01) and low (00, 11) outputs
        min_high = min(v_10, v_01)
        max_low = max(v_00, v_11)
        score = min_high - max_low
        
        # Penalty if electrodes are physically too close (short circuit risk)
        dist_AB = np.linalg.norm(np.array(loc_A) - np.array(loc_B))
        penalty = 10.0
        if dist_AB < MIN_ELECTRODE_DISTANCE:
            penalty = ELECTRODE_PROXIMITY_PENALTY
            score -= penalty
            logger.debug(f"Proximity penalty applied: dist_AB={dist_AB:.2f}mm < {MIN_ELECTRODE_DISTANCE}mm")
        
        eval_time = time.time() - eval_start
        objective_value = -score  # Negative because XX_minimize minimizes
        logger.info(f"Eval complete in {eval_time:.2f}s: V={voltage:.1f}V, High={min_high:.2f}, Low={max_low:.2f}, Score={score:.4f}, Objective={objective_value:.4f}")
        print(f"Eval: V={voltage:.1f}, High={min_high:.2f}, Low={max_low:.2f} | Score: {score:.4f} | Objective (to minimize): {objective_value:.4f}")
        return objective_value
    
    return objective_function


def optimize_xor_gate(num_nodes: int = 30, n_calls: int = 15, 
                     random_state: int = 42,
                     minimizer: str = 'gp') -> dict:
    """Optimize electrode placement for XOR gate implementation.
    
    Args:
        num_nodes: Number of nodes in the fungal network
        n_calls: Number of optimization iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing optimization results and best parameters
    """
    logger.info("="*60)
    logger.info("Starting XOR Gate Optimization")
    logger.info(f"Parameters: num_nodes={num_nodes}, n_calls={n_calls}, random_state={random_state}")
    logger.info("="*60)
    
    print("Initializing Fungal Computer...")
    opt_start_time = time.time()
    env = RealisticFungalComputer(num_nodes=num_nodes, random_seed=random_state)
    
    print(f"Network created: {env.num_nodes} nodes, {len(env.edge_list)} edges")
    print("Starting Bayesian Optimization...\n")
    logger.info(f"Starting Bayesian optimization with {n_calls} iterations")
    
    objective = create_xor_objective(env)
    
    # Define search space
    space = [
        Real(0.0, env.area_size, name='x_A'),
        Real(0.0, env.area_size, name='y_A'),
        Real(0.0, env.area_size, name='x_B'),
        Real(0.0, env.area_size, name='y_B'),
        Real(0.0, env.area_size, name='x_out'),
        Real(0.0, env.area_size, name='y_out'),
        Real(0.5, 10.0, name='voltage'),
        Real(5.0, 5000.0, name='duration'),
        Real(-200.0, 200.0, name='delay')
    ]
    
    # Run optimization
    logger.info("Beginning GP minimization...")
    gp_start = time.time()
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state)
    if minimizer == 'forest':
        res = forest_minimize(objective, space, n_calls=n_calls, random_state=random_state,
                              n_initial_points=100, initial_point_generator='lhs',
                              acq_func='EI')
    elif minimizer == 'gbrt':
        res = gbrt_minimize(objective, space, n_calls=n_calls,
                            acq_func='EI', n_initial_points=100, initial_point_generator='lhs')
    gp_time = time.time() - gp_start
    logger.info(f"GP minimization complete in {gp_time:.2f}s")
    
    # Extract results
    best_params = {
        'x_A': res.x[0],
        'y_A': res.x[1],
        'x_B': res.x[2],
        'y_B': res.x[3],
        'x_out': res.x[4],
        'y_out': res.x[5],
        'voltage': res.x[6],
        'duration': res.x[7],
        'delay': res.x[8],
        'score': -res.fun
    }
    
    # Convert all objective values back to scores for analysis
    all_scores = -np.array(res.func_vals)
    
    total_time = time.time() - opt_start_time
    
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    print(f"Best Score: {best_params['score']:.4f}")
    print(f"Score Statistics:")
    print(f"  Min Score: {np.min(all_scores):.4f}")
    print(f"  Max Score: {np.max(all_scores):.4f}")
    print(f"  Mean Score: {np.mean(all_scores):.4f}")
    print(f"  Std Score: {np.std(all_scores):.4f}")
    print(f"\nBest Parameters:")
    print(f"  Electrode A: ({best_params['x_A']:.2f}, {best_params['y_A']:.2f})")
    print(f"  Electrode B: ({best_params['x_B']:.2f}, {best_params['y_B']:.2f})")
    print(f"  Output Probe: ({best_params['x_out']:.2f}, {best_params['y_out']:.2f})")
    print(f"  Voltage: {best_params['voltage']:.2f} V")
    print(f"  Duration: {best_params['duration']:.2f} ms")
    print(f"  Delay: {best_params['delay']:.2f} ms")
    print(f"{'='*60}\n")
    
    logger.info("="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Total optimization time: {total_time:.2f}s")
    logger.info(f"Best score achieved: {best_params['score']:.4f}")
    logger.info(f"Best electrode A: ({best_params['x_A']:.2f}, {best_params['y_A']:.2f})")
    logger.info(f"Best electrode B: ({best_params['x_B']:.2f}, {best_params['y_B']:.2f})")
    logger.info(f"Best output probe: ({best_params['x_out']:.2f}, {best_params['y_out']:.2f})")
    logger.info(f"Best voltage: {best_params['voltage']:.2f}V, duration: {best_params['duration']:.2f}ms, delay: {best_params['delay']:.2f}ms")
    logger.info("="*60)
    
    return {'env': env, 'result': res, 'params': best_params}


def visualize_results(optimization_results: dict) -> None:
    """Visualize the optimized XOR gate configuration (basic view).
    
    Args:
        optimization_results: Dictionary returned by optimize_xor_gate()
    """
    env = optimization_results['env']
    params = optimization_results['params']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Network topology with electrode positions
    pos = {i: env.pos[i] for i in range(env.num_nodes)}
    nx.draw(env.G, pos, ax=ax1, node_size=30, alpha=0.7, 
            with_labels=False, node_color='lightgray', edge_color='gray')
    ax1.set_title("Fungal Network with Optimized Electrodes", fontsize=12, fontweight='bold')

    # Mark electrodes and probe
    ax1.plot(params['x_A'], params['y_A'], 'ro', markersize=12, 
             label='A Electrode', markeredgecolor='darkred', markeredgewidth=1.5)
    ax1.plot(params['x_B'], params['y_B'], 'bo', markersize=12, 
             label='B Electrode', markeredgecolor='darkblue', markeredgewidth=1.5)
    ax1.plot(params['x_out'], params['y_out'], 'go', markersize=12, 
             label='Output Probe', markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax1.legend(loc='best')
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')

    # Plot 2: Output response for input case (1,0)
    loc_A = (params['x_A'], params['y_A'])
    loc_out = (params['x_out'], params['y_out'])
    
    t, V = env.run_experiment([loc_A], params['voltage'], params['duration'])
    v_out = np.array([env.read_output_voltage(loc_out, V[i, :]) for i in range(len(V))])
    
    ax2.plot(t, v_out, 'g-', linewidth=2, label='Output Voltage')
    ax2.axvspan(DEFAULT_PULSE_START, DEFAULT_PULSE_START + params['duration'], 
                alpha=0.2, color='red', label='Stimulation')
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('Voltage (V)', fontsize=11)
    ax2.set_title('Output Response (Input A=1, B=0)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def visualize_all_xor_cases(optimization_results: dict) -> None:
    """Visualize all 4 XOR input cases side by side.
    
    Args:
        optimization_results: Dictionary returned by optimize_xor_gate()
    """
    env = optimization_results['env']
    params = optimization_results['params']
    
    loc_A = (params['x_A'], params['y_A'])
    loc_B = (params['x_B'], params['y_B'])
    loc_out = (params['x_out'], params['y_out'])
    
    # Run all 4 cases
    cases = [
        ([], None, '(0,0)', 'gray'),
        ([loc_A], None, '(1,0)', 'red'),
        ([loc_B], None, '(0,1)', 'blue'),
        ([loc_A, loc_B], [0.0, params['delay']], '(1,1)', 'purple')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    max_voltages = []
    
    for idx, (electrodes, delays, label, color) in enumerate(cases):
        t, V = env.run_experiment(electrodes, params['voltage'], params['duration'], electrode_delays=delays)
        v_out = np.array([env.read_output_voltage(loc_out, V[i, :]) for i in range(len(V))])
        max_v = np.max(v_out)
        max_voltages.append(max_v)
        
        axes[idx].plot(t, v_out, color=color, linewidth=2.5)
        axes[idx].axvspan(DEFAULT_PULSE_START, DEFAULT_PULSE_START + params['duration'], 
                         alpha=0.15, color='red')
        axes[idx].set_xlabel('Time (ms)', fontsize=10)
        axes[idx].set_ylabel('Output Voltage (V)', fontsize=10)
        axes[idx].set_title(f'Input {label} | Max: {max_v:.3f} V', 
                           fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Add overall title with XOR performance metrics
    min_high = min(max_voltages[1], max_voltages[2])
    max_low = max(max_voltages[0], max_voltages[3])
    separation = min_high - max_low
    
    fig.suptitle(f'XOR Gate Performance | Separation: {separation:.3f} V | '
                f'High: {min_high:.3f} V | Low: {max_low:.3f} V', 
                fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()


def visualize_optimization_convergence(optimization_results: dict) -> None:
    """Plot the optimization convergence over iterations.
    
    Args:
        optimization_results: Dictionary returned by optimize_xor_gate()
    """
    res = optimization_results['result']
    
    # Extract function values (negative scores)
    func_vals = res.func_vals
    scores = -func_vals  # Convert back to positive scores
    
    # Calculate running best
    running_best = np.maximum.accumulate(scores)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Score per iteration
    ax1.plot(scores, 'o-', color='steelblue', linewidth=2, markersize=6, label='Score')
    ax1.plot(running_best, '--', color='darkred', linewidth=2.5, label='Best So Far')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('XOR Score', fontsize=11)
    ax1.set_title('Optimization Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Parameter evolution for key parameters
    x_iters = res.x_iters
    voltages = [x[6] for x in x_iters]
    durations = [x[7] for x in x_iters]
    
    ax2_twin = ax2.twinx()
    ax2.plot(voltages, 'o-', color='orange', linewidth=2, markersize=6, label='Voltage')
    ax2_twin.plot(durations, 's-', color='green', linewidth=2, markersize=6, label='Duration')
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Voltage (V)', fontsize=11, color='orange')
    ax2_twin.set_ylabel('Duration (ms)', fontsize=11, color='green')
    ax2.set_title('Parameter Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.show()


def visualize_spatial_coupling(optimization_results: dict, resolution: int = 100) -> None:
    """Create a heatmap showing electrode coupling strength across the substrate.
    
    Args:
        optimization_results: Dictionary returned by optimize_xor_gate()
        resolution: Grid resolution for the heatmap
    """
    env = optimization_results['env']
    params = optimization_results['params']
    
    # Create grid
    x = np.linspace(0, env.area_size, resolution)
    y = np.linspace(0, env.area_size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate coupling for electrode A
    coupling_A = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            pos = (X[i, j], Y[i, j])
            coupling_A[i, j] = np.max(env.calculate_stimulation_coupling(pos, params['voltage']))
    
    # Calculate coupling for electrode B
    coupling_B = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            pos = (X[i, j], Y[i, j])
            coupling_B[i, j] = np.max(env.calculate_stimulation_coupling(pos, params['voltage']))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot A coupling
    im1 = axes[0].contourf(X, Y, coupling_A, levels=20, cmap='Reds')
    axes[0].plot(params['x_A'], params['y_A'], 'r*', markersize=20, 
                markeredgecolor='white', markeredgewidth=2)
    axes[0].scatter(env.node_coords[:, 0], env.node_coords[:, 1], 
                   c='white', s=10, alpha=0.5, edgecolors='black', linewidths=0.5)
    axes[0].set_title('Electrode A Coupling Field', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X Position (mm)')
    axes[0].set_ylabel('Y Position (mm)')
    plt.colorbar(im1, ax=axes[0], label='Coupling Strength')
    
    # Plot B coupling
    im2 = axes[1].contourf(X, Y, coupling_B, levels=20, cmap='Blues')
    axes[1].plot(params['x_B'], params['y_B'], 'b*', markersize=20, 
                markeredgecolor='white', markeredgewidth=2)
    axes[1].scatter(env.node_coords[:, 0], env.node_coords[:, 1], 
                   c='white', s=10, alpha=0.5, edgecolors='black', linewidths=0.5)
    axes[1].set_title('Electrode B Coupling Field', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X Position (mm)')
    axes[1].set_ylabel('Y Position (mm)')
    plt.colorbar(im2, ax=axes[1], label='Coupling Strength')
    
    # Plot combined (interference pattern)
    coupling_combined = coupling_A + coupling_B
    im3 = axes[2].contourf(X, Y, coupling_combined, levels=20, cmap='viridis')
    axes[2].plot(params['x_A'], params['y_A'], 'r*', markersize=15, 
                markeredgecolor='white', markeredgewidth=2, label='Electrode A')
    axes[2].plot(params['x_B'], params['y_B'], 'b*', markersize=15, 
                markeredgecolor='white', markeredgewidth=2, label='Electrode B')
    axes[2].plot(params['x_out'], params['y_out'], 'g*', markersize=15, 
                markeredgecolor='white', markeredgewidth=2, label='Output')
    axes[2].scatter(env.node_coords[:, 0], env.node_coords[:, 1], 
                   c='white', s=10, alpha=0.5, edgecolors='black', linewidths=0.5)
    axes[2].set_title('Combined Coupling Field', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X Position (mm)')
    axes[2].set_ylabel('Y Position (mm)')
    axes[2].legend(loc='best')
    plt.colorbar(im3, ax=axes[2], label='Coupling Strength')
    
    plt.tight_layout()
    plt.show()


def visualize_memristor_evolution(optimization_results: dict) -> None:
    """Visualize how memristor states evolve during stimulation.
    
    Args:
        optimization_results: Dictionary returned by optimize_xor_gate()
    """
    env = optimization_results['env']
    params = optimization_results['params']
    
    loc_A = (params['x_A'], params['y_A'])
    loc_B = (params['x_B'], params['y_B'])
    
    # Run experiment and capture full state
    t = np.linspace(0, 2000.0, int(2000.0 / DEFAULT_TIME_STEP))
    
    coupling_map = np.zeros(env.num_nodes)
    for pos in [loc_A, loc_B]:
        coupling_map += env.calculate_stimulation_coupling(pos, params['voltage'])
    
    def stim_protocol(time: float) -> np.ndarray:
        if DEFAULT_PULSE_START < time < (DEFAULT_PULSE_START + params['duration']):
            return coupling_map
        return np.zeros(env.num_nodes)
    
    n = env.num_nodes
    state_0 = np.concatenate([
        DEFAULT_INITIAL_V * np.ones(n),
        DEFAULT_INITIAL_W * np.ones(n),
        DEFAULT_INITIAL_M * np.ones(len(env.edge_list))
    ])
    
    solution = odeint(env.system_derivatives, state_0, t, args=(stim_protocol,))
    
    # Extract memristor states
    M_states = solution[:, 2*n:]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Individual memristor traces (sample)
    num_to_plot = min(10, len(env.edge_list))
    for i in range(num_to_plot):
        ax1.plot(t, M_states[:, i], alpha=0.7, linewidth=1.5)
    
    ax1.axvspan(DEFAULT_PULSE_START, DEFAULT_PULSE_START + params['duration'], 
               alpha=0.15, color='red', label='Stimulation')
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Memristor State', fontsize=11)
    ax1.set_title(f'Memristor Evolution (showing {num_to_plot}/{len(env.edge_list)} edges)', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Average and distribution over time
    M_mean = np.mean(M_states, axis=1)
    M_std = np.std(M_states, axis=1)
    
    ax2.plot(t, M_mean, 'b-', linewidth=2.5, label='Mean')
    ax2.fill_between(t, M_mean - M_std, M_mean + M_std, alpha=0.3, color='blue', label='±1 Std')
    ax2.axvspan(DEFAULT_PULSE_START, DEFAULT_PULSE_START + params['duration'], 
               alpha=0.15, color='red', label='Stimulation')
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('Memristor State', fontsize=11)
    ax2.set_title('Average Memristor State Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def visualize_network_metrics(optimization_results: dict) -> None:
    """Display network topology metrics and statistics.
    
    Args:
        optimization_results: Dictionary returned by optimize_xor_gate()
    """
    env = optimization_results['env']
    G = env.G
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Degree distribution
    degrees = [G.degree(n) for n in G.nodes()]
    axes[0, 0].hist(degrees, bins=range(min(degrees), max(degrees)+2), 
                    color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Node Degree', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title(f'Degree Distribution | Mean: {np.mean(degrees):.2f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Edge length distribution
    edge_lengths = []
    for u, v in G.edges():
        pos_u = np.array(env.pos[u])
        pos_v = np.array(env.pos[v])
        edge_lengths.append(np.linalg.norm(pos_u - pos_v))
    
    axes[0, 1].hist(edge_lengths, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Edge Length (mm)', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title(f'Edge Length Distribution | Mean: {np.mean(edge_lengths):.2f} mm', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Network with degree-based coloring
    pos = {i: env.pos[i] for i in range(env.num_nodes)}
    node_colors = [G.degree(n) for n in G.nodes()]
    
    nx.draw(G, pos, ax=axes[1, 0], node_size=100, alpha=0.8, 
            node_color=node_colors, cmap='YlOrRd', with_labels=False,
            edge_color='gray', width=0.5)
    axes[1, 0].set_title('Network Colored by Degree', fontsize=12, fontweight='bold')
    
    # Plot 4: Network statistics table
    axes[1, 1].axis('off')
    
    # Calculate statistics
    stats = [
        ['Metric', 'Value'],
        ['─' * 30, '─' * 15],
        ['Nodes', f'{G.number_of_nodes()}'],
        ['Edges', f'{G.number_of_edges()}'],
        ['Avg Degree', f'{np.mean(degrees):.2f}'],
        ['Density', f'{nx.density(G):.4f}'],
        ['Avg Clustering', f'{nx.average_clustering(G):.4f}'],
        ['Connected', f'{nx.is_connected(G)}'],
    ]
    
    if nx.is_connected(G):
        stats.append(['Diameter', f'{nx.diameter(G)}'])
        stats.append(['Avg Path Length', f'{nx.average_shortest_path_length(G):.2f}'])
    
    table = axes[1, 1].table(cellText=stats, cellLoc='left', loc='center',
                            colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Network Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


def visualize_comprehensive(optimization_results: dict) -> None:
    """Create a comprehensive multi-panel visualization with all key information.
    
    Args:
        optimization_results: Dictionary returned by optimize_xor_gate()
    """
    env = optimization_results['env']
    params = optimization_results['params']
    res = optimization_results['result']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Network topology (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    pos = {i: env.pos[i] for i in range(env.num_nodes)}
    nx.draw(env.G, pos, ax=ax1, node_size=25, alpha=0.6, 
            with_labels=False, node_color='lightgray', edge_color='gray', width=0.5)
    ax1.plot(params['x_A'], params['y_A'], 'ro', markersize=10, label='A')
    ax1.plot(params['x_B'], params['y_B'], 'bo', markersize=10, label='B')
    ax1.plot(params['x_out'], params['y_out'], 'go', markersize=10, label='Out')
    ax1.set_title('Network Topology', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    
    # 2. Optimization convergence (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    scores = -res.func_vals
    running_best = np.maximum.accumulate(scores)
    ax2.plot(scores, 'o-', color='steelblue', linewidth=1.5, markersize=4)
    ax2.plot(running_best, '--', color='darkred', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=9)
    ax2.set_ylabel('Score', fontsize=9)
    ax2.set_title('Optimization Convergence', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. XOR truth table (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    loc_A = (params['x_A'], params['y_A'])
    loc_B = (params['x_B'], params['y_B'])
    loc_out = (params['x_out'], params['y_out'])
    
    cases = [
        ([], '0', '0'),
        ([loc_A], '1', '0'),
        ([loc_B], '0', '1'),
        ([loc_A, loc_B], '1', '1')
    ]
    
    truth_data = [['A', 'B', 'Output (V)', 'Expected']]
    for electrodes, a, b in cases:
        _, V = env.run_experiment(electrodes, params['voltage'], params['duration'])
        v_max = np.max([env.read_output_voltage(loc_out, V[i, :]) for i in range(len(V))])
        expected = '1' if (a != b) else '0'
        truth_data.append([a, b, f'{v_max:.3f}', expected])
    
    table = ax3.table(cellText=truth_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('XOR Truth Table', fontsize=11, fontweight='bold', pad=20)
    
    # 4-7. All four XOR cases (middle row)
    for idx, (electrodes, a, b) in enumerate(cases):
        ax = fig.add_subplot(gs[1, idx if idx < 3 else 2])
        if idx == 3:
            ax = fig.add_subplot(gs[1, 2])
        
        t, V = env.run_experiment(electrodes, params['voltage'], params['duration'])
        v_out = np.array([env.read_output_voltage(loc_out, V[i, :]) for i in range(len(V))])
        color = ['gray', 'red', 'blue', 'purple'][idx]
        
        ax.plot(t, v_out, color=color, linewidth=1.5)
        ax.axvspan(DEFAULT_PULSE_START, DEFAULT_PULSE_START + params['duration'], 
                  alpha=0.1, color='red')
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_ylabel('V (V)', fontsize=9)
        ax.set_title(f'Input ({a},{b})', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 8. Memristor evolution (bottom left)
    ax8 = fig.add_subplot(gs[2, 0])
    
    coupling_map = np.zeros(env.num_nodes)
    for pos in [loc_A, loc_B]:
        coupling_map += env.calculate_stimulation_coupling(pos, params['voltage'])
    
    def stim_protocol(time: float) -> np.ndarray:
        if DEFAULT_PULSE_START < time < (DEFAULT_PULSE_START + params['duration']):
            return coupling_map
        return np.zeros(env.num_nodes)
    
    t = np.linspace(0, 2000.0, int(2000.0 / DEFAULT_TIME_STEP))
    n = env.num_nodes
    state_0 = np.concatenate([
        DEFAULT_INITIAL_V * np.ones(n),
        DEFAULT_INITIAL_W * np.ones(n),
        DEFAULT_INITIAL_M * np.ones(len(env.edge_list))
    ])
    solution = odeint(env.system_derivatives, state_0, t, args=(stim_protocol,))
    M_states = solution[:, 2*n:]
    M_mean = np.mean(M_states, axis=1)
    M_std = np.std(M_states, axis=1)
    
    ax8.plot(t, M_mean, 'b-', linewidth=2)
    ax8.fill_between(t, M_mean - M_std, M_mean + M_std, alpha=0.2, color='blue')
    ax8.set_xlabel('Time (ms)', fontsize=9)
    ax8.set_ylabel('Memristor State', fontsize=9)
    ax8.set_title('Avg Memristor Evolution', fontsize=10, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 9. Parameter evolution (bottom middle)
    ax9 = fig.add_subplot(gs[2, 1])
    x_iters = res.x_iters
    voltages = [x[6] for x in x_iters]
    durations = [x[7] for x in x_iters]
    
    ax9_twin = ax9.twinx()
    ax9.plot(voltages, 'o-', color='orange', linewidth=1.5, markersize=4)
    ax9_twin.plot(durations, 's-', color='green', linewidth=1.5, markersize=4)
    ax9.set_xlabel('Iteration', fontsize=9)
    ax9.set_ylabel('Voltage (V)', fontsize=9, color='orange')
    ax9_twin.set_ylabel('Duration (ms)', fontsize=9, color='green')
    ax9.set_title('Parameter Evolution', fontsize=10, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    # 10. Network statistics (bottom right)
    ax10 = fig.add_subplot(gs[2, 2])
    ax10.axis('off')
    
    degrees = [env.G.degree(n) for n in env.G.nodes()]
    stats_data = [
        ['Metric', 'Value'],
        ['Nodes', f'{env.G.number_of_nodes()}'],
        ['Edges', f'{env.G.number_of_edges()}'],
        ['Avg Degree', f'{np.mean(degrees):.2f}'],
        ['Best Score', f'{params["score"]:.4f}'],
        ['Voltage', f'{params["voltage"]:.2f} V'],
        ['Duration', f'{params["duration"]:.1f} ms'],
    ]
    
    table2 = ax10.table(cellText=stats_data, cellLoc='left', loc='center',
                       colWidths=[0.6, 0.4])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2)
    
    for i in range(2):
        table2[(0, i)].set_facecolor('#4CAF50')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    ax10.set_title('Summary Statistics', fontsize=10, fontweight='bold', pad=20)
    
    fig.suptitle('Comprehensive Fungal Computer Analysis', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.show()


if __name__ == "__main__":
    logger.info("Starting fungal computer simulation main program")
    logger.info(f"Logging level: {logging.getLevelName(logger.level)}")
    
    # Run optimization
    results = optimize_xor_gate(num_nodes=40, n_calls=200, random_state=42, minimizer='forest')
    
    # Choose your visualization(s):
    
    # Option 1: Basic view (network + single response)
    # visualize_results(results)
    
    # Option 2: All XOR cases comparison
    # visualize_all_xor_cases(results)
    
    # Option 3: Optimization convergence analysis
    # visualize_optimization_convergence(results)
    
    # Option 4: Spatial coupling heatmaps
    # visualize_spatial_coupling(results, resolution=100)
    
    # Option 5: Memristor state evolution
    # visualize_memristor_evolution(results)
    
    # Option 6: Network topology metrics
    # visualize_network_metrics(results)
    
    # Option 7: Comprehensive dashboard (all-in-one)
    visualize_comprehensive(results)
    
    # You can also call multiple visualizations:
    # visualize_all_xor_cases(results)
    # visualize_optimization_convergence(results)
    # visualize_spatial_coupling(results)
