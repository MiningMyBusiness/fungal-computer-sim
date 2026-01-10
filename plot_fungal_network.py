"""Script to instantiate and visualize a RealisticFungalComputer network."""

import matplotlib.pyplot as plt
import networkx as nx
from realistic_sim import RealisticFungalComputer

def plot_fungal_network(fungal_computer, figsize=(10, 10), node_size=300, 
                       edge_width=1.5, save_path=None):
    """Plot the fungal network graph in black and white.
    
    Args:
        fungal_computer: RealisticFungalComputer instance
        figsize: Figure size (width, height)
        node_size: Size of nodes in the plot
        edge_width: Width of edges in the plot
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges in black
    nx.draw_networkx_edges(
        fungal_computer.G,
        fungal_computer.pos,
        width=edge_width,
        edge_color='black',
        alpha=0.6,
        ax=ax
    )
    
    # Draw nodes in black with white fill
    nx.draw_networkx_nodes(
        fungal_computer.G,
        fungal_computer.pos,
        node_size=node_size,
        node_color='white',
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        fungal_computer.G,
        fungal_computer.pos,
        font_size=8,
        font_color='black',
        ax=ax
    )
    
    ax.set_aspect('equal')
    ax.set_xlim(-1, fungal_computer.area_size + 1)
    ax.set_ylim(-1, fungal_computer.area_size + 1)
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title('Fungal Network Graph', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    # Add network statistics to the plot
    num_nodes = fungal_computer.G.number_of_nodes()
    num_edges = fungal_computer.G.number_of_edges()
    is_connected = nx.is_connected(fungal_computer.G)
    avg_degree = sum(dict(fungal_computer.G.degree()).values()) / num_nodes
    
    stats_text = f'Nodes: {num_nodes}\nEdges: {num_edges}\nAvg Degree: {avg_degree:.2f}\nConnected: {is_connected}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Instantiate the RealisticFungalComputer
    print("Creating RealisticFungalComputer instance...")
    fungal_computer = RealisticFungalComputer(
        num_nodes=40,
        area_size=20.0,
        random_seed=42
    )
    
    print(f"\nNetwork created successfully!")
    print(f"Number of nodes: {fungal_computer.G.number_of_nodes()}")
    print(f"Number of edges: {fungal_computer.G.number_of_edges()}")
    print(f"Network connected: {nx.is_connected(fungal_computer.G)}")
    
    # Plot the network
    print("\nPlotting fungal network...")
    plot_fungal_network(
        fungal_computer,
        figsize=(12, 12),
        node_size=400,
        edge_width=2.0,
        save_path='fungal_network_plot.png'
    )
