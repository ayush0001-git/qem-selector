"""Production scaling blueprint: GNN Graph extraction and GPU simulator config.

This script demonstrates two advanced production features:
1. Extracting the graph structure (nodes + edges) of a circuit for Graph Neural Networks (GNN).
2. Configuring Qiskit's AerSimulator for GPU acceleration to scale up simulations to 10,000+ configurations.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qemsel.features import convert_circuit_to_graph


def main():
    print("--- Production Scale Up Blueprint ---")

    # =========================================================================
    # 1. GRAPH NEURAL NETWORK (GNN) INPUT PREPARATION
    # =========================================================================
    print("\n[1] Preparing Quantum Circuit as a Graph for GNN...")

    # Create a simple circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Convert to graph representation
    graph = convert_circuit_to_graph(qc)

    print("\nExtracted Graph Nodes (Gates):")
    for node in graph["nodes"]:
        print(f"  Node ID {node['id']}: Gate '{node['op']}' acting on Qubits {node['qargs']}")

    print("\nExtracted Graph Edges (Data Flow Adjacency):")
    for edge in graph["edge_index"]:
        print(f"  Edge: Gate {edge[0]} ---> Gate {edge[1]} (data flows from gate {edge[0]} to {edge[1]})")

    print("\nBlueprint for PyTorch Geometric (GNN Model Input):")
    print("""
    # PyTorch Geometric code template to load this circuit graph:
    # -----------------------------------------------------------
    # import torch
    # from torch_geometric.data import Data
    #
    # # Node Features (e.g. mapping op names like 'h' -> 0, 'cx' -> 1)
    # op_mapping = {'h': 0, 'cx': 1}
    # x = torch.tensor([[op_mapping[node['op']]] for node in graph['nodes']], dtype=torch.float)
    #
    # # Edge Index (adjacency matrix)
    # edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).t().contiguous()
    #
    # # Create GNN Data Object
    # data = Data(x=x, edge_index=edge_index)
    """)

    # =========================================================================
    # 2. GPU ACCELERATION FOR 10,000+ SIMULATIONS
    # =========================================================================
    print("\n[2] Setting up GPU Simulator for Large Scale Runs (10,000+)...")

    try:
        # Create an Aer simulator configured to run on NVIDIA GPUs
        sim_gpu = AerSimulator(method="statevector", device="GPU")
        print("  - AerSimulator successfully initialized with GPU device support!")
        print("  - Configuration: device='GPU', method='statevector'")
    except Exception:
        print("  - Note: GPU simulator initialization skipped (No CUDA/NVIDIA GPU detected on this local machine).")
        print("  - To run 10,000+ configurations in production, deploy this code to a GPU-enabled cloud instance.")

    print("\nScale up tips for GPU clusters:")
    print("  - Use multi-processing or Ray/Dask to distribute different circuits across multiple GPUs.")
    print("  - Enable cuStateVec/CuPy in Qiskit Aer for accelerated tensor-network simulations.")

if __name__ == "__main__":
    main()
