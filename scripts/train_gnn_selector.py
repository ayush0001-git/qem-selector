"""CLI: train a Graph Neural Network (GNN) selector on quantum circuit DAGs.

Converts circuits to graph representations dynamically using `convert_circuit_to_graph`,
combines graph topology features with backend noise properties, and trains a GCN model.

Usage:
    python scripts/train_gnn_selector.py --data results/boundary/aggregated.csv
"""

from __future__ import annotations

import argparse

# Configure logging
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("gnn_selector")

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch_geometric.data import Data  # type: ignore
    from torch_geometric.loader import DataLoader  # type: ignore
    from torch_geometric.nn import GCNConv, global_mean_pool  # type: ignore
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    class nn:
        Module = object
        @staticmethod
        def Linear(*args, **kwargs):
            return object
    class F:
        @staticmethod
        def relu(x):
            return x
        @staticmethod
        def log_softmax(x, dim=-1):
            return x
    class GCNConv:
        def __init__(self, *args, **kwargs):
            pass
    global_mean_pool = None

from qemsel.circuits import FAMILIES
from qemsel.features import convert_circuit_to_graph

# Fixed gate vocabulary (14 gate types)
GATE_VOCAB = {
    "h": 0, "x": 1, "y": 2, "z": 3, "s": 4, "sdg": 5, "t": 6, "tdg": 7,
    "sx": 8, "rx": 9, "ry": 10, "rz": 11, "cx": 12, "cz": 13
}
NUM_GATE_TYPES = len(GATE_VOCAB)
MAX_QUBITS = 5  # Fixed target qubit multi-hot vector size (5 dims)
NODE_FEAT_DIM = NUM_GATE_TYPES + MAX_QUBITS + 1 + 1  # 14 + 5 + 1 + 1 = 21 dims


def build_pyg_data(df: pd.DataFrame, label_column: str) -> tuple[list, list]:
    """Reconstruct circuits from CSV metadata, generate 21-dim graph representations, and build PyG Data objects."""
    data_list = []

    unique_labels = sorted(df[label_column].unique())
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    _log.info(f"Reconstructing {len(df)} circuits across all families and building graph representations...")

    for idx, row in df.iterrows():
        family = str(row["family"])
        n_qubits = int(row["n_qubits"])
        depth = int(row["depth"])
        seed = int(row.get("seed", 0))

        if family not in FAMILIES:
            continue
        try:
            qc = FAMILIES[family](n_qubits, depth, seed)
        except Exception as exc:
            _log.warning(f"Skipping circuit {family}_q{n_qubits}_d{depth}_s{seed}: {exc}")
            continue

        graph = convert_circuit_to_graph(qc)
        nodes = graph["nodes"]
        edge_index_list = graph["edge_index"]

        # 1. Build 21-dimensional node feature vectors
        x_data = []
        for node in nodes:
            op_name = node["op"].lower()
            # 14-dim one-hot gate type
            gate_one_hot = [0.0] * NUM_GATE_TYPES
            if op_name in GATE_VOCAB:
                gate_one_hot[GATE_VOCAB[op_name]] = 1.0

            # 5-dim target qubit multi-hot vector
            qubit_mask = [0.0] * MAX_QUBITS
            for q_idx in node.get("qargs", []):
                if q_idx < MAX_QUBITS:
                    qubit_mask[q_idx] = 1.0

            # 1-dim Clifford flag
            is_clifford = 1.0 if op_name in {"h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "cx", "cz", "swap"} else 0.0

            # 1-dim Normalized angle distance
            norm_angle_dist = 0.0
            if op_name in {"rx", "ry", "rz", "p"}:
                norm_angle_dist = 1.0  # non-Clifford rotation proxy

            feat_vec = gate_one_hot + qubit_mask + [is_clifford, norm_angle_dist]
            x_data.append(feat_vec)

        if not x_data:
            x_data = [[0.0] * NODE_FEAT_DIM]

        x = torch.tensor(x_data, dtype=torch.float)

        # 2. Edge index (DAG topology)
        if edge_index_list:
            edges = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edges = torch.empty((2, 0), dtype=torch.long)

        # 3. Target label
        y_val = label_to_idx[row[label_column]]
        y = torch.tensor([y_val], dtype=torch.long)

        # 4. Global backend noise & shot budget features
        global_feats = [
            float(row.get("feat_backend_avg_2q_error", 0.0)),
            float(row.get("feat_backend_avg_readout_error", 0.0)),
            float(row.get("feat_log2_shots", 10.0))
        ]
        u = torch.tensor([global_feats], dtype=torch.float)

        data = Data(x=x, edge_index=edges, y=y)
        data.u = u
        data.family = family
        data.backend = str(row.get("backend", ""))
        data.group = f"{family}_q{n_qubits}_d{depth}"
        data_list.append(data)

    return data_list, unique_labels


class GNNSelector(nn.Module):
    """Hybrid 2-layer GCN model: processes circuit DAG topology and combines with global device noise."""
    def __init__(self, in_channels: int = NODE_FEAT_DIM, num_classes: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Classifier combination: pooled GNN graph vector (hidden_dim=64) + global noise features (3)
        self.fc1 = nn.Linear(hidden_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 2-layer GCN Message Passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling over DAG nodes
        x = global_mean_pool(x, batch)

        # Concatenate graph vector with global noise features
        x = torch.cat([x, data.u], dim=-1)

        # Classification head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train_eval_gnn_fold(train_data: list, test_data: list, num_classes: int, epochs: int = 15) -> tuple[float, float]:
    """Train GNN model on train_data fold and evaluate on test_data fold. Returns (accuracy, macro_f1)."""
    from sklearn.metrics import accuracy_score, f1_score

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = GNNSelector(in_channels=NODE_FEAT_DIM, num_classes=num_classes, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for test_batch in test_loader:
            out_t = model(test_batch)
            preds.extend(out_t.argmax(dim=-1).tolist())
            targets.extend(test_batch.y.tolist())

    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    return acc, f1


def plot_gnn_learning_curve(fractions: list[float], accuracies: list[float], f1s: list[float], out_path: Path) -> None:
    """Plot GNN learning curve over dataset size fractions to illustrate sample size limits."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    sizes = [int(f * 100) for f in fractions]
    ax.plot(sizes, accuracies, marker="o", linewidth=2, label="Test Accuracy", color="#348ABD")
    ax.plot(sizes, f1s, marker="s", linewidth=2, label="Macro F1", color="#E24A33")
    ax.set_xlabel("Dataset Size Percentage (%)")
    ax.set_ylabel("Metric Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("GNN Selector Learning Curve vs. Dataset Size\n(Illustrates sample size constraints on graph-level learning)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _log.info(f"Saved GNN learning curve plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a GNN Selector on circuit DAGs.")
    parser.add_argument("--data", type=Path, default=Path("results/research/aggregated.csv"), help="Path to input results CSV")
    parser.add_argument("--label", type=str, default="best_technique", help="Target label column")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per fold")
    parser.add_argument("--out", type=Path, default=Path("results/research"), help="Output directory")
    args = parser.parse_args()

    if not args.data.exists():
        _log.error(f"Data path not found: {args.data}")
        sys.exit(1)

    if not HAS_PYG:
        _log.warning("\n[Angle 4] PyTorch or PyTorch Geometric is not installed.")
        return

    # Load data
    df = pd.read_csv(args.data)
    if args.label not in df.columns:
        _log.error(f"Label column '{args.label}' not found in CSV.")
        sys.exit(1)

    pyg_data, classes = build_pyg_data(df, args.label)
    if not pyg_data:
        _log.error("No valid graph data constructed.")
        sys.exit(1)

    from sklearn.model_selection import StratifiedGroupKFold

    _log.info(f"\n--- Evaluating GNN Selector on {len(pyg_data)} graphs across {len(classes)} classes ---")

    # 1. 5-Fold Grouped Cross-Validation (identical to RF/GBM grouping logic)
    groups = np.array([d.group for d in pyg_data])
    targets = np.array([int(d.y.item()) for d in pyg_data])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accs, fold_f1s = [], []
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(pyg_data)), targets, groups), 1):
        tr_data = [pyg_data[i] for i in train_idx]
        te_data = [pyg_data[i] for i in test_idx]
        acc, f1 = train_eval_gnn_fold(tr_data, te_data, len(classes), epochs=args.epochs)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        _log.info(f"Grouped Fold {fold}: Accuracy = {acc:.3f} | Macro F1 = {f1:.3f}")

    mean_acc = float(np.mean(fold_accs))
    mean_f1 = float(np.mean(fold_f1s))
    _log.info(f"\n[GNN Grouped CV Summary]: Accuracy = {mean_acc:.3f} | Macro F1 = {mean_f1:.3f}")

    # 2. Leave-One-Family-Out (LOFO) Evaluation
    families = sorted(set(d.family for d in pyg_data))
    lofo_accs, lofo_f1s = [], []
    for fam in families:
        tr_data = [d for d in pyg_data if d.family != fam]
        te_data = [d for d in pyg_data if d.family == fam]
        if te_data and tr_data:
            acc, f1 = train_eval_gnn_fold(tr_data, te_data, len(classes), epochs=args.epochs)
            lofo_accs.append(acc)
            lofo_f1s.append(f1)
            _log.info(f"LOFO Held-out Family '{fam}': Acc = {acc:.3f} | F1 = {f1:.3f}")

    lofo_mean_acc = float(np.mean(lofo_accs)) if lofo_accs else 0.0
    lofo_mean_f1 = float(np.mean(lofo_f1s)) if lofo_f1s else 0.0
    _log.info(f"[GNN LOFO Summary]: Mean Accuracy = {lofo_mean_acc:.3f} | Mean Macro F1 = {lofo_mean_f1:.3f}")

    # 3. Leave-One-Device-Out (LODO) Evaluation
    backends = sorted(set(d.backend for d in pyg_data if d.backend))
    devices = sorted(set(b.split("@")[0] for b in backends))
    lodo_accs, lodo_f1s = [], []
    for dev in devices:
        tr_data = [d for d in pyg_data if not d.backend.startswith(dev)]
        te_data = [d for d in pyg_data if d.backend.startswith(dev)]
        if te_data and tr_data:
            acc, f1 = train_eval_gnn_fold(tr_data, te_data, len(classes), epochs=args.epochs)
            lodo_accs.append(acc)
            lodo_f1s.append(f1)
            _log.info(f"LODO Held-out Device '{dev}': Acc = {acc:.3f} | F1 = {f1:.3f}")

    lodo_mean_acc = float(np.mean(lodo_accs)) if lodo_accs else 0.0
    lodo_mean_f1 = float(np.mean(lodo_f1s)) if lodo_f1s else 0.0
    _log.info(f"[GNN LODO Summary]: Mean Accuracy = {lodo_mean_acc:.3f} | Mean Macro F1 = {lodo_mean_f1:.3f}")

    # 4. Learning Curve Sweep across dataset fractions
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_accs, lc_f1s = [], []
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(pyg_data))
    for frac in fractions:
        sub_size = max(10, int(frac * len(pyg_data)))
        sub_indices = shuffled_idx[:sub_size]
        sub_data = [pyg_data[i] for i in sub_indices]
        tr_sub = sub_data[:int(0.8 * len(sub_data))]
        te_sub = sub_data[int(0.8 * len(sub_data)):]
        if tr_sub and te_sub:
            acc, f1 = train_eval_gnn_fold(tr_sub, te_sub, len(classes), epochs=10)
            lc_accs.append(acc)
            lc_f1s.append(f1)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_gnn_learning_curve(fractions, lc_accs, lc_f1s, args.out / "gnn_learning_curve.png")


if __name__ == "__main__":
    main()
