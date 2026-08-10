import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from slow_lane.memgraph_client import MemgraphClient

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
GNN_MODEL_PATH = os.path.join(MODEL_DIR, "gnn_model.pth")
GNN_METADATA_PATH = os.path.join(MODEL_DIR, "gnn_metadata.pkl")

class RawSageConv(nn.Module):
    """
    Lightweight, custom GraphSAGE Mean Aggregator layer implemented in raw PyTorch.
    Eliminates dependency on torch-geometric / CUDA binary compile issues.
    """
    def __init__(self, in_features, out_features):
        super(RawSageConv, self).__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges] or list of tuples
        num_nodes = x.size(0)
        
        # Step 1: Aggregate neighbor features
        # Create an adjacency aggregation tensor
        agg = torch.zeros_like(x)
        degrees = torch.zeros(num_nodes, 1, device=x.device)
        
        # Pull source & target indices
        sources, targets = edge_index
        
        # Scatter add neighbors (undirected representation)
        agg.index_add_(0, targets, x[sources])
        degrees.index_add_(0, targets, torch.ones(len(sources), 1, device=x.device))
        
        # Self loop aggregation (add source to source)
        agg.index_add_(0, sources, x[targets])
        degrees.index_add_(0, sources, torch.ones(len(sources), 1, device=x.device))
        
        # Normalize by node degrees (avoid divide by zero)
        degrees = torch.clamp(degrees, min=1.0)
        agg_mean = agg / degrees
        
        # Step 2: Combine and project
        out = self.linear_self(x) + self.linear_neigh(agg_mean)
        return out

class FraudGraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1):
        super(FraudGraphSAGE, self).__init__()
        self.conv1 = RawSageConv(in_features, hidden_features)
        self.conv2 = RawSageConv(hidden_features, hidden_features)
        self.classifier = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        out = torch.sigmoid(self.classifier(h))
        return out

def generate_node_features(nodes, edges, node_labels):
    """
    Generates feature vectors for each account node in the graph.
    Features:
        1. log(degree + 1)
        2. Average tx amount (placeholder logic or computed)
        3. Transaction frequency ratio
        4. is_labeled_fraud indicator (for training nodes)
    """
    num_nodes = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Calculate degree
    degrees = np.zeros(num_nodes)
    for s, d in edges:
        degrees[s] += 1
        degrees[d] += 1
        
    features = []
    for i, node in enumerate(nodes):
        # Feature vector: [log_degree, ratio_active, synthetic_structural_property]
        log_deg = np.log1p(degrees[i])
        ratio = degrees[i] / (max(degrees) + 1.0)
        struct_val = hash(node) % 100 / 100.0  # Stable synthetic feature
        
        features.append([log_deg, ratio, struct_val])
        
    x = torch.tensor(features, dtype=torch.float32)
    
    # Create labels
    y = np.zeros((num_nodes, 1))
    for node, label in node_labels.items():
        if node in node_to_idx:
            y[node_to_idx[node]] = label
            
    return x, torch.tensor(y, dtype=torch.float32)

def train_gnn():
    os.makedirs(MODEL_DIR, exist_ok=True)
    client = MemgraphClient()
    
    print("[*] Retrieving graph structure from database client...")
    nodes, edges, node_labels = client.get_all_edges_and_nodes()
    
    if len(nodes) < 2:
        print("[!] Not enough nodes/edges in graph to train GNN. Loading fallback sample dataset.")
        # Create a mock small graph for baseline compile test if DB is empty
        nodes = [f"C{i}" for i in range(100)]
        edges = []
        # Create circular & hub structures
        for i in range(99):
            edges.append((i, i+1))
        # Add a circular fraudulent ring
        edges.append((99, 0))
        # Hub node
        for i in range(5):
            edges.append((0, i*10))
        node_labels = {"C0": 1, "C1": 1, "C99": 1} # Fraud ring
    
    print(f"[+] Loaded graph: {len(nodes)} nodes, {len(edges)} edges.")
    
    # Prepare data for PyTorch
    x, y = generate_node_features(nodes, edges, node_labels)
    
    # Convert edges to tensor
    if len(edges) > 0:
        edge_sources = [e[0] for e in edges]
        edge_targets = [e[1] for e in edges]
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        
    # Model definition
    in_features = x.shape[1]
    model = FraudGraphSAGE(in_features=in_features, hidden_features=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # Training Loop
    print("[*] Training FraudGraphSAGE...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/100 | Loss: {loss.item():.4f}")
            
    # Save GNN model weights
    torch.save(model.state_dict(), GNN_MODEL_PATH)
    
    # Save GNN metadata (node names to indexes map)
    metadata = {
        "nodes": nodes,
        "node_to_idx": {n: i for i, n in enumerate(nodes)},
        "edges": edges,
        "features": x.tolist()
    }
    with open(GNN_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
        
    print(f"[+] GNN weights saved to {GNN_MODEL_PATH}")
    print(f"[+] GNN node map saved to {GNN_METADATA_PATH}")
    client.close()

if __name__ == "__main__":
    train_gnn()
