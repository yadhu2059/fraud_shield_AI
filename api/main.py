import os
import time
import pickle
import numpy as np
import torch
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

from fast_lane.features import FeatureStore, calculate_haversine_distance
from fast_lane.onnx_ensemble import FastLaneEnsemble
from slow_lane.memgraph_client import MemgraphClient
from slow_lane.train_gnn import FraudGraphSAGE, generate_node_features
from api.explain import ExplainerLayer

app = FastAPI(title="Fraud Shield AI Decision Engine")

# Enable CORS for the dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
feature_store = FeatureStore()
fast_ensemble = FastLaneEnsemble()
db_client = MemgraphClient()
explainer = ExplainerLayer(xgb_model=fast_ensemble.xgb_pickle)

# Global variables for recent transactions & cached GNN risk scores
RECENT_TRANSACTIONS = []
GNN_SCORES_CACHE = {} # Map of username -> GNN risk score
TRUST_SCORES_DYNAMIC = {} # In-memory profile to track credit-score like trust levels
CUMULATIVE_STATS = {
    "total": 0,
    "allowed": 0,
    "step_up": 0,
    "flagged": 0,
    "blocked": 0
}

GNN_MODEL = None
GNN_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "gnn_model.pth"))
GNN_METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "gnn_metadata.pkl"))

class TransactionPayload(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    device_id: str
    os: str
    browser: str
    latitude: float
    longitude: float
    biometric_score: float
    ip_address: str
    isFraud: Optional[int] = 0
    timestamp: Optional[str] = None

def load_gnn_cached_scores():
    """
    Loads slow-lane GNN model weights and metadata to execute forward passes and populate relationship risk scores.
    """
    global GNN_SCORES_CACHE, GNN_MODEL
    
    # 1. Initialize and load model weights
    if os.path.exists(GNN_MODEL_PATH):
        try:
            # We initialize the model with in_features=3 (log_degree, ratio, struct_val)
            GNN_MODEL = FraudGraphSAGE(in_features=3, hidden_features=16)
            GNN_MODEL.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=torch.device('cpu')))
            GNN_MODEL.eval()
            print("[+] Loaded PyTorch GNN model weights successfully.")
        except Exception as e:
            print(f"[-] Failed to load GNN model weights: {e}")
            GNN_MODEL = None

    # 2. Load metadata and run predictions
    if os.path.exists(GNN_METADATA_PATH):
        try:
            with open(GNN_METADATA_PATH, "rb") as f:
                meta = pickle.load(f)
                
            nodes = meta.get("nodes", [])
            node_to_idx = meta.get("node_to_idx", {})
            edges = meta.get("edges", [])
            features = meta.get("features", [])
            
            if len(nodes) > 0 and len(features) > 0 and GNN_MODEL is not None:
                x = torch.tensor(features, dtype=torch.float32)
                if len(edges) > 0:
                    edge_sources = [e[0] for e in edges]
                    edge_targets = [e[1] for e in edges]
                    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
                else:
                    edge_index = torch.tensor([[], []], dtype=torch.long)
                
                # Execute forward pass
                with torch.no_grad():
                    predictions = GNN_MODEL(x, edge_index)
                    predictions = predictions.numpy().flatten()
                
                for node in nodes:
                    idx = node_to_idx.get(node)
                    if idx is not None and idx < len(predictions):
                        GNN_SCORES_CACHE[node] = float(predictions[idx])
                        
                print(f"[+] Populated GNN risk cache for {len(GNN_SCORES_CACHE)} nodes using live model inference.")
            else:
                # Fallback to degree rule if GNN_MODEL load failed
                print("[*] GNN model not loaded. Populating cache using fallback degree rule...")
                deg = {}
                for s, d in edges:
                    deg[s] = deg.get(s, 0) + 1
                    deg[d] = deg.get(d, 0) + 1
                for node in nodes:
                    idx = node_to_idx.get(node)
                    if node.startswith("C") and idx is not None and deg.get(idx, 0) > 4:
                        GNN_SCORES_CACHE[node] = 0.85
                    else:
                        GNN_SCORES_CACHE[node] = 0.05
                print(f"[+] Loaded GNN fallback cache for {len(GNN_SCORES_CACHE)} nodes.")
        except Exception as e:
            print(f"[-] Failed to build GNN cache: {e}")

def load_historical_transactions():
    """
    Loads the last 50 transactions from the database (Memgraph or SQLite) 
    and populates RECENT_TRANSACTIONS on startup.
    """
    global RECENT_TRANSACTIONS
    try:
        if db_client.use_sqlite and hasattr(db_client, 'conn'):
            with db_client.lock:
                cursor = db_client.conn.cursor()
                cursor.execute("""
                    SELECT source, destination, amount, timestamp, is_fraud 
                    FROM transfers 
                    ORDER BY timestamp DESC, id DESC LIMIT 50
                """)
                rows = cursor.fetchall()
                
                loaded_txs = []
                for src, dest, amt, ts, is_f in rows:
                    cursor.execute("SELECT device_id FROM device_links WHERE account = ? LIMIT 1", (src,))
                    dev_row = cursor.fetchone()
                    dev_id = dev_row[0] if dev_row else "dev_unknown"
                    
                    trust_score = 95.5 if is_f == 0 else 25.0
                    decision = "ALLOW" if is_f == 0 else "BLOCK"
                    reason = "Transaction exhibits standard transaction behavior and verified device credentials." if is_f == 0 else "Flagged because: connection to known fraud rings, high risk."
                    
                    loaded_txs.append({
                        "timestamp": ts,
                        "account_id": src,
                        "destination": dest,
                        "amount": amt,
                        "trust_score": trust_score,
                        "decision": decision,
                        "reason": reason,
                        "device_fingerprint": dev_id,
                        "geolocation": "28.6139, 77.2090",
                        "biometric_score": 0.85,
                        "ip_address": "192.168.1.1",
                        "xgb_prob": 0.001 if is_f == 0 else 0.95,
                        "if_anomaly": 0 if is_f == 0 else 1,
                        "gnn_risk": 0.01 if is_f == 0 else 0.85,
                        "shap_attributions": {
                            "Transaction Amount": 0.01 if is_f == 0 else 0.35,
                            "Amount to Avg Ratio": 0.01 if is_f == 0 else 0.25,
                            "Transaction Velocity": 0.01 if is_f == 0 else 0.20,
                            "Geographical Distance": 0.01 if is_f == 0 else 0.15
                        }
                    })
                
                RECENT_TRANSACTIONS = loaded_txs
                # Also update cumulative counts on startup to match history
                CUMULATIVE_STATS["total"] = len(RECENT_TRANSACTIONS)
                CUMULATIVE_STATS["allowed"] = sum(1 for tx in RECENT_TRANSACTIONS if tx["decision"] == "ALLOW")
                CUMULATIVE_STATS["blocked"] = sum(1 for tx in RECENT_TRANSACTIONS if tx["decision"] == "BLOCK")
                CUMULATIVE_STATS["flagged"] = sum(1 for tx in RECENT_TRANSACTIONS if tx["decision"] == "FLAG")
                CUMULATIVE_STATS["step_up"] = sum(1 for tx in RECENT_TRANSACTIONS if tx["decision"] == "STEP_UP")
                print(f"[+] Loaded {len(RECENT_TRANSACTIONS)} historical transactions from SQLite.")
            
    except Exception as e:
        print(f"[-] Failed to load historical transactions: {e}")

@app.on_event("startup")
def startup_event():
    load_gnn_cached_scores()
    load_historical_transactions()

def get_dynamic_gnn_risk(source_node: str, dest_node: str) -> float:
    """
    Dynamically queries the latest graph topology, runs the PyTorch GNN,
    and returns the computed risk score for the active nodes.
    """
    global GNN_SCORES_CACHE
    if GNN_MODEL is None:
        return max(GNN_SCORES_CACHE.get(source_node, 0.0), GNN_SCORES_CACHE.get(dest_node, 0.0))
        
    try:
        nodes, edges, node_labels = db_client.get_all_edges_and_nodes()
        
        if len(nodes) == 0:
            return max(GNN_SCORES_CACHE.get(source_node, 0.0), GNN_SCORES_CACHE.get(dest_node, 0.0))
            
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        if source_node not in node_to_idx:
            nodes.append(source_node)
            node_to_idx[source_node] = len(nodes) - 1
        if dest_node not in node_to_idx:
            nodes.append(dest_node)
            node_to_idx[dest_node] = len(nodes) - 1
            
        x, _ = generate_node_features(nodes, edges, node_labels)
        
        if len(edges) > 0:
            edge_sources = [e[0] for e in edges]
            edge_targets = [e[1] for e in edges]
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            
        with torch.no_grad():
            predictions = GNN_MODEL(x, edge_index)
            predictions = predictions.numpy().flatten()
            
        src_risk = float(predictions[node_to_idx[source_node]])
        dest_risk = float(predictions[node_to_idx[dest_node]])
        
        GNN_SCORES_CACHE[source_node] = src_risk
        GNN_SCORES_CACHE[dest_node] = dest_risk
        
        return max(src_risk, dest_risk)
        
    except Exception as e:
        print(f"[-] Dynamic GNN evaluation failed: {e}. Falling back to cache.")
        return max(GNN_SCORES_CACHE.get(source_node, 0.0), GNN_SCORES_CACHE.get(dest_node, 0.0))

@app.post("/api/v1/transaction")
def process_transaction(tx: TransactionPayload):
    global RECENT_TRANSACTIONS
    
    # 1. Fetch historical statistics from FeatureStore (Redis or local)
    history = feature_store.get_features(tx.nameOrig)
    
    # Update Feature Store with latest values
    feature_store.update_features(
        account_id=tx.nameOrig,
        amount=tx.amount,
        latitude=tx.latitude,
        longitude=tx.longitude,
        device_id=tx.device_id
    )
    
    # 2. Sync to Memgraph / Graph SQLite database (Slow Lane Sync)
    db_client.add_transaction(
        source=tx.nameOrig,
        destination=tx.nameDest,
        amount=tx.amount,
        device_id=tx.device_id,
        timestamp=tx.timestamp or str(time.time()),
        is_fraud=tx.isFraud
    )
    
    # 3. Calculate Fast Lane Feature Indicators
    avg_ratio = tx.amount / (history["avg_amount"] + 1e-5) if history["total_tx_count"] > 0 else 1.0
    
    # Device Change Check
    is_new_device = 0
    if history["devices"] and tx.device_id not in history["devices"]:
        is_new_device = 1
        
    # Geolocation Distance Check (Impossible Travel)
    geo_distance = 0.0
    if history["last_latitude"] != 0.0:
        geo_distance = calculate_haversine_distance(
            history["last_latitude"], 
            history["last_longitude"], 
            tx.latitude, 
            tx.longitude
        )
        
    # 4. Fast Lane Model Inference (XGBoost + Isolation Forest)
    xgb_prob, if_anomaly = fast_ensemble.predict(
        amount=tx.amount,
        amount_to_avg_ratio=avg_ratio,
        tx_count=history["total_tx_count"],
        geo_distance=geo_distance,
        is_new_device=is_new_device,
        biometric_score=tx.biometric_score
    )
    
    # 5. Slow Lane GNN Lookup
    # Query GraphSAGE risk score dynamically using PyTorch model
    gnn_risk = get_dynamic_gnn_risk(tx.nameOrig, tx.nameDest)
    
    # 6. Trust Score (Credit-style) Calculations
    # Start with baseline normal score
    # Fetch existing dynamic trust score or initialize
    prev_trust = TRUST_SCORES_DYNAMIC.get(tx.nameOrig, 95.0)
    
    # Calculate deductions
    deductions = 0.0
    deductions += xgb_prob * 55.0         # Supervised threat
    deductions += if_anomaly * 15.0        # Unsupervised anomaly
    deductions += gnn_risk * 30.0          # Slow-lane GNN relationship risk
    deductions += is_new_device * 10.0     # Device change
    
    # Biometric scoring contribution
    if tx.biometric_score < 0.5:
        deductions += (1.0 - tx.biometric_score) * 20.0
        
    # Travel anomaly contribution
    if geo_distance > 300.0:
        deductions += 15.0

    raw_trust = 100.0 - deductions
    
    # Credit score style recovery/dampening
    # Trust score drops sharply but recovers slowly (e.g. 5% towards new score if increasing)
    if raw_trust < prev_trust:
        # Sharp drop
        trust_score = max(0.0, raw_trust)
    else:
        # Slow recovery
        trust_score = min(100.0, prev_trust + (raw_trust - prev_trust) * 0.1)
        
    TRUST_SCORES_DYNAMIC[tx.nameOrig] = trust_score
    
    # 7. Action Decision Tiers
    if trust_score >= 90:
        decision = "ALLOW"
    elif trust_score >= 60:
        decision = "STEP_UP" # Needs multi-factor authentication
    elif trust_score >= 30:
        decision = "FLAG"    # Human analyst review queue
    else:
        decision = "BLOCK"   # Instant rejection

    # Update cumulative statistics
    CUMULATIVE_STATS["total"] += 1
    if decision == "ALLOW":
        CUMULATIVE_STATS["allowed"] += 1
    elif decision == "STEP_UP":
        CUMULATIVE_STATS["step_up"] += 1
    elif decision == "FLAG":
        CUMULATIVE_STATS["flagged"] += 1
    elif decision == "BLOCK":
        CUMULATIVE_STATS["blocked"] += 1

    # 8. SHAP Explainability
    reason, shap_attributions = explainer.explain_transaction(
        amount=tx.amount,
        avg_ratio=avg_ratio,
        tx_count=history["total_tx_count"] + 1,
        geo_dist=geo_distance,
        new_device=is_new_device,
        bio_score=tx.biometric_score,
        decision=decision
    )
    
    response = {
        "timestamp": tx.timestamp or datetime.utcnow().isoformat(),
        "account_id": tx.nameOrig,
        "destination": tx.nameDest,
        "amount": tx.amount,
        "trust_score": trust_score,
        "decision": decision,
        "reason": reason,
        "device_fingerprint": tx.device_id,
        "geolocation": f"{tx.latitude:.4f}, {tx.longitude:.4f}",
        "biometric_score": tx.biometric_score,
        "ip_address": tx.ip_address,
        "xgb_prob": float(xgb_prob),
        "if_anomaly": int(if_anomaly),
        "gnn_risk": float(gnn_risk),
        "shap_attributions": shap_attributions
    }
    
    # Store in recent transactions list
    RECENT_TRANSACTIONS.insert(0, response)
    # Cap at last 50 transactions
    if len(RECENT_TRANSACTIONS) > 50:
        RECENT_TRANSACTIONS.pop()
        
    return response

@app.get("/api/v1/dashboard")
def get_dashboard_stats():
    """
    Returns statistics and logs for the frontend monitoring system.
    """
    # Calculate average trust score
    avg_trust = np.mean([tx["trust_score"] for tx in RECENT_TRANSACTIONS]) if RECENT_TRANSACTIONS else 95.0
    
    return {
        "recent_transactions": RECENT_TRANSACTIONS,
        "stats": {
            "total_transactions": CUMULATIVE_STATS["total"],
            "blocked_count": CUMULATIVE_STATS["blocked"],
            "step_up_count": CUMULATIVE_STATS["step_up"],
            "flagged_count": CUMULATIVE_STATS["flagged"],
            "allowed_count": CUMULATIVE_STATS["allowed"],
            "average_trust_score": float(avg_trust)
        },
        "health": {
            "onnx": "ONLINE" if fast_ensemble.xgb_pickle is not None else "ERROR",
            "gnn": "ONLINE" if GNN_MODEL is not None else "CACHED"
        }
    }

@app.post("/api/v1/retrain")
def trigger_retrain():
    """
    Reloads the models and cached GNN scores after Red vs. Blue training.
    """
    global fast_ensemble, explainer
    # Reinitialize FAST models
    fast_ensemble = FastLaneEnsemble()
    explainer = ExplainerLayer(xgb_model=fast_ensemble.xgb_pickle)
    load_gnn_cached_scores()
    return {"status": "success", "message": "Decision Engine models reloaded."}

import subprocess
import sys
import json

@app.post("/api/v1/simulate")
def trigger_simulation():
    """
    Runs the Red Team vs Blue Team adversary simulation script.
    """
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "red_blue", "adversary.py"))
    try:
        # Run the Python script as a subprocess
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
        
        # Parse JSON_OUTPUT from stdout
        metrics = {}
        for line in result.stdout.split("\n"):
            if line.startswith("JSON_OUTPUT:"):
                try:
                    json_str = line[len("JSON_OUTPUT:"):].strip()
                    metrics = json.loads(json_str)
                except Exception as ex:
                    print(f"[-] Failed to parse JSON_OUTPUT line: {ex}")
                break
        
        before_pct = metrics.get("initial_detection_rate", 0.0)
        after_pct = metrics.get("retrained_detection_rate", 0.0)
        
        # Filter JSON line out of logs to keep the console printout clean
        log_lines = [line for line in result.stdout.split("\n") if not line.startswith("JSON_OUTPUT:")]
        clean_logs = "\n".join(log_lines)
        
        return {
            "status": "success", 
            "before_detection": f"{before_pct:.2f}%", 
            "after_detection": f"{after_pct:.2f}%",
            "logs": clean_logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

@app.get("/api/v1/graph/{account_id}")
def get_graph_neighborhood(account_id: str):
    """
    Returns the rich graph neighborhood (nodes & edges) around a given account for UI visualization.
    """
    try:
        data = db_client.get_rich_graph_neighborhood(account_id, RECENT_TRANSACTIONS)
        
        # Add status field to nodes and edges
        nodes = data.get("nodes", [])
        for n in nodes:
            is_fraud = bool(n.get("risk", 0.0) > 0.5 or n.get("in_ring", False))
            n["status"] = "fraud" if is_fraud else "safe"
            
        edges = data.get("edges", [])
        for e in edges:
            is_transaction = e.get("type") in ("transaction", "TRANSFERRED")
            is_fraud = bool(e.get("risk_score", 0.0) > 0.5 or e.get("decision") == "BLOCK" or e.get("in_ring", False))
            e["status"] = "fraud_transfer" if (is_transaction and is_fraud) else "normal"
            
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch neighborhood for {account_id}: {e}")

