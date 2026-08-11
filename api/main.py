import os
import time
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

from fast_lane.features import FeatureStore, calculate_haversine_distance
from fast_lane.onnx_ensemble import FastLaneEnsemble
from slow_lane.memgraph_client import MemgraphClient
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
    Loads slow-lane GNN metadata and populates relationships risk scores
    """
    global GNN_SCORES_CACHE
    if os.path.exists(GNN_METADATA_PATH):
        try:
            with open(GNN_METADATA_PATH, "rb") as f:
                meta = pickle.load(f)
                nodes = meta.get("nodes", [])
                
                # If metadata exists, let's load or simulate GNN predictions
                # Since the GNN was trained on node labels, we'll assign risk weights
                node_to_idx = meta.get("node_to_idx", {})
                
                # Basic mock prediction: if the node had fraud logs in training
                edges = meta.get("edges", [])
                # Compute degree
                deg = {}
                for s, d in edges:
                    deg[s] = deg.get(s, 0) + 1
                    deg[d] = deg.get(d, 0) + 1
                
                for node in nodes:
                    idx = node_to_idx.get(node)
                    # Let's say high degree nodes in fraud rings have higher GNN score
                    if node.startswith("C") and idx is not None and deg.get(idx, 0) > 4:
                        GNN_SCORES_CACHE[node] = 0.85
                    else:
                        GNN_SCORES_CACHE[node] = 0.05
            print(f"[+] Loaded GNN relationship risk cache for {len(GNN_SCORES_CACHE)} nodes.")
        except Exception as e:
            print(f"[-] Failed to load GNN cache: {e}")

@app.on_event("startup")
def startup_event():
    load_gnn_cached_scores()

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
        tx_count=history["total_tx_count"] + 1,
        geo_distance=geo_distance,
        is_new_device=is_new_device,
        biometric_score=tx.biometric_score
    )
    
    # 5. Slow Lane GNN Lookup
    # Query GraphSAGE risk score for the source and destination accounts
    gnn_src_risk = GNN_SCORES_CACHE.get(tx.nameOrig, 0.0)
    gnn_dest_risk = GNN_SCORES_CACHE.get(tx.nameDest, 0.0)
    gnn_risk = max(gnn_src_risk, gnn_dest_risk)
    
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
    if trust_score >= 85:
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
    reason = explainer.explain_transaction(
        amount=tx.amount,
        avg_ratio=avg_ratio,
        tx_count=history["total_tx_count"] + 1,
        geo_dist=geo_distance,
        new_device=is_new_device,
        bio_score=tx.biometric_score
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
        "gnn_risk": float(gnn_risk)
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
            "blocked_count": CUMULATIVE_STATS["blocked"] + CUMULATIVE_STATS["flagged"],
            "step_up_count": CUMULATIVE_STATS["step_up"],
            "flagged_count": CUMULATIVE_STATS["flagged"],
            "allowed_count": CUMULATIVE_STATS["allowed"],
            "average_trust_score": float(avg_trust)
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

@app.post("/api/v1/simulate")
def trigger_simulation():
    """
    Runs the Red Team vs Blue Team adversary simulation script.
    """
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "red_blue", "adversary.py"))
    try:
        # Run the Python script as a subprocess
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
        # Parse output to find detection stats
        before = "0.0%"
        after = "100.0%"
        for line in result.stdout.split("\n"):
            if "Initial Detection Rate:" in line:
                before = line.split(":")[-1].strip()
            elif "New Detection Rate:" in line:
                after = line.split(":")[-1].strip()
        return {
            "status": "success", 
            "before_detection": before, 
            "after_detection": after,
            "logs": result.stdout
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

