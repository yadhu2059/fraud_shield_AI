# Fraud Shield AI - Real-time Risk Decisioning & Graph AI Pipeline

**Fraud Shield AI** is a real-time, hybrid financial transaction protection engine designed for high-throughput transactional environments. It implements a dual-speed pipeline ("Fast Lane" and "Slow Lane") combined with an active defense feedback loop (Red vs. Blue Team simulation) to detect and block complex threat vectors like account takeover, mule rings, synthetic identity, and evasive adversarial attacks.


## ⚙️ Fallback Safeguards

To allow instant execution in local developer environments without complex infrastructure setups, the codebase implements automatic local fallbacks for all services:

* **Redis Feature Store** $\rightarrow$ Falls back to an **In-memory Dictionary Store**.
* **Memgraph Graph DB** $\rightarrow$ Falls back to a **Local SQLite database (`data/graph_fallback.db`)**.
* **ONNX Runtime** $\rightarrow$ Falls back to standard **Python Pickle files (`.pkl`)**.
* **SHAP Library** $\rightarrow$ Falls back to a **Static Rule-based Explainer**.
* **PyG / CUDA** $\rightarrow$ Custom GraphSAGE is written in raw PyTorch using a mean aggregator, running on **CPU** out-of-the-box.

---

## 🚀 Execution & Quick Start Guide

Follow these steps to build the pipeline, spin up the decision backend, and launch the stream simulation:

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Install the package requirements:
```bash
pip install -r requirements.txt
```

### 2. Launch Infrastructure Containers (Optional)
If Docker is installed, spin up Redis, Redpanda, and Memgraph:
```bash
docker-compose up -d
```
*(If Docker is omitted, all fallback safeguards will engage automatically.)*

### 3. Initialize Data & Train Models
Downloads the PaySim Kaggle dataset (or generates synthetic transactions if Kaggle API keys are missing) and trains the XGBoost, Isolation Forest, and GraphSAGE models:
```bash
python run_pipeline.py
```

#### Dataset Download Troubleshooting
If the automated Kaggle CLI download fails (e.g., due to missing `~/.kaggle/kaggle.json` credentials):
* **Manual Download Option**:
  1. Visit the dataset page: https://www.kaggle.com/datasets/ealaxi/paysim1
  2. Download the `paysim1.zip` archive (approx. 178MB).
  3. Save the ZIP file directly as `data/paysim1.zip` (create the `data/` folder in the project root if it does not exist).
  4. Run `python run_pipeline.py` again. The setup utility will detect the ZIP file, extract the structured CSV log, and train all models.
* **Automatic Synthetic Fallback**:
  * If no Kaggle token or local ZIP file is present, the script automatically triggers a fallback generator to build 30,000 synthetic transaction records, allowing you to run the API and dashboard out-of-the-box.

### 4. Start the Decision Server
Boot the FastAPI server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Stream Live Transactions
Start the streaming simulator to feed live normal transactions, travel shifts, and device-sharing rings into the decision server:
```bash
python scripts/stream_producer.py
```

### 6. View the Dashboard
Simply open the local HTML file in your web browser:
📁 **`frontend/index.html`**

---

## 📝 API Documentation

FastAPI automatically hosts interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

### 1. Evaluate Transaction
* **Endpoint**: `POST /api/v1/transaction`
* **Request Schema**:
  ```json
  {
    "step": 1,
    "type": "TRANSFER",
    "amount": 2500.0,
    "nameOrig": "C1064",
    "nameDest": "C1379",
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 2500.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 2500.0,
    "isFraud": 0,
    "device_id": "dev_phone_12",
    "latitude": 28.6139,
    "longitude": 77.2090,
    "biometric_score": 0.85
  }
  ```
* **Response Tiers**:
  * `"ALLOW"` (Trust Score $\ge$ 90)
  * `"STEP_UP"` (MFA Verification Required, Trust Score $\ge$ 60)
  * `"FLAG"` (Human analyst queue, Trust Score $\ge$ 30)
  * `"BLOCK"` (Instant rejection, Trust Score $<$ 30)

### 2. Dashboard Statistics
* **Endpoint**: `GET /api/v1/dashboard`
* **Response**:
  ```json
  {
    "stats": {
      "total_transactions": 250,
      "allowed_count": 180,
      "step_up_count": 15,
      "flagged_count": 10,
      "blocked_count": 45,
      "average_trust_score": 82.4
    },
    "recent_transactions": [...]
  }
  ```

### 3. Account Graph Neighborhood
* **Endpoint**: `GET /api/v1/graph/{account_id}`
* **Response**: Returns Cytoscape.js compatible graph neighborhood data including connecting accounts, device identifiers, status classifications (`"safe"` / `"fraud"`), and edges (`"normal"` / `"fraud_transfer"`).

### 4. Retrain Adversarial Simulation
* **Endpoint**: `POST /api/v1/simulate`
* **Response**: Simulates a Red Team smurfing/structuring run, retrains model thresholds, and returns logs showing detection rates.
