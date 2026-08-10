import os
import sys

# Add root folder to python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

def run_pipeline_test():
    print("="*60)
    print(" FRAUD SHIELD AI INTEGRATION & PIPELINE TESTING")
    print("="*60)
    
    # 1. Compile/Train Fast Lane Baseline Models
    print("\n[*] Step 1: Compiling Fast Lane baseline models...")
    try:
        from fast_lane.train_fast import train_models
        train_models()
        print("[+] Success: XGBoost and Isolation Forest models compiled.")
    except Exception as e:
        print(f"[-] Failed compiling Fast Lane models: {e}")
        return False
        
    # 2. Compile/Train Slow Lane GNN Baseline Models
    print("\n[*] Step 2: Compiling GNN GraphSAGE model...")
    try:
        from slow_lane.train_gnn import train_gnn
        train_gnn()
        print("[+] Success: GraphSAGE model compiled.")
    except Exception as e:
        print(f"[-] Failed compiling GNN model: {e}")
        return False

    # 3. Test Inference
    print("\n[*] Step 3: Verifying inference engines...")
    try:
        from fast_lane.onnx_ensemble import FastLaneEnsemble
        ensemble = FastLaneEnsemble()
        # Test predict with sample parameters
        prob, anomaly = ensemble.predict(
            amount=150.0, 
            amount_to_avg_ratio=1.2, 
            tx_count=5, 
            geo_distance=2.5, 
            is_new_device=0, 
            biometric_score=0.85
        )
        print(f"[+] Success: Inference returned XGBoost Prob: {prob:.4f}, Anomaly flag: {anomaly}")
    except Exception as e:
        print(f"[-] Failed running inference tests: {e}")
        return False

    # 4. Test Graph Database fallbacks
    print("\n[*] Step 4: Verifying graph database client...")
    try:
        from slow_lane.memgraph_client import MemgraphClient
        client = MemgraphClient()
        # Insert a sample relationship
        client.add_transaction("Alice", "Bob", 300.0, "dev_xyz123", "12345678", is_fraud=0)
        nodes, edges, labels = client.get_all_edges_and_nodes()
        client.close()
        print(f"[+] Success: Graph Client resolved {len(nodes)} nodes, {len(edges)} edges.")
    except Exception as e:
        print(f"[-] Failed graph database tests: {e}")
        return False

    # 5. Verify FastAPI boot imports
    print("\n[*] Step 5: Validating decision API imports...")
    try:
        from api.main import app
        print("[+] Success: FastAPI router validated.")
    except Exception as e:
        print(f"[-] Failed importing FastAPI app: {e}")
        return False

    print("\n" + "="*60)
    print(" [+] INTEGRATION VERIFICATION: ALL SYSTEMS SECURE")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    success = run_pipeline_test()
    sys.exit(0 if success else 1)
