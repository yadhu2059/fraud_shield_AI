import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_path):
    print(f"\n[*] Running: {os.path.basename(script_path)}...")
    result = subprocess.run([sys.executable, script_path], cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"[-] Error: {os.path.basename(script_path)} failed with exit code {result.returncode}")
        return False
    print(f"[+] Success: {os.path.basename(script_path)} completed.")
    return True

def main():
    print("="*60)
    # 1. Setup Data (Extract / download Kaggle dataset)
    setup_path = os.path.join(ROOT_DIR, "scripts", "setup_data.py")
    if not run_script(setup_path):
        sys.exit(1)
        
    # 2. Train Fast Lane Baseline Models (XGBoost + Isolation Forest)
    fast_path = os.path.join(ROOT_DIR, "fast_lane", "train_fast.py")
    if not run_script(fast_path):
        sys.exit(1)
        
    # 3. Train Slow Lane Graph Neural Network (GraphSAGE)
    gnn_path = os.path.join(ROOT_DIR, "slow_lane", "train_gnn.py")
    if not run_script(gnn_path):
        sys.exit(1)

    print("\n" + "="*60)
    print(" [+] PIPELINE COMPLETE: ALL MODELS TRAINED AND READY")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
