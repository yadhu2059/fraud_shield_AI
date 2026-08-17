import os
import zipfile
import subprocess
import sys

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ZIP_PATH = os.path.join(DATA_DIR, "paysim1.zip")
CSV_PATH = os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv")

def setup_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[*] Created/verified data directory at: {DATA_DIR}")

def install_kaggle_cli():
    print("[*] Checking for Kaggle CLI installation...")
    try:
        import kaggle
        print("[+] Kaggle Python package is already installed.")
    except ImportError:
        print("[*] Installing Kaggle CLI package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("[+] Kaggle CLI successfully installed.")

def download_dataset():
    if os.path.exists(CSV_PATH):
        print(f"[+] Dataset already extracted at {CSV_PATH}")
        return True

    # Check for kaggle credentials
    home_dir = os.path.expanduser("~")
    kaggle_config_dir = os.path.join(home_dir, ".kaggle")
    kaggle_json_path = os.path.join(kaggle_config_dir, "kaggle.json")

    if not os.path.exists(kaggle_json_path):
        print("\n" + "="*80)
        print(" [!] KAGGLE CREDENTIALS NOT FOUND")
        print("="*80)
        print("To download the dataset automatically, please place your 'kaggle.json' file in:")
        print(f"  {kaggle_config_dir}")
        print("\nAlternatively, download the PaySim dataset manually:")
        print("  1. Go to: https://www.kaggle.com/datasets/ealaxi/paysim1")
        print("  2. Click 'Download' (approx. 178MB zip file).")
        print(f"  3. Save the zip file as 'paysim1.zip' directly inside the data folder:")
        print(f"     {DATA_DIR}")
        print("  4. Re-run this script to extract it.")
        print("="*80 + "\n")
        
        # Check if the user already downloaded the zip file manually
        if os.path.exists(ZIP_PATH):
            print("[*] Found manually downloaded 'paysim1.zip'. Proceeding to extract...")
            extract_zip()
            return True
        return False

    print("[*] Downloading PaySim dataset from Kaggle...")
    try:
        # We invoke the kaggle CLI tool via subprocess
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "ealaxi/paysim1", 
            "-p", DATA_DIR, 
            "--unzip"
        ], check=True)
        print("[+] Download and extraction complete.")
        
        # Check if it was downloaded as a zip or if --unzip worked
        # The kaggle CLI might download paysim1.zip or extract it.
        # Let's check files in DATA_DIR
        for f in os.listdir(DATA_DIR):
            if f.endswith(".csv"):
                # Rename to standard path if it matches the pattern
                src = os.path.join(DATA_DIR, f)
                if "PS_" in f:
                    os.rename(src, CSV_PATH)
                    print(f"[+] Standardized CSV filename to: {CSV_PATH}")
                    break
        return True
    except Exception as e:
        print(f"[-] Kaggle download failed: {e}")
        if os.path.exists(ZIP_PATH):
            print("[*] Found 'paysim1.zip'. Extracting...")
            extract_zip()
            return True
        return False

def extract_zip():
    if not os.path.exists(ZIP_PATH):
        print(f"[-] Zip file not found at {ZIP_PATH}")
        return False
    
    print(f"[*] Extracting {ZIP_PATH} to {DATA_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Check and rename CSV
    for f in os.listdir(DATA_DIR):
        if f.endswith(".csv") and "PS_" in f:
            os.rename(os.path.join(DATA_DIR, f), CSV_PATH)
            print(f"[+] Extracted and standardized: {CSV_PATH}")
            return True
    print("[-] CSV extraction could not find the standardized PaySim CSV.")
    return False

def generate_synthetic_data(file_path, num_rows=30000):
    print(f"[*] Generating {num_rows} rows of synthetic PaySim data as a fallback...")
    import pandas as pd
    import numpy as np
    import random
    
    random.seed(42)
    np.random.seed(42)
    
    data = []
    
    # Generate account pools
    num_accounts = num_rows // 5
    account_pool = [f"C{random.randint(1000000, 9999999)}" for _ in range(num_accounts)]
    dest_pool = [f"C{random.randint(1000000, 9999999)}" for _ in range(num_accounts)]
    merchant_pool = [f"M{random.randint(1000000, 9999999)}" for _ in range(num_accounts // 10)]
    
    for i in range(num_rows):
        step = int(1 + (i // (num_rows // 5))) # steps 1 to 5
        tx_type = random.choices(
            ["CASH_OUT", "TRANSFER", "CASH_IN", "PAYMENT", "DEBIT"],
            weights=[0.35, 0.20, 0.25, 0.18, 0.02]
        )[0]
        
        amount = float(np.random.exponential(scale=15000.0))
        if tx_type == "TRANSFER" and random.random() < 0.10:
            # 10% transfers are high amount
            amount = float(random.uniform(200000.0, 900000.0))
            
        name_orig = random.choice(account_pool)
        name_dest = random.choice(merchant_pool) if tx_type == "PAYMENT" else random.choice(dest_pool)
        
        # Balance details
        old_bal_orig = float(np.random.exponential(scale=50000.0))
        if amount > old_bal_orig:
            new_bal_orig = 0.0
        else:
            new_bal_orig = old_bal_orig - amount
            
        old_bal_dest = float(np.random.exponential(scale=100000.0))
        if tx_type in ["CASH_OUT", "TRANSFER"]:
            new_bal_dest = old_bal_dest + amount
        else:
            new_bal_dest = old_bal_dest
            
        # 1% fraud rate
        is_fraud = 0
        if tx_type in ["TRANSFER", "CASH_OUT"]:
            if amount > 150000.0 and random.random() < 0.08:
                is_fraud = 1
                old_bal_orig = amount
                new_bal_orig = 0.0
                
        is_flagged_fraud = 0
        if is_fraud and amount > 200000.0:
            is_flagged_fraud = 1 if random.random() < 0.01 else 0
            
        data.append({
            "step": step,
            "type": tx_type,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_bal_orig,
            "newbalanceOrig": new_bal_orig,
            "nameDest": name_dest,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged_fraud
        })
        
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"[+] Successfully wrote synthetic dataset fallback to {file_path}")

if __name__ == "__main__":
    setup_directories()
    try:
        install_kaggle_cli()
    except Exception as e:
        print(f"[-] Failed to install Kaggle CLI: {e}. You can download manually.")
    
    success = download_dataset()
    if not success:
        print("[!] Dataset download failed or skipped. Triggering synthetic generator fallback...")
        generate_synthetic_data(CSV_PATH)
