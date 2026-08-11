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

if __name__ == "__main__":
    setup_directories()
    try:
        install_kaggle_cli()
    except Exception as e:
        print(f"[-] Failed to install Kaggle CLI: {e}. You can download manually.")
    
    download_dataset()
