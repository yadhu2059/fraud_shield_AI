import os
import pickle
import numpy as np

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
XGB_PICKLE_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
IF_PICKLE_PATH = os.path.join(MODEL_DIR, "iforest_model.pkl")
XGB_ONNX_PATH = os.path.join(MODEL_DIR, "xgb_model.onnx")
IF_ONNX_PATH = os.path.join(MODEL_DIR, "iforest_model.onnx")

class FastLaneEnsemble:
    def __init__(self):
        self.xgb_ort_session = None
        self.if_ort_session = None
        self.xgb_pickle = None
        self.if_pickle = None
        self.use_onnx = False

        # Attempt to load ONNX models first if runtime is available
        if ORT_AVAILABLE and os.path.exists(XGB_ONNX_PATH) and os.path.exists(IF_ONNX_PATH):
            try:
                self.xgb_ort_session = ort.InferenceSession(XGB_ONNX_PATH)
                self.if_ort_session = ort.InferenceSession(IF_ONNX_PATH)
                self.use_onnx = True
                print("[+] Loaded ONNX models for fast-lane inference.")
                return
            except Exception as e:
                print(f"[-] Failed to load ONNX sessions: {e}. Falling back to pickle models.")

        # Fallback to standard Pickle models
        print("[*] Loading pickle models for inference...")
        if os.path.exists(XGB_PICKLE_PATH):
            with open(XGB_PICKLE_PATH, "rb") as f:
                self.xgb_pickle = pickle.load(f)
        else:
            print(f"[-] XGBoost model pickle not found at {XGB_PICKLE_PATH}")

        if os.path.exists(IF_PICKLE_PATH):
            with open(IF_PICKLE_PATH, "rb") as f:
                self.if_pickle = pickle.load(f)
        else:
            print(f"[-] Isolation Forest model pickle not found at {IF_PICKLE_PATH}")

    def predict(self, amount, amount_to_avg_ratio, tx_count, geo_distance, is_new_device, biometric_score):
        """
        Runs inference on transaction features.
        Returns:
            xgb_prob (float): supervised risk probability [0, 1]
            if_anomaly (float): unsupervised anomaly status [0, 1] (where 1 is anomalous)
        """
        # Create input array
        input_data = np.array([[
            float(amount),
            float(amount_to_avg_ratio),
            float(tx_count),
            float(geo_distance),
            int(is_new_device),
            float(biometric_score)
        ]], dtype=np.float32)

        # ONNX inference path
        if self.use_onnx:
            try:
                # XGBoost inference
                xgb_inputs = {self.xgb_ort_session.get_inputs()[0].name: input_data}
                xgb_outputs = self.xgb_ort_session.run(None, xgb_inputs)
                # Output format of XGBoost ONNX classifier is typically label and probabilities
                xgb_prob = float(xgb_outputs[1][0][1]) # Class 1 (fraud) probability
                
                # Isolation Forest inference
                if_inputs = {self.if_ort_session.get_inputs()[0].name: input_data}
                if_outputs = self.if_ort_session.run(None, if_inputs)
                # Isolation Forest returns label (-1 for anomaly, 1 for normal)
                if_label = if_outputs[0][0]
                if_anomaly = 1.0 if if_label == -1 else 0.0
                
                return xgb_prob, if_anomaly
            except Exception as e:
                print(f"[-] ONNX inference error: {e}. Attempting Pickle fallback...")

        # Pickle fallback inference path
        xgb_prob = 0.0
        if_anomaly = 0.0

        if self.xgb_pickle:
            try:
                # XGBoost predict_proba
                xgb_prob = float(self.xgb_pickle.predict_proba(input_data)[0][1])
            except Exception as e:
                print(f"[-] XGBoost pickle prediction error: {e}")

        if self.if_pickle:
            try:
                # Isolation Forest predict
                if_label = self.if_pickle.predict(input_data)[0]
                if_anomaly = 1.0 if if_label == -1 else 0.0
            except Exception as e:
                print(f"[-] Isolation Forest pickle prediction error: {e}")

        return xgb_prob, if_anomaly
