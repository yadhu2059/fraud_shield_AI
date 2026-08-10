import numpy as np

# Try importing shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class ExplainerLayer:
    def __init__(self, xgb_model=None):
        self.xgb_model = xgb_model
        self.shap_explainer = None
        
        if SHAP_AVAILABLE and xgb_model is not None:
            try:
                # Use TreeExplainer for XGBoost
                self.shap_explainer = shap.TreeExplainer(xgb_model)
                print("[+] Initialized SHAP TreeExplainer.")
            except Exception as e:
                print(f"[-] SHAP initialization failed: {e}. Using rule-based fallback explainer.")

    def explain_transaction(self, amount, avg_ratio, tx_count, geo_dist, new_device, bio_score):
        """
        Generates a human-readable explanation of why a transaction was scored the way it was.
        """
        features = {
            "Transaction Amount": (amount, 500.0, "high amount"),
            "Amount to Avg Ratio": (avg_ratio, 3.0, "amount is significantly higher than your average"),
            "Transaction Velocity": (tx_count, 10.0, "high frequency of transactions"),
            "Geographical Distance": (geo_dist, 50.0, "location changed rapidly compared to last login"),
            "Device Change": (new_device, 0.5, "using a new device"),
            "Biometric Consistency": (bio_score, 0.5, "unusual or superhuman typing speed/patterns")
        }

        reasons = []

        # If SHAP is available, use feature attributions to select the main driver
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                input_data = np.array([[
                    float(amount),
                    float(avg_ratio),
                    float(tx_count),
                    float(geo_dist),
                    int(new_device),
                    float(bio_score)
                ]], dtype=np.float32)
                
                shap_values = self.shap_explainer.shap_values(input_data)
                # For binary classification, shap_values might be 2D array [classes, features] or 1D array [features]
                if isinstance(shap_values, list):
                    # For some versions, it returns a list of classes
                    shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                elif len(shap_values.shape) == 3:
                    shap_vals = shap_values[0][0]
                elif len(shap_values.shape) == 2:
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
                
                # Order features by importance (attributions)
                feature_names = [
                    "Transaction Amount", 
                    "Amount to Avg Ratio", 
                    "Transaction Velocity", 
                    "Geographical Distance", 
                    "Device Change", 
                    "Biometric Consistency"
                ]
                
                sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
                
                for idx in sorted_idx:
                    feat_name = feature_names[idx]
                    val, thresh, text = features[feat_name]
                    
                    # If this feature had a strong positive attribution towards fraud
                    if shap_vals[idx] > 0.05:
                        reasons.append(text)
                        
                if reasons:
                    return "Flagged because: " + ", ".join(reasons[:3]) + "."
            except Exception as e:
                # Silent fallback to rule-based explanation
                pass

        # Rule-based fallback explanation
        if avg_ratio > 4.0:
            reasons.append("transaction amount is 4x your historical average")
        if new_device > 0:
            reasons.append("device fingerprint was not recognized")
        if geo_dist > 100.0:
            reasons.append("geographic location changed too fast (impossible travel)")
        if bio_score < 0.4:
            reasons.append("keystroke and mouse dynamics match automated bot speed")
        if amount > 5000.0:
            reasons.append("transaction amount exceeds limit rules")
            
        if not reasons:
            return "Transaction exhibits standard transaction behavior and verified device credentials."
            
        return "Flagged because: " + ", ".join(reasons[:2]) + "."
