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

    def explain_transaction(self, amount, avg_ratio, tx_count, geo_dist, new_device, bio_score, decision="ALLOW"):
        """
        Generates a human-readable explanation and returns the SHAP feature attributions.
        Returns:
            narrative (str): human readable reasoning sentence
            attributions (dict): top 4 feature attributions
        """
        feature_names = [
            "Transaction Amount", 
            "Amount to Avg Ratio", 
            "Transaction Velocity", 
            "Geographical Distance", 
            "Device Change", 
            "Biometric Consistency"
        ]
        
        # Default baseline attributions based on rules if SHAP fails
        default_attribs = {
            "Transaction Amount": 0.0,
            "Amount to Avg Ratio": 0.0,
            "Transaction Velocity": 0.0,
            "Geographical Distance": 0.0,
            "Device Change": 0.0,
            "Biometric Consistency": 0.0
        }
        
        # Estimate attribution weights if SHAP is not loaded
        if avg_ratio > 3.0:
            default_attribs["Amount to Avg Ratio"] = 0.35
        if amount > 500.0:
            default_attribs["Transaction Amount"] = 0.15
        if tx_count > 10:
            default_attribs["Transaction Velocity"] = 0.25
        if geo_dist > 50.0:
            default_attribs["Geographical Distance"] = 0.30
        if new_device > 0.5:
            default_attribs["Device Change"] = 0.20
        if bio_score < 0.5:
            default_attribs["Biometric Consistency"] = 0.35

        features_rules = {
            "Transaction Amount": (amount, 500.0, "high amount"),
            "Amount to Avg Ratio": (avg_ratio, 3.0, "amount is significantly higher than your average"),
            "Transaction Velocity": (tx_count, 10.0, "high frequency of transactions"),
            "Geographical Distance": (geo_dist, 50.0, "location changed rapidly compared to last login"),
            "Device Change": (new_device, 0.5, "using a new device"),
            "Biometric Consistency": (bio_score, 0.5, "unusual or superhuman typing speed/patterns")
        }

        reasons = []
        shap_vals = None

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
                
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                elif len(shap_values.shape) == 3:
                    shap_vals = shap_values[0][0]
                elif len(shap_values.shape) == 2:
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
            except Exception as e:
                pass

        if shap_vals is not None:
            attributions = {feature_names[i]: float(shap_vals[i]) for i in range(len(feature_names))}
            for i, name in enumerate(feature_names):
                val, thresh, text = features_rules[name]
                if shap_vals[i] > 0.05:
                    reasons.append(text)
        else:
            attributions = default_attribs
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

        # Select Top 4 attributions (by absolute magnitude)
        top_4_attributions = dict(
            sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)[:4]
        )

        # Build narrative
        if decision == "ALLOW":
            narrative = "Transaction exhibits standard transaction behavior and verified device credentials."
        else:
            if not reasons:
                # Fallback if no specific trigger reasons were populated
                narrative = "Flagged because: transaction deviates from baseline customer profiles."
            else:
                narrative = "Flagged because: " + ", ".join(reasons[:3]) + "."

        return narrative, top_4_attributions
