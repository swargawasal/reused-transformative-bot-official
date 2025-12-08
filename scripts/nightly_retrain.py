"""
Nightly Auto-Retraining Script
------------------------------
Runs automatically to improve the Watermark Brain.
1. Checks lock file.
2. Loads data and Validates against Static Schema (`models/feature_schema.json`).
3. Trains Balanced Random Forest.
4. Checks Regression.
5. Logs Feature Importance.
6. Atomically updates model.
"""

import os
import sys
import time
import json
import pickle
import logging
import shutil
from datetime import datetime

# Add parent dir to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    print("❌ ML libraries missing. Skipping retrain.")
    sys.exit(0)

from ml_logger import log_feature_importance



# Config
MODEL_DIR = "models"
DATASET_FILE = os.path.join(MODEL_DIR, "watermark_dataset.csv")
SCHEMA_FILE = os.path.join(MODEL_DIR, "feature_schema.json")
MODEL_FILE = os.path.join(MODEL_DIR, "watermark_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "watermark_scaler.pkl")
LOG_DIR = "logs"
LOCK_FILE = os.path.join(MODEL_DIR, "retrain.lock")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, f"retrain_{datetime.now().strftime('%Y%m%d')}.log"), level=logging.INFO)
logger = logging.getLogger("nightly_retrain")

def retrain():
    if os.path.exists(LOCK_FILE):
        # Check if stale (older than 1 hour)
        if time.time() - os.path.getmtime(LOCK_FILE) > 3600:
            os.remove(LOCK_FILE)
        else:
            print(json.dumps({"status": "skipped", "reason": "locked"}))
            return

    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))

    try:
        logger.info("🚀 Starting Nightly Retrain...")
        
        if not os.path.exists(DATASET_FILE):
            print(json.dumps({"status": "skipped", "reason": "no_data"}))
            return

        # 1. LOAD SCHEMA
        if not os.path.exists(SCHEMA_FILE):
             logger.error(f"❌ Critical Schema Missing: {SCHEMA_FILE}")
             return
             
        with open(SCHEMA_FILE, 'r') as f:
            schema_def = json.load(f)
            expected_features = schema_def.get("features", [])

        if not expected_features:
             logger.error("❌ Schema has no features defined!")
             return

        # 2. LOAD DATA
        try:
            # Pandas 1.3+ uses on_bad_lines
            if int(pd.__version__.split('.')[0]) >= 1 and int(pd.__version__.split('.')[1]) >= 3:
                 df = pd.read_csv(DATASET_FILE, on_bad_lines='skip')
            else:
                 df = pd.read_csv(DATASET_FILE, error_bad_lines=False)
        except Exception:
             df = pd.read_csv(DATASET_FILE, on_bad_lines='skip')
        
        # 3. VALIDATE SCHEMA
        obs_cols = set(df.columns)
        missing_feats = [f for f in expected_features if f not in obs_cols]
        
        if missing_feats:
            logger.error(f"❌ Dataset missing required features: {missing_feats}")
            print(json.dumps({"status": "skipped", "reason": "schema_mismatch_missing", "missing": missing_feats}))
            return
            
        if 'label' not in obs_cols:
            logger.error("❌ Dataset missing 'label' column.")
            return

        # STRICT: Check for EXTRA columns (Ghost features prevention)
        extra_cols = list(obs_cols - set(expected_features) - {'label'})
        if extra_cols:
             logger.error(f"❌ Dataset has extra/unknown columns: {extra_cols}")
             print(json.dumps({"status": "skipped", "reason": "schema_mismatch_extra", "extra": extra_cols}))
             return

        # Patch 2: Safe Schema Migration (Missing Features)
        missing_feats = [f for f in expected_features if f not in obs_cols]
        use_features = expected_features[:] # Copy
        
        if missing_feats:
            missing_count = len(missing_feats)
            msg = f"⚠️ Dataset missing {missing_count} features: {missing_feats}"
            logger.warning(msg)
            
            # Migration Policies
            if missing_count <= 3:
                logger.info("✅ Partial Training Allowed (Missing <= 3). Dropping missing columns.")
                # We train on intersection
                use_features = [f for f in expected_features if f not in missing_feats]
            else:
                logger.error("❌ Too many missing features (>3). Aborting Retrain.")
                print(json.dumps({"status": "skipped", "reason": "schema_mismatch_missing", "missing": missing_feats}))
                return

        # Log Report
        migration_report = {
             "timestamp": datetime.now().isoformat(),
             "expected": expected_features,
             "observed": list(obs_cols),
             "missing": missing_feats,
             "used": use_features,
             "action": "partial_train" if missing_feats else "full_train"
        }
        with open(os.path.join(LOG_DIR, "schema_migration_report.json"), "w") as f:
            json.dump(migration_report, f, indent=2)

        # Basic Data Check
        if len(df) < 20: 
            print(json.dumps({"status": "skipped", "reason": "insufficient_data", "count": len(df)}))
            return
            
        # Ensure we only use schema features + label
        # use_cols = expected_features + ['label'] -> CHANGED via migration logic
        use_cols = use_features + ['label']

        df = df[use_cols].dropna() # Drop rows with NaNs
        
        if len(df['label'].unique()) < 2:
            logger.warning("⚠️ Training skipped: Only one class present in data.")
            return

        X = df[use_features] # Use the migrated features list
        y = df['label']

        
        # Split for Validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. SCALING
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 5. TRAIN
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # 6. VALIDATE & LOG FEATURE IMPORTANCE
        val_acc = model.score(X_val_scaled, y_val)
        logger.info(f"📊 Validation Accuracy: {val_acc:.2%}")
        
        # Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        logger.info("🌟 Top 5 Predictive Features:")
        for i in range(min(5, len(expected_features))):
            idx = indices[i]
            logger.info(f"   {i+1}. {expected_features[idx]}: {importances[idx]:.4f}")
        
        # Log to JSON (Patch 1)
        log_feature_importance(model, expected_features)

        
        # 7. REGRESSION CHECK
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    old_model = pickle.load(f)
                
                if hasattr(old_model, "predict"):
                    # Test old model on NEW validation set
                    # Note: We assume old model expects same features. 
                    # If schema changed, old model will crash/misbehave.
                    # We wrap in try block.
                    try:
                        old_acc = old_model.score(X_val_scaled, y_val)
                        logger.info(f"📉 Previous Model Accuracy: {old_acc:.2%}")
                        
                        if val_acc < (old_acc - 0.01): # 1% Tolerance
                            logger.warning(f"⚠️ Regression detected! New ({val_acc:.2%}) < Old ({old_acc:.2%}). Skipping update.")
                            print(json.dumps({"status": "skipped_due_to_regression", "new_acc": val_acc, "old_acc": old_acc}))
                            return
                    except Exception:
                        logger.warning("⚠️ Could not test old model (schema change?). Proceeding with update.")
            except Exception as e:
                logger.warning(f"⚠️ Could not load old model: {e}")

        # 8. SAVE ATOMICALLY
        timestamp = int(time.time())
        new_model_path = os.path.join(MODEL_DIR, f"watermark_model_v{timestamp}.pkl")
        
        with open(new_model_path, 'wb') as f:
            pickle.dump(model, f)
            
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
            
        shutil.copy2(new_model_path, "temp_model_swap.pkl")
        os.replace("temp_model_swap.pkl", MODEL_FILE)
        
        # Cleanup
        models = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("watermark_model_v")])
        if len(models) > 5:
            for m in models[:-5]:
                try: os.remove(os.path.join(MODEL_DIR, m))
                except: pass
                
        result = {
            "retrain_ok": True,
            "new_model": new_model_path,
            "val_acc": val_acc,
            "samples": len(df)
        }
        print(json.dumps(result))
        logger.info("✅ Retrain Complete.")

    except Exception as e:
        logger.error(f"❌ Retrain Failed: {e}")
        print(json.dumps({"status": "failed", "error": str(e)}))
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

if __name__ == "__main__":
    retrain()
