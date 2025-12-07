"""
Adaptive OpenCV Watermark Detector (ORB Feature Matching)
---------------------------------------------------------
Advanced fallback system that "learns" from Gemini.
Uses ORB (Oriented FAST and Rotated BRIEF) for robust, invariant detection.
PHASE 2 UPGRADE: Feature Scaling, Balanced RF, Long-Term Memory.
"""

import cv2
import numpy as np
import os
import logging
import uuid
import glob
import pickle
import csv
import threading
import time
import shutil

# Try to import ML libraries
try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False

logger = logging.getLogger("opencv_watermark")

# Define Paths
MODEL_DIR = "models"
TEMPLATE_DIR = os.path.join(MODEL_DIR, "watermark_templates")
POSITIVE_DIR = os.path.join(TEMPLATE_DIR, "positive")
NEGATIVE_DIR = os.path.join(TEMPLATE_DIR, "negative")
DATASET_FILE = os.path.join(MODEL_DIR, "watermark_dataset.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "watermark_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "watermark_scaler.pkl")

# Defined fields to ensure CSV consistency
CSV_FIELDS = [
    "rel_x", "rel_y", "rel_w", "rel_h", "aspect_ratio",
    "std_dev", "edge_density", "orb_kp_count",
    "template_pos_score", "template_neg_score", "label"
]

# Ensure directories exist
os.makedirs(POSITIVE_DIR, exist_ok=True)
os.makedirs(NEGATIVE_DIR, exist_ok=True)

class AsyncTrainer:
    """
    Handles model training in a background thread to avoid blocking the main video pipeline.
    """
    _training_thread = None
    _lock = threading.Lock()
    _is_training = False

    @staticmethod
    def train_in_background():
        """
        Triggers training in a separate thread.
        """
        if not HAS_ML:
            return

        with AsyncTrainer._lock:
            if AsyncTrainer._is_training:
                logger.info("⏳ Training already in progress, skipping new request.")
                return
            AsyncTrainer._is_training = True

        def _train_job():
            try:
                # Delegate to the centralized nightly_retrain script
                # This ensures consistent logic and lock handling
                from scripts import nightly_retrain
                logger.info("🧠 Triggering Background Training via Nightly Retrainer...")
                nightly_retrain.retrain()
                
                # Reload model in main detector after training
                WatermarkDetector.load_model()
                
                # Phase 4: Auto-Cleaning
                WatermarkLearner.clean_templates()

            except Exception as e:
                logger.error(f"❌ Background Training Failed: {e}")
            finally:
                with AsyncTrainer._lock:
                    AsyncTrainer._is_training = False

        AsyncTrainer._training_thread = threading.Thread(target=_train_job, daemon=True)
        AsyncTrainer._training_thread.start()

class WatermarkLearner:
    @staticmethod
    def extract_features(frame: np.ndarray, coords: dict, match_data: dict = None) -> dict:
        """
        Extract numerical features for ML training.
        """
        try:
            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
            h_img, w_img = frame.shape[:2]
            
            # Geometric Features
            rel_x = x / w_img
            rel_y = y / h_img
            rel_w = w / w_img
            rel_h = h / h_img
            aspect_ratio = w / h if h > 0 else 0
            
            # Visual Features (Texture/Complexity)
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: return None
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(gray_roi)
            edge_density = np.mean(cv2.Canny(gray_roi, 100, 200)) / 255.0
            
            # ORB Features
            orb_kp_count = match_data.get("kp_count", 0) if match_data else 0
            template_pos_score = match_data.get("template_pos_score", 0) if match_data else 0
            template_neg_score = match_data.get("template_neg_score", 0) if match_data else 0
            
            features = {
                "rel_x": rel_x, "rel_y": rel_y, "rel_w": rel_w, "rel_h": rel_h,
                "aspect_ratio": aspect_ratio,
                "std_dev": std_dev, "edge_density": edge_density,
                "orb_kp_count": orb_kp_count,
                "template_pos_score": template_pos_score,
                "template_neg_score": template_neg_score
            }
            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None



    @staticmethod
    def log_feedback(features: dict, label: int):
        """
        Log features and label (1=Watermark, 0=False Positive) to CSV.
        """
        if not features: return
        
        file_exists = os.path.exists(DATASET_FILE)
        
        try:
            with open(DATASET_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                
                row = features.copy()
                row["label"] = label
                writer.writerow(row)
                
            logger.info(f"📝 Logged feedback to dataset (Label: {label})")
            
            if HAS_ML:
                # OPTIMIZATION: Don't train immediately. Let Main Loop handle it.
                # AsyncTrainer.train_in_background()
                pass
                
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")

    @staticmethod
    def save_template(frame: np.ndarray, coords: dict, is_positive: bool = True):
        """
        Save a detection as a template AND log features.
        """
        try:
            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
            
            if w < 20 or h < 20: return
            
            # 1. Save Template Image
            roi = frame[y:y+h, x:x+w]
            label_str = "pos" if is_positive else "neg"
            target_dir = POSITIVE_DIR if is_positive else NEGATIVE_DIR
            
            filename = f"{label_str}_{uuid.uuid4().hex[:8]}.png"
            path = os.path.join(target_dir, filename)
            cv2.imwrite(path, roi)
            
            # 2. Log Features for ML
            # We assume high confidence for manual feedback
            match_data = {
                "kp_count": 50, 
                "template_pos_score": 1.0 if is_positive else 0.0,
                "template_neg_score": 1.0 if not is_positive else 0.0
            }
            features = WatermarkLearner.extract_features(frame, coords, match_data=match_data)
            if features:
                WatermarkLearner.log_feedback(features, 1 if is_positive else 0)
            
            # 3. Update ORB Database
            WatermarkDetector.train()
            
            logger.info(f"🧠 Learned new {'POSITIVE' if is_positive else 'NEGATIVE'} template: {filename}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to learn template: {e}")

    @staticmethod
    def clean_templates():
        """
        Phase 4: Auto-Cleaning. Keep only most recent 150 templates if > 200.
        """
        try:
            for d in [POSITIVE_DIR, NEGATIVE_DIR]:
                files = sorted(glob.glob(os.path.join(d, "*.png")), key=os.path.getmtime)
                if len(files) > 200:
                    to_remove = files[:-150] # Remove oldest, keep recent 150
                    for f in to_remove:
                        os.remove(f)
                    logger.info(f"🧹 Auto-Cleaned {len(to_remove)} old templates from {d}")
        except Exception as e:
            logger.warning(f"Template cleanup failed: {e}")

class WatermarkDetector:
    _orb = None
    _bf = None
    _positive_features = [] 
    _negative_features = []
    _model = None
    _scaler = None
    _model_lock = threading.Lock()
    
    @classmethod
    def init(cls):
        if cls._orb is None:
            cls._orb = cv2.ORB_create(nfeatures=1000)
            cls._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            cls.load_features()
            cls.load_model()

    @classmethod
    def load_model(cls):
        """Safe model loading with lock"""
        if os.path.exists(MODEL_FILE):
            try:
                with cls._model_lock:
                    with open(MODEL_FILE, 'rb') as f:
                        cls._model = pickle.load(f)
                    if os.path.exists(SCALER_FILE):
                        with open(SCALER_FILE, 'rb') as f:
                            cls._scaler = pickle.load(f)
                logger.info("🤖 Loaded/Reloaded ML Watermark Classifier & Scaler")
            except: pass

    @classmethod
    def train(cls):
        """
        Re-build feature database from all templates.
        """
        cls.init()
        
        def load_from_dir(directory):
            feats = []
            templates = glob.glob(os.path.join(directory, "*.png"))
            for t_path in templates:
                img = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                kp, des = cls._orb.detectAndCompute(img, None)
                if des is not None and len(kp) > 5:
                    feats.append((kp, des, img.shape))
            return feats

        cls._positive_features = load_from_dir(POSITIVE_DIR)
        cls._negative_features = load_from_dir(NEGATIVE_DIR)

    @classmethod
    def load_features(cls):
        cls.train()

    @staticmethod
    def detect(frame: np.ndarray, override_threshold: float = None) -> list:
        """
        Detect watermark using ORB + ML Validation.
        """
        WatermarkDetector.init()
        if not WatermarkDetector._positive_features:
            return None
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = WatermarkDetector._orb.detectAndCompute(gray_frame, None)
        
        if des_frame is None: return None
        
        # 1. Find All Positive Matches
        matches_list = []
        
        for (kp_template, des_template, t_shape) in WatermarkDetector._positive_features:
            if des_template is None: continue
            
            matches = WatermarkDetector._bf.match(des_template, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            
            score = len(good_matches)
            
            if score > 3: # Threshold (Emergency Low)
                src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = t_shape[:2]
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    x_min = int(np.min(dst[:, :, 0]))
                    y_min = int(np.min(dst[:, :, 1]))
                    x_max = int(np.max(dst[:, :, 0]))
                    y_max = int(np.max(dst[:, :, 1]))
                    
                    w_rect = x_max - x_min
                    h_rect = y_max - y_min
                    
                    if x_min >= 0 and y_min >= 0 and x_max < frame.shape[1] and y_max < frame.shape[0]:
                         if w_rect > 10 and h_rect > 5:
                             match_cand = {
                                'x': x_min, 'y': y_min, 'w': w_rect, 'h': h_rect,
                                'confidence': min(1.0, score / max(len(kp_template), 20)),
                                'orb_matches': score,
                                'template_pos_score': score,
                                'template_neg_score': 0 # Will verify later
                            }
                             matches_list.append(match_cand)

        # Apply NMS (Non-Maximum Suppression) to merge overlaps
        unique_matches = WatermarkDetector._nms(matches_list)
        
        final_matches = []
        for match in unique_matches:
            # 2. Check against Negative Templates
            best_neg_score = 0
            roi = gray_frame[match['y']:match['y']+match['h'], 
                             match['x']:match['x']+match['w']]
            
            if roi.size > 0:
                kp_roi, des_roi = WatermarkDetector._orb.detectAndCompute(roi, None)
                if des_roi is not None: 
                    for (kp_neg, des_neg, _) in WatermarkDetector._negative_features:
                        if des_neg is None: continue
                        neg_matches = WatermarkDetector._bf.match(des_neg, des_roi)
                        neg_good = [m for m in neg_matches if m.distance < 50]
                        if len(neg_good) > best_neg_score:
                            best_neg_score = len(neg_good)
            
            match['template_neg_score'] = best_neg_score

            # 3. ML Model Validation
            rf_conf = 0.0
            with WatermarkDetector._model_lock:
                model = WatermarkDetector._model
                scaler = WatermarkDetector._scaler
                
            if model and scaler:
                match_data = {
                    "kp_count": len(kp_frame),
                    "template_pos_score": match['template_pos_score'],
                    "template_neg_score": best_neg_score
                }
                features = WatermarkLearner.extract_features(frame, match, match_data)
                
                if features:
                    try:
                        import pandas as pd
                        X_df = pd.DataFrame([features])
                        X_scaled = scaler.transform(X_df)
                        
                        probs = model.predict_proba(X_scaled)[0]
                        rf_conf = probs[1]
                        match['rf_conf'] = rf_conf
                        
                        threshold = 0.35
                        if override_threshold is not None:
                            threshold = override_threshold
                        
                        if rf_conf < threshold:
                             logger.warning(f"🛑 Match rejected by ML Model (Conf: {rf_conf:.2f} < {threshold})")
                             continue # Skip this match
                    except Exception: pass
            
            final_matches.append(match)
            
        # Merging Logic:
        # If we are in "Emergency Mode" (override_threshold set), we combine ORB + MSER
        # to ensure we catch text even if ORB only finds a logo fragment.
        mser_boxes = []
        if override_threshold is not None or not final_matches:
             mser_boxes = WatermarkDetector._mser_fallback(frame)
             
        if mser_boxes:
            final_matches.extend(mser_boxes)

        if final_matches:
            return final_matches
            
        # Fallback to MSER for unknown text watermarks (already ran above if override set)
        return []

    @staticmethod
    def _mser_fallback(frame: np.ndarray) -> list:
        """
        Fallback: Use MSER to find text-like regions in the bottom 20% of the screen.
        """
        try:
            h, w = frame.shape[:2]
            roi_y = int(h * 0.8) # Bottom 20%
            roi = frame[roi_y:, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            mser = cv2.MSER_create(_max_area=2000, _min_area=30)
            regions, _ = mser.detectRegions(gray)
            
            boxes = []
            for p in regions:
                x_r, y_r, w_r, h_r = cv2.boundingRect(p)
                # Filter for text-like aspect ratios
                aspect = w_r / h_r
                if aspect > 0.5 and aspect < 5.0: # Letters or small words
                     boxes.append([x_r, y_r, x_r + w_r, y_r + h_r])
            
            if not boxes: return []
            
            boxes = np.array(boxes)
            
            # Return bounding box of ALL text regions found
            # (Aggressive approach for bottom-corner text)
            x_min = np.min(boxes[:, 0])
            y_min = np.min(boxes[:, 1])
            x_max = np.max(boxes[:, 2])
            y_max = np.max(boxes[:, 3])
                 
            # Add padding
            pad_x = 10
            pad_y = 5
                 
            final_w = (x_max - x_min) + 2*pad_x
            final_h = (y_max - y_min) + 2*pad_y
            final_x = x_min - pad_x
            final_y = y_min - pad_y + roi_y # Adjust to global frame
                 
            # Clamp
            final_x = max(0, final_x)
            final_y = max(0, final_y)
            final_w = min(w - final_x, final_w)
            final_h = min(h - final_y, final_h)
                 
            if final_w > 20 and final_h > 10:
                 logger.info("📝 MSER Fallback found text region")
                 return [{
                    'x': int(final_x), 'y': int(final_y), 'w': int(final_w), 'h': int(final_h),
                    'confidence': 0.5, # Lower confidence for fallback
                    'label': 'MSER_TEXT',
                    'rf_conf': 0.0
                 }]
        except Exception:
            pass
        return []

    @staticmethod
    def _nms(boxes, overlap_thresh=0.3):
        if not boxes: return []
        
        # Convert to list of [x, y, x2, y2, score]
        proposals = []
        for b in boxes:
            proposals.append([b['x'], b['y'], b['x']+b['w'], b['y']+b['h'], b['confidence']])
            
        proposals = np.array(proposals)
        pick = []
        
        x1 = proposals[:,0]
        y1 = proposals[:,1]
        x2 = proposals[:,2]
        y2 = proposals[:,3]
        scores = proposals[:,4]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
            
        return [boxes[i] for i in pick]

# Helper functions
def learn(frame, coords, is_positive=True):
    WatermarkLearner.save_template(frame, coords, is_positive)

def detect(frame, override_threshold=None):
    return WatermarkDetector.detect(frame, override_threshold=override_threshold)

def refine_roi(frame: np.ndarray, roi: dict) -> dict:
    """
    Refine the ROI using OpenCV contour analysis to find the tightest box around content.
    """
    try:
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Extract the ROI
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0: return roi
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Thresholding (Otsu + Adaptive Fallback)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # FALLBACK: If Otsu fails (empty), try Adaptive Threshold (for faint text)
        if not contours:
             binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
             contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: return roi
        
        # Find the bounding box of all contours
        min_x, min_y = w, h
        max_x, max_y = 0, 0
        
        found_content = False
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            # Filter noise (Keep small text parts)
            if cw * ch > 15: 
                found_content = True
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx + cw)
                max_y = max(max_y, cy + ch)
                
        if found_content:
            # Map back to global coordinates
            refined_x = x + min_x
            refined_y = y + min_y
            refined_w = max_x - min_x
            refined_h = max_y - min_y
            
            # Add small padding 
            # ACCURACY: Reduced from 2 to 1 for generic tight fit.
            pad = 1
            refined_x = max(0, refined_x - pad)
            refined_y = max(0, refined_y - pad)
            refined_w += 2*pad
            refined_h += 2*pad
            
            logger.info(f"👁️ OpenCV Refined Box: {refined_x},{refined_y} {refined_w}x{refined_h}")
            return {"x": refined_x, "y": refined_y, "w": refined_w, "h": refined_h}
            
    except Exception as e:
        logger.warning(f"OpenCV refinement failed: {e}")
        
    return roi

def get_watermark_mask(frame: np.ndarray, roi: dict, watermark_type: str = "TEXT_WHITE") -> np.ndarray:
    """
    Extracts the binary mask of the watermark content within the ROI.
    Returns a mask of the full frame size (0=Background, 255=Watermark).
    Uses Contour Filling to ensure no holes in the letters.
    """
    try:
        h_img, w_img = frame.shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        
        x, y, w, h = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])
        
        # Valid crop check
        if w <= 0 or h <= 0 or x < 0 or y < 0: return mask
        
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0: return mask
        
        # Convert to HSV to separate Brightness (V)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hue, s_sat, v_val = cv2.split(hsv) 
        
        binary = None

        # STRATEGY A: Explicit White Text
        if watermark_type == "TEXT_WHITE":
            # 1. Luminance AND Saturation Thresholding
            # PROBLEM: Bright Skin/Gold Dress passes the "Brightness" check (V > 225), creating ugly blobs.
            # SOLUTION: White Text is Achromatic (Low Saturation). Skin/Gold is Chromatic (High Saturation).
            
            # A. Brightness Mask (Keep Bright stuff)
            _, mask_v = cv2.threshold(v_val, 180, 255, cv2.THRESH_BINARY)
            
            # B. Saturation Mask (Keep Colorless stuff)
            # Relaxed to < 80 to catch slightly tinted text
            _, s_high = cv2.threshold(s_sat, 80, 255, cv2.THRESH_BINARY)
            mask_s_low = cv2.bitwise_not(s_high)
            
            # C. Combine: Bright AND Colorless = White Text
            binary = cv2.bitwise_and(mask_v, mask_v, mask=mask_s_low)
        
        # STRATEGY B: Explicit Colored/Transparent Logo (Force Edge Detection)
        # Or Fallback if Strategy A failed.
        if binary is None or cv2.countNonZero(binary) < 10:
            logger.info("   🎨 White-Mask Empty. Switching to Universal Edge-Mask (for Colored/Dark logos).")
            # 1. Edge Detection (Finds structure of any color)
            # Use lower threshold to catch faint text
            edges = cv2.Canny(crop, 50, 150) 
            
            # 2. Connect the Edges to form a solid blob
            # We dilate more aggressively here because edges are thin
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dilated_edges = cv2.dilate(edges, kernel_edge, iterations=2)
            
            # 3. Fill the blobs
            # We treat the dilated edges as the "content"
            binary = dilated_edges
            
            # (Optional) We could find contours and fill holes, 
            # but dilated edges are usually enough for a blur mask.
        
        # 2. Sparkle Filter (Contour Area)
        # Remove small disconnected blobs (sparkles).
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 15:
                cv2.drawContours(binary, [cnt], -1, 0, -1) # Erase sparkle
        
        # 3. Dilation to connect letters (Minimal)
        # Reduced to 1 iteration to match 'Black Dress' precision.
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        # 3. Fill Holes: Find Contours and Draw Filled
        # We process the crop-sized mask first
        mask_crop = np.zeros((h, w), dtype=np.uint8)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw filled contours (White)
        cv2.drawContours(mask_crop, contours, -1, (255), thickness=cv2.FILLED)
        
        # 4. Final Small Safety Dilation
        # Just 1 iteration to smooth edges, not expand wildly.
        mask_crop = cv2.dilate(mask_crop, kernel_dilate, iterations=1)
        
        # Place into full size mask
        mask[y:y+h, x:x+w] = mask_crop
        
        return mask
        
    except Exception as e:
        logger.warning(f"Failed to extract watermark mask: {e}")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

def inpaint_video(video_path, mask_paths, output_path):
    """
    Powerful Inpainting: Uses Navier-Stokes/Telea to strictly replace pixels 
    inside the mask with plausible texture from surroundings.
    Supports MULTIPLE masks (batch inpainting).
    """
    if isinstance(mask_paths, str):
        mask_paths = [mask_paths]
        
    logger.info(f"🎨 Starting Neural-Style Inpainting (Telea) for {os.path.basename(video_path)} with {len(mask_paths)} masks...")
    
    cap = cv2.VideoCapture(video_path)
    cap_masks = [cv2.VideoCapture(m) for m in mask_paths]
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use MP4V for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Write to a TEMP file first (Silent)
    temp_silent_path = output_path.replace(".mp4", "_silent.mp4")
    out = cv2.VideoWriter(temp_silent_path, fourcc, fps, (w, h))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Combine all masks
        combined_mask = None
        
        for cap_mask in cap_masks:
            ret_m, mask_frame = cap_mask.read()
            if not ret_m: continue # Should generally match frame count
            
            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
            
            if combined_mask is None:
                combined_mask = mask_bin
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask_bin)
        
        if combined_mask is not None and cv2.countNonZero(combined_mask) > 0:
            # DILATE MASK slightly to cover edges
            # Optimization: 1 iteration is usually enough for 5x5 kernel
            kernel = np.ones((5,5), np.uint8)
            mask_dilated = cv2.dilate(combined_mask, kernel, iterations=1)
            
            # INPAINT: Radius 3 is faster and sharper for text
            res = cv2.inpaint(frame, mask_dilated, 3, cv2.INPAINT_TELEA)
        else:
            res = frame
        
        out.write(res)
        frame_count += 1
        
        if frame_count % 100 == 0:
            logger.info(f"    🖌️ Inpainted {frame_count}/{total_frames} frames...")
            
    cap.release()
    for cm in cap_masks: cm.release()
    out.release()
    
    # RESTORE AUDIO
    logger.info("    🔇 Merging Audio back into Inpainted Video...")
    import subprocess
    try:
        # Use simple map to copy audio from source
        # Note: 'ffmpeg' might need full path if not in path, but usually 'ffmpeg' works if installed.
        cmd_mux = [
            'ffmpeg', "-y",
            "-i", temp_silent_path,
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        # Run silently
        subprocess.run(cmd_mux, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Cleanup
        if os.path.exists(temp_silent_path):
            os.remove(temp_silent_path)
            
        logger.info("✅ Inpainting Complete (Audio Restored).")
        return True
        
    except Exception as e:
        logger.error(f"❌ Audio Mux Failed: {e}. Final video might be silent.")
        # Fallback: Just rename silent to output
        if os.path.exists(temp_silent_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_silent_path, output_path)
        return True
