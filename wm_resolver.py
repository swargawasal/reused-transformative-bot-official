"""
Watermark Resolver (The Logic Core)
-----------------------------------
PHASE 3: Moving Watermark Tracking (M-Tracker)
PHASE 5: Anti False-Positive Defense
PHASE 6: Super-Stable Multi-Frame Refinement (SSR-3)
"""

import cv2
import numpy as np
import logging
from collections import deque

logger = logging.getLogger("wm_resolver")

class WatermarkTracker:
    def __init__(self):
        self.history = deque(maxlen=3) # SSR-3 Window
        self.last_box = None
        self.consecutive_hits = 0
        self.consecutive_misses = 0

    def compute_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA['x'], boxB['x'])
        yA = max(boxA['y'], boxB['y'])
        xB = min(boxA['x'] + boxA['w'], boxB['x'] + boxB['w'])
        yB = min(boxA['y'] + boxA['h'], boxB['y'] + boxB['h'])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = boxA['w'] * boxA['h']
        boxBArea = boxB['w'] * boxB['h']

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, new_box):
        """
        Updates the tracker with a new candidate box.
        Returns the refined, stabilized box or None if unstable.
        """
        if new_box is None:
            self.consecutive_misses += 1
            self.consecutive_hits = 0
            if self.consecutive_misses > 2:
                self.history.clear()
                self.last_box = None
            return None

        # PREVENT MUTATION: Shallow copy the input dictionary
        new_box = dict(new_box)

        # Phase 5: Anti False-Positive Defense
        if not self._passes_safety_checks(new_box):
            logger.warning("🛡️ Box rejected by Anti-FP Defense")
            return None

        # Phase 3: Motion Smoothing
        if self.last_box:
            iou = self.compute_iou(self.last_box, new_box)
            if iou < 0.50:
                logger.info("💨 Fast moving watermark detected")
                # Local search logic could go here, but we assume upstream detector did its best
            
            # Exponential Smoothing
            smooth_x = int(0.7 * self.last_box['x'] + 0.3 * new_box['x'])
            smooth_y = int(0.7 * self.last_box['y'] + 0.3 * new_box['y'])
            smooth_w = int(0.5 * self.last_box['w'] + 0.5 * new_box['w'])
            smooth_h = int(0.5 * self.last_box['h'] + 0.5 * new_box['h'])
            
            new_box['x'], new_box['y'] = smooth_x, smooth_y
            new_box['w'], new_box['h'] = smooth_w, smooth_h

        self.history.append(new_box)
        self.last_box = new_box
        self.consecutive_hits += 1
        self.consecutive_misses = 0

        # Phase 6: SSR-3 (Super-Stable Refinement)
        if len(self.history) == 3:
            final_x = int(np.median([b['x'] for b in self.history]))
            final_y = int(np.median([b['y'] for b in self.history]))
            final_w = int(np.median([b['w'] for b in self.history]))
            final_h = int(np.median([b['h'] for b in self.history]))
            
            refined_box = {
                'x': final_x, 'y': final_y, 'w': final_w, 'h': final_h,
                'confidence': new_box.get('confidence', 0),
                'rf_conf': new_box.get('rf_conf', 0)
            }
            return refined_box
        
        return None # Wait for 3 frames to stabilize

    def _passes_safety_checks(self, box):
        """
        Phase 5: Anti False-Positive Defense Rules
        """
        # We need features for this. 
        # If box comes from opencv_watermark, it might have them.
        # If not, we might need to re-extract or trust the source.
        # For now, we check geometric properties.
        
        aspect_ratio = box['w'] / box['h'] if box['h'] > 0 else 0
        if aspect_ratio > 12:
            logger.warning(f"🚫 Rejected: Aspect Ratio {aspect_ratio:.1f} > 12")
            return False
            
        # Additional checks would require image access (std_dev, edge_density)
        # We assume opencv_watermark.detect() already filtered low confidence ones
        # or we check 'rf_conf' if available.
        
        if 'rf_conf' in box and box['rf_conf'] < 0.40:
            return False
            
        return True

class Resolver:
    def __init__(self):
        self.tracker = WatermarkTracker()

    def resolve(self, frame, candidate_box):
        """
        Main entry point for resolving watermark for a frame.
        """
        # 1. Refine Geometry (Stage A - if needed, usually done by detector)
        
        # 2. Track and Stabilize
        final_box = self.tracker.update(candidate_box)
        
        # 3. Final Decision
        if final_box:
            # Hard-Stabilize Output Rule
            # smoothed IoU check is implicit in tracker history
            # Check confidence thresholds
            rf_conf = final_box.get('rf_conf', 0)
            orb_matches = final_box.get('orb_matches', 0)
            
            if rf_conf >= 0.55 or orb_matches >= 4:
                return final_box
            else:
                logger.info(f"⚠️ Unstable/Low Conf: RF={rf_conf:.2f}, ORB={orb_matches}")
                return None
                
        return None
