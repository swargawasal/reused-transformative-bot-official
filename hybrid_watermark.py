
"""
Hybrid Video Vision Orchestrator
--------------------------------
Controls 4 modules:
1. Gemini Vision (if enabled)
2. OpenCV Template Matching
3. ORB Feature Matching
4. RandomForest Watermark Brain (Active Learning)

Outputs STRICT JSON rules for the pipeline.
"""

import cv2
import numpy as np
import os
import logging
import json
import uuid
import time
import shutil
from typing import Dict, Optional, List

# Import our modules
import opencv_watermark
from wm_resolver import Resolver

logger = logging.getLogger("hybrid_watermark")

class HybridOrchestrator:
    def __init__(self):
        self.resolver = Resolver()
        self.gemini_enabled = os.getenv("ENABLE_GEMINI_WATERMARK_DETECT", "yes").lower() == "yes"
        self.temporal_enabled = os.getenv("ENABLE_TEMPORAL_SEMANTIC", "yes").lower() == "yes"
        self.self_improve = os.getenv("ENABLE_SELF_IMPROVE", "yes").lower() == "yes"
        
        # Load settings
        self.max_area_pct = float(os.getenv("WATERMARK_MAX_AREA_PERCENT", "20"))
        self.force_no_crop = os.getenv("FORCE_NO_CROP", "yes").lower() == "yes"
        self.my_wm_text = os.getenv("MY_WATERMARK_TEXT", "swargawasal")
        self.two_step_verification = os.getenv("WATERMARK_2STEP_VERIFICATION", "off").lower() == "on"
        
    def _extract_frame(self, video_path: str, pct: float = 0.2):
        """Extracts a representative frame from the video at a specific percentage."""
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            target_frame = int(total_frames * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Could not read video frame from {video_path} at {pct*100}%")
            return None
        return frame

    def _extract_keywords(self, video_path: str) -> Optional[str]:
        """Extracts dynamic watermark keywords from video metadata."""
        keywords = None
        try:
            meta_path = os.path.splitext(video_path)[0] + '.json'
            if os.path.exists(meta_path):
                import re
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            else:
                meta = {}
                
            kw_list = []
            # Uploader
            if meta.get('uploader'):
                u = str(meta['uploader'])
                kw_list.append(u)
                kw_list.append(f"@{u}")
                kw_list.append(u.replace(" ", ""))
                
            # Uploader ID
            if meta.get('uploader_id'):
                uid = str(meta['uploader_id'])
                kw_list.append(uid)
                
            # URL Handle Extraction (e.g. instagram.com/username)
            url = meta.get('webpage_url', '')
            if url:
                # Try to find username in URL
                # Instagram: instagram.com/username/
                # YouTube: youtube.com/@username
                handle_match = re.search(r'(?:instagram\.com|youtube\.com)/(?:@)?([a-zA-Z0-9_\.]+)', url)
                if handle_match:
                    handle = handle_match.group(1)
                    if handle not in ['reel', 'p', 'shorts', 'watch']: # Ignore common paths
                        kw_list.append(handle)
                        kw_list.append(f"@{handle}")
            
            # Caption tags/mentions
            if meta.get('caption'):
                tags = re.findall(r'[#@]([\w\.]+)', str(meta['caption']))
                kw_list.extend(tags)
                
            if meta.get('tags'):
                kw_list.extend(meta['tags'])
                
            # Deduplicate and join
            if kw_list:
                # Filter short keywords
                kw_list = [k for k in kw_list if len(k) > 3]
                keywords = ", ".join(list(set(kw_list))[:10]) # Limit to 10 unique keywords
                
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {e}")
            
        return keywords

    def process_video(self, video_path: str, aggressive: bool = False) -> str:
        """
        Main entry point. Returns JSON result.
        """
        try:
            logger.info(f"🎬 Processing video for watermark: {video_path} (Aggressive: {aggressive})")
            
            # 1. Multi-Frame Detection Strategy
            # Emergency: Full scan always to catch tricky watermarks
            checkpoints = [0.1, 0.5, 0.9]
            all_candidates = []
            frame = None
            
            # Extract Keywords once
            keywords = self._extract_keywords(video_path)
            if keywords:
                logger.info(f"   🔍 Dynamic Watermark Keywords: {keywords}...")

            # Get Video Duration
            video_duration = 0.0
            try:
                cap_temp = cv2.VideoCapture(video_path)
                if cap_temp.isOpened():
                    fps_temp = cap_temp.get(cv2.CAP_PROP_FPS)
                    frames_temp = cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps_temp > 0:
                        video_duration = frames_temp / fps_temp
                cap_temp.release()
            except: pass

            for cp in checkpoints:
                logger.info(f"   Scanning frame at {cp*100:.0f}%...")
                current_frame = self._extract_frame(video_path, pct=cp)
                if current_frame is None: continue
                
                if frame is None: frame = current_frame # Keep first valid frame for context
                
                # Reverted: Using full resolution frame as requested
                detect_frame = current_frame
                scale_ratio = 1.0

                # Gemini Detection (List)
                gemini_boxes = []
                gemini_failed = False
                if self.gemini_enabled:
                    try:
                        import gemini_enhance
                        gemini_boxes = gemini_enhance.detect_watermark(detect_frame, keywords=keywords) or []
                        if not gemini_boxes: gemini_failed = True 
                    except Exception: 
                        gemini_failed = True

                # OpenCV Detection (List)
                cv_boxes = opencv_watermark.detect(detect_frame, override_threshold=0.15 if gemini_failed else None) or []
                
                # Fuse for this frame
                frame_candidates = self._fuse_results(cv_boxes, gemini_boxes)
                if frame_candidates:
                    for c in frame_candidates:
                        # JUNK FILTER: Immediately discard extremely low confidence detections (<15%)
                        # This prevents "Scanner Spam" where the bot loops through 15+ garbage candidates.
                        conf = c.get('confidence', 0)
                        rf_conf = c.get('rf_conf', 0)
                        
                        if conf < 0.15 and rf_conf < 0.15:
                            continue

                        # Add timestamp context to candidate
                        c['detected_at_pct'] = cp
                        c['gemini_failed'] = gemini_failed
                        all_candidates.append(c)
            
            if frame is None:
                return self._error_json("Failed to extract any frames")

            # Fuse across all frames (Temporal NMS)
            # We treat detections at different times as the SAME watermark if they overlap
            unique_watermarks = self._temporal_nms(all_candidates)
            
            final_watermarks = []
            
            for i, candidate_box in enumerate(unique_watermarks):
                scan_pos = candidate_box.get('detected_at_pct', 0) * 100
                logger.info(f"🔄 Processing Watermark Candidate #{i+1}: {candidate_box} (Found at {scan_pos:.0f}% of video)")
                
                # STAGE 1: VALIDATION
                is_valid = True # Default trust if Gemini disabled
                gemini_none = False
                
                # QUOTA SAVER: Limit Gemini Verification to TOP N Candidates per video
                # Sort remaining by area/confidence to prioritize likely watermarks
                # (Big ones or ones with high local confidence)
                
                limit = int(os.getenv("GEMINI_CANDIDATE_LIMIT", "3"))
                
                if i >= limit: 
                    logger.info(f"🛑 Quota Saver: Skipping Gemini verification for lower-priority candidates (Limit {limit}).")
                    gemini_none = True # Force fallback logic
                    is_valid = False   # Default to false for low priority unless local confidence is high below
                
                elif self.gemini_enabled and not aggressive:
                    import gemini_enhance
                    # verify_watermark returns True, False, or None (Error/Quota)
                    is_valid = gemini_enhance.verify_watermark(frame, candidate_box)
                    if is_valid is None:
                        is_valid = False # Assume false initially, check fallback
                        gemini_none = True
                
                # FALLBACK LOGIC: If Gemini Failed (None) OR We Skipped It (Quota Saver), trust High-Conf RF
                # User requested "Increase Accuracy" of fallback.
                # Use strict 0.85 threshold to avoid false positives.
                if gemini_none:
                     rf_conf = candidate_box.get('rf_conf', 0)
                     # Lower threshold slightly for Skipped candidates if they look reasonable
                     threshold = 0.80 if i >= limit else 0.85
                     
                     if rf_conf > threshold:
                         logger.warning(f"⚠️ Gemini Skipped/Failed. TRUSTING High-Conf Local Detection (RF: {rf_conf:.2f}).")
                         is_valid = True
                     else:
                         if i < limit: # Only log error for the ones we TRIED
                             logger.error(f"❌ Gemini Failed & Local Confidence Low ({rf_conf:.2f} < {threshold}). REJECTING Candidate.")
                         is_valid = False

                if is_valid is False:
                    # Explicit False Check (since None handled above)
                    logger.info(f"❌ Watermark #{i+1} Validation Failed")
                    opencv_watermark.learn(frame, candidate_box, is_positive=False)
                    continue
                else:
                    logger.info(f"✅ Watermark #{i+1} Validation Passed")
                    
                    # STAGE 2: REFINEMENT
                    gemini_refined = candidate_box
                    if self.gemini_enabled:
                        import gemini_enhance
                        gemini_refined = gemini_enhance.refine_watermark(frame, candidate_box)
                        
                    if self.gemini_enabled and gemini_refined != candidate_box:
                         logger.info(f"💎 Gemini Box found: {gemini_refined}. Shrink-wrapping with OpenCV...")
                    
                    refined_box = opencv_watermark.refine_roi(frame, gemini_refined)
                    logger.info(f"   ✨ Hybrid Refined Box: {refined_box}")
                    
                    # Sanity Check: Reject Tiny Boxes (prevents Tracker Crash)
                    if refined_box['w'] < 10 or refined_box['h'] < 10:
                        logger.warning(f"⚠️ Rejecting TINY refined box: {refined_box} (likely detection artifact)")
                        continue # Skip invalid box

                    # Safety Check
                    safe_to_remove = self._is_safe(refined_box, frame, aggressive=aggressive)
                    decision = "remove" if safe_to_remove else "replace"
                    
                    # ACTIVE LEARNING: Teacher-Student Model
                    if safe_to_remove:
                        logger.info(f"🎓 Teaching Local Model (OpenCV) from Gemini detection...")
                        opencv_watermark.learn(frame, refined_box, is_positive=True)
                    
                    # Motion Tracking (Per Watermark)
                    is_moving = False
                    trajectory = []
                    start_time = 0.0
                    end_time = 0.0
                    
                    if self.temporal_enabled:
                        # We use the frame where it was detected as start
                        # But we have multiple detections merged. Use the one with 'detected_at_pct'
                        # Actually, track_watermark scans the whole video anyway.
                        tracking_result = self.track_watermark(video_path, refined_box)
                        # Anchor Check:
                        # If we have multiple detections spanning a large time range with stable coords,
                        # FORCE STATIC. This overrides tracker drift.
                        force_static = False
                        
                        timestamps = []
                        relevant_candidates = []
                        for c in all_candidates:
                            # Check overlap with refined_box
                            # Simple intersection over union or just distance
                            cx = c['x'] + c['w']/2
                            cy = c['y'] + c['h']/2
                            rcx = refined_box['x'] + refined_box['w']/2
                            rcy = refined_box['y'] + refined_box['h']/2
                            
                            dist = ((cx - rcx)**2 + (cy - rcy)**2)**0.5
                            if dist < 50: # Close enough to be the same watermark
                                relevant_candidates.append(c)
                        
                        if relevant_candidates:
                            # Smart Filtering: Outlier Rejection using Median Box
                            # 1. Calc Median Box of relevant candidates
                            med_x = np.median([c['x'] for c in relevant_candidates])
                            med_y = np.median([c['y'] for c in relevant_candidates])
                            med_w = np.median([c['w'] for c in relevant_candidates])
                            med_h = np.median([c['h'] for c in relevant_candidates])
                            
                            filtered_candidates = []
                            for c in relevant_candidates:
                                # Distance from median center
                                cx = c['x'] + c['w']/2
                                cy = c['y'] + c['h']/2
                                mcx = med_x + med_w/2
                                mcy = med_y + med_h/2
                                dist = ((cx - mcx)**2 + (cy - mcy)**2)**0.5
                                
                                # Criteria: Dist < 50px from MEDIAN (Tight Cluster)
                                if dist < 50:
                                    filtered_candidates.append(c)
                            
                            if not filtered_candidates: filtered_candidates = relevant_candidates
                            
                            # CRITICAL FIX: Use PERCENTILE (Robust Union) to exclude outliers
                            # Previously we used the single-frame 'refined_box' which might be partial.
                            # Now we take the P5-P95 extent of valid detections to avoid exploding box size.
                            xs = [c['x'] for c in filtered_candidates]
                            ys = [c['y'] for c in filtered_candidates]
                            ws = [c['w'] for c in filtered_candidates]
                            hs = [c['h'] for c in filtered_candidates]
                            x2s = [c['x'] + c['w'] for c in filtered_candidates]
                            y2s = [c['y'] + c['h'] for c in filtered_candidates]
                            
                            min_x = np.percentile(xs, 5) # 5th percentile (ignore bottom 5% outliers)
                            min_y = np.percentile(ys, 5)
                            max_xw = np.percentile(x2s, 95) # 95th percentile (ignore top 5% outliers)
                            max_yh = np.percentile(y2s, 95)
                            
                            # Update refined_box to be the UNION
                            # But keep it centered/contained? Union is safe.
                            new_w = max_xw - min_x
                            new_h = max_yh - min_y
                            
                            # UPDATE: Always update to the robust UNION of the cluster (Percentile 5-95)
                            # This allows the box to SHRINK if the single-frame detection was too large/noisy.
                            # We trust the temporal cluster more than a single frame.
                            logger.info(f"   📦 updating box to Robust Union of {len(filtered_candidates)} candidates: {min_x:.0f},{min_y:.0f} {new_w:.0f}x{new_h:.0f}")
                            refined_box['x'] = int(min_x)
                            refined_box['y'] = int(min_y)
                            refined_box['w'] = int(new_w)
                            refined_box['h'] = int(new_h)
                            
                            # Use Filtered candidates for stats
                            timestamps = [c['detected_at_pct'] for c in filtered_candidates]
                            xs = [c['x'] for c in filtered_candidates]
                            ys = [c['y'] for c in filtered_candidates]
                            
                            time_spread = max(timestamps) - min(timestamps)
                            
                            # Robust Variance
                            p90_x = np.percentile(xs, 90)
                            p10_x = np.percentile(xs, 10)
                            p90_y = np.percentile(ys, 90)
                            p10_y = np.percentile(ys, 10)
                            
                            spatial_var_x = p90_x - p10_x
                            spatial_var_y = p90_y - p10_y
                            
                            # FALLBACK SAFETY (Fix for "Wrong Blur"):
                            # Only reject if CONFIDENCE IS LOW. High confidence local detections (RF > 0.8)
                            # should be trusted even if they drift/jitter (we can stabilize them later).
                            max_conf = 0.0
                            if filtered_candidates:
                                max_conf = max([c.get('rf_conf', 0) for c in filtered_candidates])

                            gemini_failed = any(c.get("gemini_failed", False) for c in filtered_candidates)
                            
                            if gemini_failed and (spatial_var_x > 30 or spatial_var_y > 30) and max_conf < 0.8:
                                logger.warning(f"🛑 REJECTING Weak Fallback Candidate (Conf {max_conf:.2f}) due to MOTION. Likely False Positive.")
                                continue 
                            
                            # Aggressive Static Lock:
                            # Strict threshold (20px) to ensure we only lock TRULY static watermarks.
                            # UPDATED: Relaxed to 45px to prevent false "Moving" classification on jittery static logos.
                            if time_spread > 0.1 and spatial_var_x < 45 and spatial_var_y < 45:
                                logger.info(f"      ⚓ Anchor Check: Stable detection across {time_spread*100:.0f}% of video. FORCING STATIC.")
                                force_static = True
                                # refined_box is already updated to UNION above.
                                
                        is_moving = tracking_result["is_moving"]
                        trajectory = tracking_result["trajectory"]
                        
                        if force_static:
                            is_moving = False
                            logger.info("      ⚓ Motion override: Anchor Check enforced static mode.")
                        
                        # Determine Time Range
                        # Default to full duration ONLY if coverage is high (>70%) or specifically tracked
                        target_start = 0.0
                        target_end = video_duration
                        
                        detected_duration = 0
                        if trajectory:
                             detected_duration = max(t['ts'] for t in trajectory) - min(t['ts'] for t in trajectory)
                        elif timestamps:
                             # Approximating from timestamps list if trajectory missing
                             detected_duration = (max(timestamps) - min(timestamps)) * video_duration
                        
                        coverage_ratio = detected_duration / video_duration if video_duration > 0 else 1.0

                        if not is_moving:
                            # STATIC LOGIC
                            # ALWAYS Force Full Duration for Static Watermarks
                            # Tracker failure shouldn't mean the watermark disappears.
                            # Standard watermarks/logos are permanent.
                            start_time = 0.0
                            end_time = video_duration
                            logger.info(f"      🕒 Force-Extending Static Duration: 0.0s - {video_duration:.1f}s (Assumed Permanent)")
                            
                            # Deprecated the old "coverage_ratio" logic which caused disappearing boxes.
                            # If it's detected and static, it's there for good.

                        else:
                            # MOVING LOGIC
                            start_time = 0.0
                            end_time = video_duration
                            
                        # Apply Trajectory Envelope (Valid for both Moving and Drift-Static)
                        if trajectory:
                            min_x = min(t['x'] for t in trajectory)
                            min_y = min(t['y'] for t in trajectory)
                            max_xw = max(t['x'] + t['w'] for t in trajectory)
                            max_yh = max(t['y'] + t['h'] for t in trajectory)
                            
                            env_w = max_xw - min_x
                            env_h = max_yh - min_y
                            
                            # Only use envelope if it's reasonable (not massive explosion)
                            # If Force Static, be strict (we verified it's stable). Limit growth to 1.5x (minor drift).
                            limit = 1.5 if force_static else 3
                            area_orig = refined_box['w'] * refined_box['h']
                            area_env = env_w * env_h
                            
                            if area_env < area_orig * limit: 
                                refined_box['x'] = min_x
                                refined_box['y'] = min_y
                                refined_box['w'] = env_w
                                refined_box['h'] = env_h
                                logger.info(f"   📦 Applied Trajectory Envelope: Expanded static box to {refined_box}")
                    
                    final_watermarks.append({
                        "id": i+1,
                        "coordinates": refined_box,
                        "confidence": refined_box.get('rf_conf', 0.9),
                        "safe_to_remove": safe_to_remove,
                        "decision": decision,
                        "time_range": {"start": start_time, "end": end_time},
                        "is_moving": is_moving,
                        "trajectory": trajectory
                    })

            # DEDUPLICATION STEP: Merge overlapping boxes
            final_watermarks = self._merge_overlapping_results(final_watermarks)

            # FINAL SAFETY CHECK: Reject Vertical Boxes (People/Objects)
            # This runs AFTER merging to catch any "Frankenstein" vertical merges.
            safe_watermarks = []
            for wm in final_watermarks:
                box = wm['coordinates']
                ratio = box['h'] / box['w'] if box['w'] > 0 else 0
                
                # RELAXED THRESHOLD: 1.2 -> 3.5 to allow Vertical Text Watermarks
                if ratio > 3.5:
                    logger.warning(f"🚨 Final Safety Reject: Vertical Box detected {ratio:.2f} (Likely Person/Artifact). Dropping {box}")
                    continue
                safe_watermarks.append(wm)
            final_watermarks = safe_watermarks

            # Re-index
            for i, wm in enumerate(final_watermarks):
                wm['id'] = i + 1

            # Construct Output
            frame_path = os.path.join(os.path.dirname(video_path), f"frame_{uuid.uuid4().hex[:6]}.jpg")
            cv2.imwrite(frame_path, frame)
            
            result = {
                "watermarks": final_watermarks,
                "count": len(final_watermarks),
                "context": {
                    "frame_path": frame_path
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Orchestrator Error: {e}")
            return self._error_json(str(e))

    def _merge_overlapping_results(self, watermarks):
        """
        Merges watermarks that overlap significantly (>30% IoU).
        Prioritizes MOVING result over STATIC if overlapping, as Moving tracks better.
        """
        if len(watermarks) < 2:
            return watermarks
            
        merged = []
        keep_indices = set(range(len(watermarks)))
        
        for i in range(len(watermarks)):
            if i not in keep_indices: continue
            
            for j in range(i + 1, len(watermarks)):
                if j not in keep_indices: continue
                
                wm1 = watermarks[i]
                wm2 = watermarks[j]
                
                # Calc IoU
                box1 = wm1['coordinates']
                box2 = wm2['coordinates']
                
                x1 = max(box1['x'], box2['x'])
                y1 = max(box1['y'], box2['y'])
                x2 = min(box1['x']+box1['w'], box2['x']+box2['w'])
                y2 = min(box1['y']+box1['h'], box2['y']+box2['h'])
                
                inter_w = max(0, x2 - x1)
                inter_h = max(0, y2 - y1)
                intersection = inter_w * inter_h
                
                area1 = box1['w'] * box1['h']
                area2 = box2['w'] * box2['h']
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                # Check for significant overlap or Inclusion
                u_overlap_1 = intersection / area1 if area1 > 0 else 0
                u_overlap_2 = intersection / area2 if area2 > 0 else 0
                
                # PROXIMITY MERGE (For fragmented text like "Telly" ... "TV")
                # Calculate Horizontal Gap
                x1_right = box1['x'] + box1['w']
                x2_right = box2['x'] + box2['w']
                
                # Gap is distance between inner edges
                if box1['x'] < box2['x']:
                    gap_x = box2['x'] - x1_right
                else:
                    gap_x = box1['x'] - x2_right
                    
                # Calculate Vertical Overlap Ratio
                y_inter_h = max(0, min(box1['y']+box1['h'], box2['y']+box2['h']) - max(box1['y'], box2['y']))
                min_h = min(box1['h'], box2['h'])
                y_overlap_ratio = y_inter_h / min_h if min_h > 0 else 0
                
                # MERGE CONDITION:
                # 1. Significant Overlap (IoU > 0.3) OR
                # 2. One inside another (>80% coverage) OR
                # 3. Horizontal Proximity (< 30px gap) AND Vertical Alignment (> 50% overlap) - "Telly...TV"
                should_merge = (iou > 0.3) or (u_overlap_1 > 0.8) or (u_overlap_2 > 0.8)
                
                # Condition 3: Horizontal Proximity
                if not should_merge and gap_x < 30 and y_overlap_ratio > 0.5:
                    logger.info(f"   ↔️ Horizontal Merge Triggered: Gap {gap_x}px")
                    should_merge = True

                # Condition 4: SAFE Vertical Stacking (Multi-line Text) - "Filmy\nGalaxy"
                # STRICT Rules to avoid merging body parts:
                # 1. Must be strictly aligned vertically (High X-Overlap).
                # 2. Must be very close (Small Y-Gap).
                # 3. Result must NOT be a tall vertical strip (Person).
                
                # Calculate Vertical Gap
                if box1['y'] < box2['y']:
                    gap_y = box2['y'] - (box1['y'] + box1['h'])
                else:
                    gap_y = box1['y'] - (box2['y'] + box2['h'])
                
                # Calculate Horizontal Overlap Ratio
                x_inter_w = max(0, min(box1['x']+box1['w'], box2['x']+box2['w']) - max(box1['x'], box2['x']))
                min_w = min(box1['w'], box2['w'])
                x_overlap_ratio = x_inter_w / min_w if min_w > 0 else 0

                if not should_merge and gap_y < 20 and x_overlap_ratio > 0.7:
                     # PREDICTED DIMENSIONS
                     new_x = min(box1['x'], box2['x'])
                     new_y = min(box1['y'], box2['y'])
                     new_x2 = max(box1['x']+box1['w'], box2['x']+box2['w'])
                     new_y2 = max(box1['y']+box1['h'], box2['y']+box2['h'])
                     new_w = new_x2 - new_x
                     new_h = new_y2 - new_y
                     
                     # ASPECT RATIO GUARDRAIL:
                     # If the merged result is Taller than Wide (Ratio > 1.2), IT IS A PERSON/LIMB.
                     # We reject the merge to be safe. 
                     # "Filmy\nGalaxy" is usually roughly square or wider.
                     merge_ratio = new_h / new_w if new_w > 0 else 0
                     if merge_ratio < 1.2: 
                         logger.info(f"   jh Vertical Merge Triggered (Safe): Gap {gap_y}px, Ratio {merge_ratio:.2f}")
                         should_merge = True
                     else:
                         logger.warning(f"   ⚠️ Vertical Merge BLOCKED: Resulting box would be too vertical (Ratio {merge_ratio:.2f})")

                if should_merge:
                    logger.info(f"   🔄 Merging Watermarks #{i+1} and #{j+1} (IoU: {iou:.2f})")
                    
                    # Create UNION Box
                    new_x = min(box1['x'], box2['x'])
                    new_y = min(box1['y'], box2['y'])
                    new_x2 = max(box1['x']+box1['w'], box2['x']+box2['w'])
                    new_y2 = max(box1['y']+box1['h'], box2['y']+box2['h'])
                    
                    union_box = {
                        'x': new_x, 'y': new_y,
                        'w': new_x2 - new_x,
                        'h': new_y2 - new_y
                    }
                    
                    # Update wm1 (Kept) with Union Box
                    watermarks[i]['coordinates'] = union_box
                    
                    # Merge Logic (Keep Moving status if either is moving)
                    if watermarks[j]['is_moving']:
                        watermarks[i]['is_moving'] = True
                        if not watermarks[i]['trajectory'] and watermarks[j]['trajectory']:
                             watermarks[i]['trajectory'] = watermarks[j]['trajectory']
                    
                    keep_indices.remove(j) # Drop j
                    # i has absorbed j, continue checking i against others?
                    # Since we modified i, we should conceptually re-check, but for simple sweep this is okay.
                            
        return [watermarks[k] for k in sorted(keep_indices)]

    def _fuse_results(self, cv_boxes, gemini_boxes):
        """
        Merges two lists of boxes using NMS.
        """
        all_boxes = (cv_boxes or []) + (gemini_boxes or [])
        # Use OpenCV's NMS logic (we can reuse the static method if we make it public or duplicate)
        # Let's implement a simple one here or import
        import opencv_watermark
        return opencv_watermark.WatermarkDetector._nms(all_boxes, overlap_thresh=0.3)

    def _temporal_nms(self, candidates):
        """
        Merges candidates from different frames that are likely the same watermark.
        """
        import opencv_watermark
        return opencv_watermark.WatermarkDetector._nms(candidates, overlap_thresh=0.5)

    def _is_safe(self, box, frame, aggressive=False):
        if not box: return False
        
        # If 2-Step Verification is OFF, assume safe if detected
        if not self.two_step_verification:
            return True
        
        # Area Check
        h_img, w_img = frame.shape[:2]
        area_pct = (box['w'] * box['h']) / (w_img * h_img) * 100
        
        # Aggressive Mode: Allow larger watermarks (up to 30%)
        max_area = 30 if aggressive else self.max_area_pct
        
        if area_pct > max_area:
            logger.info(f"⚠️ Watermark rejected: Area too large ({area_pct:.1f}% > {max_area}%)")
            return False
            
        # Confidence Check
        # Check RF confidence if available, otherwise check generic confidence
        rf_conf = box.get('rf_conf', 0)
        gen_conf = box.get('confidence', 0)
        
        # Aggressive Mode: Lower thresholds
        min_gen = 0.3 if aggressive else 0.4 # Dramatic Reduction
        min_rf = 0.2 if aggressive else 0.35
        
        if rf_conf == 0 and gen_conf < min_gen: # If no RF, need high generic
             logger.info(f"⚠️ Watermark rejected: Low confidence (Generic: {gen_conf:.2f} < {min_gen})")
             return False
        if rf_conf > 0 and rf_conf < min_rf: # If RF exists, must be decent
             logger.info(f"⚠️ Watermark rejected: Low RF confidence ({rf_conf:.2f} < {min_rf})")
             return False
            
        if box.get('template_neg_score', 0) > box.get('template_pos_score', 0):
            # In aggressive mode, we might ignore this if confidence is high enough
            if not aggressive:
                logger.info("⚠️ Watermark rejected: Template negative score > positive score")
                return False
            else:
                logger.info("⚠️ Template negative > positive, but ignoring in AGGRESSIVE mode.")
        
        # Aspect Ratio Check (Anti-Person Filter)
        # Watermarks are usually Horizontal (Text/Logo). 
        # Vertical boxes (Height > 1.5x Width) are usually people/arms/objects.
        ratio = box['h'] / box['w'] if box['w'] > 0 else 0
        if ratio > 1.5:
             logger.info(f"⚠️ Watermark rejected: Vertical Aspect Ratio {ratio:.2f} (Likely Person/Object)")
             return False

        # Moving Object Safety Check
        # If it's a "Moving" candidate (detected across multiple frames but shifting),
        # but it's LARGE (>10% area), it's overwhelmingly likely to be an actor, not a watermark.
        # Watermarks are tiny.
        if area_pct > 10 and not self.two_step_verification:
             logger.info(f"⚠️ Watermark rejected: Large Moving Object ({area_pct:.1f}%) matches 'Actor' profile.")
             return False
            
        return True

    def _create_tracker(self):
        """Creates an OpenCV tracker with fallback."""
        # Try CSRT (Best accuracy)
        try:
            return cv2.TrackerCSRT_create()
        except AttributeError:
            pass
            
        # Try KCF (Fast)
        try:
            return cv2.TrackerKCF_create()
        except AttributeError:
            pass
            
        # Try Legacy API
        try:
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            pass
            
        return None

    def track_watermark(self, video_path: str, initial_box: dict) -> dict:
        """
        Tracks the watermark across the entire video (Forward AND Backward).
        Returns: {
            "is_moving": bool,
            "trajectory": list of {"ts": float, "x": int, "y": int, "w": int, "h": int}
        }
        """
        logger.info("🕵️ Starting Motion Tracking Analysis...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"is_moving": False, "trajectory": []}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Start from the frame where detection happened (approx 20%)
        start_frame_idx = int(total_frames * 0.2)
        
        # --- 1. FORWARD TRACKING ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"is_moving": False, "trajectory": []}
            
        tracker_fw = self._create_tracker()
        if not tracker_fw:
            logger.warning("⚠️ No suitable tracker found. Assuming static.")
            cap.release()
            return {"is_moving": False, "trajectory": []}
            
        h_img, w_img = frame.shape[:2]
        x = max(0, min(initial_box['x'], w_img - 1))
        y = max(0, min(initial_box['y'], h_img - 1))
        w = max(1, min(initial_box['w'], w_img - x))
        h = max(1, min(initial_box['h'], h_img - y))
        
        bbox = (x, y, w, h)
        tracker_fw.init(frame, bbox)
        
        # VISUAL TEMPLATE CAPTURE
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h_img, w_img = frame.shape[:2]
        # Safety clamp
        x = max(0, min(x, w_img-1))
        y = max(0, min(y, h_img-1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        template = None
        if w > 0 and h > 0:
            template = frame[y:y+h, x:x+w].copy()
            template_h, template_w = template.shape[:2]
            logger.info(f"📸 Visual Anchor Captured: {w}x{h}")
        else:
             logger.warning("⚠️ Could not capture visual anchor (invalid coords).")
             
        trajectory = []
        
        # Add start frame
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        trajectory.append({"ts": ts, "x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])})
        
        frames_tracked = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            success, box = tracker_fw.update(frame)
            if success:
                x, y, w, h = map(int, box)
                frames_tracked += 1
                
                # VISUAL CONSISTENCY CHECK (Optimized: Every 5 frames)
                if frames_tracked % 5 == 0:
                    # Extract tracked region
                    h_img, w_img = frame.shape[:2]
                    x_safe = max(0, min(x, w_img-1))
                    y_safe = max(0, min(y, h_img-1))
                    w_safe = min(w, w_img - x_safe)
                    h_safe = min(h, h_img - y_safe)
                    
                    if w_safe > 0 and h_safe > 0:
                        tracked_roi = frame[y_safe:y_safe+h_safe, x_safe:x_safe+w_safe]
                        # Resize to match template size for comparison if needed, or use template matching on Resize
                        # Actually, simple template matching requires template <= image.
                        # Let's simple Resize and Correlation.
                        try:
                            tracked_roi_resized = cv2.resize(tracked_roi, (template_w, template_h))
                            
                            # Compare
                            res = cv2.matchTemplate(tracked_roi_resized, template, cv2.TM_CCOEFF_NORMED)
                            score = res[0][0]
                            
                            # Relaxed Drift Threshold: 0.5 -> 0.15
                            # We trust the CSRT tracker. Visual check should only stop CATASTROPHIC failures.
                            if score < 0.15: 
                                logger.warning(f"🛑 Visual Drift Detected! Score {score:.2f} < 0.15. Stopping Forward Tracking.")
                                break
                            # Silent warning for minor drift to avoid spamming user logs
                            # elif score < 0.4: logger.warning(...)
                        except Exception: pass

                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                trajectory.append({"ts": ts, "x": x, "y": y, "w": w, "h": h})
            else:
                break

        # --- 2. BACKWARD TRACKING ---
        # Read frames from 0 to start_frame_idx-1
        # Since this is usually short (start of video), reading into memory is okay for Shorts.
        # For long videos, this might be heavy, but we assume Shorts (<60s).
        
        backward_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(start_frame_idx):
            ret, frame = cap.read()
            if ret:
                backward_frames.append(frame)
            else:
                break
        
        cap.release()
        
        if backward_frames:
            # Reverse the list to track "backwards" in time
            backward_frames.reverse()
            
            tracker_bw = self._create_tracker()
            if tracker_bw:
                # Re-init with the SAME initial box at the start frame (which is the first of our reversed list effectively)
                # Wait, the start frame for backward tracking is the one BEFORE start_frame_idx.
                # But we initialized tracker on start_frame_idx.
                # Let's re-init tracker on start_frame_idx again (we need that frame content).
                
                # Actually, we can just use the initial_box and init on the first frame of backward_frames 
                # IF we assume continuity. But the "first" frame of backward_frames is start_frame_idx - 1.
                # The tracker needs to be initialized on start_frame_idx to track into start_frame_idx - 1.
                
                # Let's grab start_frame_idx content again
                cap_temp = cv2.VideoCapture(video_path)
                cap_temp.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
                ret, start_frame_img = cap_temp.read()
                cap_temp.release()
                
                if ret:
                    tracker_bw.init(start_frame_img, bbox)
                    
                    frames_tracked_bw = 0
                    for i, frame in enumerate(backward_frames):
                        success, box = tracker_bw.update(frame)
                        if success:
                            x, y, w, h = map(int, box)
                            
                            frames_tracked_bw += 1
                            
                            # VISUAL CONSISTENCY CHECK (Backward - Periodic)
                            if template is not None and frames_tracked_bw % 5 == 0:
                                h_img, w_img = frame.shape[:2]
                                x_safe = max(0, min(x, w_img-1))
                                y_safe = max(0, min(y, h_img-1))
                                w_safe = min(w, w_img - x_safe)
                                h_safe = min(h, h_img - y_safe)
                                
                                if w_safe > 0 and h_safe > 0:
                                    try:
                                        tracked_roi = frame[y_safe:y_safe+h_safe, x_safe:x_safe+w_safe]
                                        tracked_roi_resized = cv2.resize(tracked_roi, (template_w, template_h))
                                        res = cv2.matchTemplate(tracked_roi_resized, template, cv2.TM_CCOEFF_NORMED)
                                        score = res[0][0]
                                        
                                        if score < 0.5:
                                            logger.warning(f"🛑 Visual Drift Detected (Backward)! Score {score:.2f} < 0.5.")
                                            break
                                    except Exception: pass
                            
                            # Calculate timestamp: (start_frame_idx - 1 - i) / fps
                            frame_num = start_frame_idx - 1 - i
                            ts = frame_num / fps if fps > 0 else 0
                            trajectory.append({"ts": ts, "x": x, "y": y, "w": w, "h": h})
                        else:
                            break

        # Sort trajectory by timestamp
        trajectory.sort(key=lambda k: k['ts'])

        if not trajectory:
             return {"is_moving": False, "trajectory": []}

        # Analyze Range (Max - Min) with OUTLIER REJECTION
        # Filter out tracking glitches/drift.
        if trajectory:
            med_x = np.median([t['x'] for t in trajectory])
            med_y = np.median([t['y'] for t in trajectory])
            
            # Keep only points within reasonable distance from median (e.g. 150px)
            # Real moving watermarks (tickers) move linearly, but static ones shouldn't jump.
            valid_traj = [t for t in trajectory if abs(t['x'] - med_x) < 150 and abs(t['y'] - med_y) < 150]
            
            if not valid_traj: valid_traj = trajectory # Fallback
            
            xs = [t['x'] for t in valid_traj]
            ys = [t['y'] for t in valid_traj]
            
            range_x = (max(xs) - min(xs)) if xs else 0
            range_y = (max(ys) - min(ys)) if ys else 0
        else:
            range_x = 0
            range_y = 0
        
        # Base Threshold: Increased to 100px to avoid minor drift being seen as motion
        threshold = 100 
        
        # Corner Bias: If in a corner, be even stricter (likely a static logo)
        if trajectory:
            fx, fy = trajectory[0]['x'], trajectory[0]['y']
            w_img = frame_w
            h_img = frame_h
            
            is_corner = (fx < w_img*0.15 or fx > w_img*0.85) or (fy < h_img*0.15 or fy > h_img*0.85)
            if is_corner:
                threshold = 150 # Very strict for corners
                logger.info("      📐 Corner detection: Increased motion threshold to 150px")

        is_moving = bool(range_x > threshold or range_y > threshold)
        
        # DRIFT GUARD REMOVED - Visual Check replaces it.
        # The visual check ensures that if the box drifts to something that doesn't look like the logo,
        # tracking stops. This effectively handles the "small logo vs background" case naturally.
        
        if is_moving:
            logger.info(f"🏃 Motion Detected! Range X:{range_x}px, Y:{range_y}px (Threshold: {threshold})")
        else:
            logger.info(f"🗿 Static Watermark Confirmed. Range X:{range_x}px, Y:{range_y}px (Threshold: {threshold})")
            
        return {
            "is_moving": is_moving,
            "trajectory": trajectory
        }

    def _analyze_temporal(self, box):
        # Deprecated/Placeholder - logic moved to track_watermark
        return {"pattern": "static", "consistency_score": 1.0}

    def _get_reason(self, box, safe):
        if not box: return "no_detection"
        if safe:
            return "stable_across_frames;rf_positive;safe_area"
        else:
            reasons = []
            if box.get('rf_conf', 0) < 0.40: reasons.append("low_confidence")
            # We'd need to pass the area check result here to be precise, but for now:
            reasons.append("unsafe_criteria") 
        return ",".join(reasons)

    def _error_json(self, msg):
        return json.dumps({
            "watermark_detected": False,
            "confidence": 0.0,
            "coordinates": {"x": 0, "y": 0, "w": 0, "h": 0},
            "safe_to_remove": False,
            "decision": "none",
            "reason": f"error: {msg}",
            "summary": {
                "do_enhancement": False,
                "do_watermark_remove": False,
                "do_watermark_replace": False,
                "do_crop": False
            }
        }, indent=2)

    def confirm_learning(self, context: dict, is_positive: bool):
        """
        Feedback loop for Active Learning.
        """
        if not context: return
        
        frame_path = context.get('frame_path')
        coords = context.get('coords')
        
        if frame_path and os.path.exists(frame_path) and coords:
            try:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    opencv_watermark.learn(frame, coords, is_positive=is_positive)
                    logger.info(f"✅ Learning confirmed: {'POSITIVE' if is_positive else 'NEGATIVE'}")
                    
                    # If Self-Improve is enabled, we might want to save this sample specifically for nightly retrain
                    # opencv_watermark.log_feedback already appends to the CSV which is used for retraining.
                    
            except Exception as e:
                logger.error(f"❌ Learning failed: {e}")
            finally:
                try:
                    os.remove(frame_path)
                except: pass

    def save_missed_detection(self, context: dict):
        """
        Saves the frame when a watermark was missed (User said 'No' to 'No watermark detected').
        """
        if not context: return
        
        frame_path = context.get('frame_path')
        if frame_path and os.path.exists(frame_path):
            try:
                missed_dir = "missed_watermarks"
                os.makedirs(missed_dir, exist_ok=True)
                
                timestamp = int(time.time())
                dst = os.path.join(missed_dir, f"missed_{timestamp}.jpg")
                shutil.copy2(frame_path, dst)
                logger.info(f"📸 Saved missed detection sample: {dst}")
            except Exception as e:
                logger.error(f"❌ Failed to save missed sample: {e}")
            finally:
                try:
                    os.remove(frame_path)
                except: pass

    def generate_dynamic_mask(self, video_path: str, trajectory: list, output_path: str, template_mask: np.ndarray = None):
        """
        Generates a black and white mask video from the trajectory.
        White = Watermark Area, Black = Background.
        
        Args:
            template_mask: Optional binary mask (0/255) of the watermark shape. 
                           If provided, it's resized and placed at the tracked coords.
                           If None, falls back to Rectangle.
        """
        logger.info(f"🎭 Generating Dynamic 'Exact-Shape' Mask for {len(trajectory)} frames...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("❌ Could not open video for mask generation")
            return False
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
        
        # --- SMOOTHING & INTERPOLATION LOGIC ---
        # Convert trajectory to dense map first
        traj_map = {round(t['ts'] * fps): t for t in trajectory}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = int(fps * 60) # Fallback
        
        # 1. INTERPOLATION: Fill gaps in traj_map
        known_indices = sorted(traj_map.keys())
        full_traj_map = traj_map.copy()
        
        if known_indices:
            # Fill gaps between known indices
            for k in range(len(known_indices) - 1):
                f_start = known_indices[k]
                f_end = known_indices[k+1]
                
                if f_end - f_start > 1: # Gap found
                    box_start = traj_map[f_start]
                    box_end = traj_map[f_end]
                    
                    steps = f_end - f_start
                    for step in range(1, steps):
                        curr_f = f_start + step
                        alpha = step / steps
                        
                        # Linear Interpolation
                        interp_x = int(box_start['x'] * (1-alpha) + box_end['x'] * alpha)
                        interp_y = int(box_start['y'] * (1-alpha) + box_end['y'] * alpha)
                        interp_w = int(box_start['w'] * (1-alpha) + box_end['w'] * alpha)
                        interp_h = int(box_start['h'] * (1-alpha) + box_end['h'] * alpha)
                        
                        full_traj_map[curr_f] = {'x': interp_x, 'y': interp_y, 'w': interp_w, 'h': interp_h}
                        
            # Extrapolate edges (Hold first/last position)
            first_f = known_indices[0]
            last_f = known_indices[-1]
            
            # Backfill 0 to first_f
            for f in range(first_f):
                full_traj_map[f] = traj_map[first_f]
                
            # Forwardfill last_f to end
            for f in range(last_f + 1, total_frames):
                full_traj_map[f] = traj_map[last_f]
        
        smoothed_trajectory = {}
        
        # SMOOTHING UPDATED: Increased to 30 frames to fix "Dancing/Jittery" mask.
        window_size = 30 
        
        # DEADZONE STATE (Hysteresis)
        last_smoothed_x = -1
        last_smoothed_y = -1
        last_smoothed_w = -1
        last_smoothed_h = -1
        
        for i in range(total_frames):
            # Gather window from FULL (Interpolated) map
            window_boxes = []
            for j in range(i - window_size // 2, i + window_size // 2 + 1):
                if j in full_traj_map:
                    window_boxes.append(full_traj_map[j])
            
            if window_boxes:
                avg_x = sum(b['x'] for b in window_boxes) / len(window_boxes)
                avg_y = sum(b['y'] for b in window_boxes) / len(window_boxes)
                avg_w = sum(b['w'] for b in window_boxes) / len(window_boxes)
                avg_h = sum(b['h'] for b in window_boxes) / len(window_boxes)
                
                # REMOVED DEADZONE: Rely on Moving Average for smooth continuous motion.
                # Deadzone caused "Stick-Slip" jerkiness ("Dancing").
                
                last_smoothed_x, last_smoothed_y, last_smoothed_w, last_smoothed_h = avg_x, avg_y, avg_w, avg_h
                
                smoothed_trajectory[i] = {
                    'x': int(avg_x), 'y': int(avg_y), 
                    'w': int(avg_w), 'h': int(avg_h)
                }
            
        # OPTIMIZED LOOP: Iterate without reading input video (IO Bottleneck Removed)
        for frame_idx in range(total_frames):
            # Create black frame
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Use SMOOTHED trajectory
            t_data = smoothed_trajectory.get(frame_idx)
            # Fallback to original if smoothing failed (edges)
            if not t_data: t_data = traj_map.get(frame_idx) 
            
            if t_data:
                x, y, w, h = t_data['x'], t_data['y'], t_data['w'], t_data['h']
                
                # Pad slightly for motion fit (Dynamic needs to be looser than static)
                pad = 4 
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(width - x, w + 2*pad)
                h = min(height - y, h + 2*pad)
                
                if template_mask is not None and template_mask.size > 0 and cv2.countNonZero(template_mask) > 50:
                    # EXACT SHAPE TRACKING
                    try:
                        # Resize template to match current tracked box size
                        # Note: template_mask is 0-255 binary
                        resized_shape = cv2.resize(template_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # Apply to mask
                        # Handle boundaries
                        target_y2 = min(height, y + h)
                        target_x2 = min(width, x + w)
                        actual_h = target_y2 - y
                        actual_w = target_x2 - x
                        
                        if actual_h > 0 and actual_w > 0:
                             # Crop resized shape if it hits image boundary
                             roi_mask = resized_shape[:actual_h, :actual_w]
                             mask[y:y+actual_h, x:x+actual_w] = roi_mask
                    except Exception as e:
                         # Fallback to rect if resize fails
                         cv2.rectangle(mask, (x, y), (x+w, y+h), (255), -1)
                else:
                    # RECTANGLE FALLBACK
                    cv2.rectangle(mask, (x, y), (x+w, y+h), (255), -1)
            
            out.write(mask)
            
        cap.release()
        out.release()
        logger.info(f"✅ Dynamic Mask Generated: {output_path}")
        return True

# Singleton for easy import
orchestrator = HybridOrchestrator()

def process_video(video_path: str, aggressive: bool = False):
    return orchestrator.process_video(video_path, aggressive=aggressive)

def confirm_learning(context: dict, is_positive: bool):
    return orchestrator.confirm_learning(context, is_positive)

def save_missed_detection(context: dict):
    return orchestrator.save_missed_detection(context)

def generate_dynamic_mask(video_path: str, trajectory: list, output_path: str, template_mask: np.ndarray = None):
    return orchestrator.generate_dynamic_mask(video_path, trajectory, output_path, template_mask)

def generate_static_mask(video_path: str, box: dict, output_path: str):
    """
    Generates a STATIC mask video (full duration) using the exact watermark shape.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 1. Extract Representative Frame (20%)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.2))
        ret, frame = cap.read()
        cap.release()
        
        if not ret: return False
        
        # 2. Get Exact Mask (with Padding to catch edges)
        import opencv_watermark
        
        # Expand box by 3px (Precise).
        # Matching Black Dress tightness.
        pad = 3
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        
        # Clamp to image boundaries
        x_pad = max(0, x - pad)
        y_pad = max(0, y - pad)
        w_pad = min(width - x_pad, w + 2*pad)
        h_pad = min(height - y_pad, h + 2*pad)
        
        padded_box = {'x': x_pad, 'y': y_pad, 'w': w_pad, 'h': h_pad}
        
        mask_frame = opencv_watermark.get_watermark_mask(frame, padded_box, watermark_type=box.get('type', 'TEXT_WHITE'))
        
        # FALLBACK: If mask is too small (weak detection or just noise), use the BOX.
        # < 300 pixels means we failed to capture the main body of the logo/text.
        # Better to have a Box Blur than a visible logo.
        if cv2.countNonZero(mask_frame) < 300:
            logger.warning("⚠️ Exact-Shape Mask is too small/empty (<300px). Falling back to Rectangular Blur.")
            # Use the Padded Box area
            cv2.rectangle(mask_frame, (x_pad, y_pad), (x_pad+w_pad, y_pad+h_pad), 255, -1)
        
        # Dilation for safety (User wants "fully covered")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) # 5px dilation
        mask_frame = cv2.dilate(mask_frame, kernel, iterations=1)
        
        # 3. Write Video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
        
        # Optimization: Loop is slow in Python, but needed for video file.
        # Since it's static, maybe we can use FFmpeg loop?
        # But `compiler.py` expects a file.
        # Let's write frames.
        
        for _ in range(total_frames):
             out.write(mask_frame)
             
        out.release()
        logger.info(f"✅ Static Exact-Shape Mask Generated: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate static mask: {e}")
        return False
