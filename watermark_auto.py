"""
Automated Watermark Detection and Replacement System
Production-Ready Orchestrator (Consolidated)
"""

import os
import cv2
import numpy as np
import subprocess
import logging
import shutil
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("watermark_auto")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
TEMP_DIR = "temp_watermark"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_frame(video_path: str, output_path: str) -> bool:
    try:
        cmd = [
            FFMPEG_BIN, "-y", "-ss", "00:00:01", "-i", video_path,
            "-vframes", "1", "-q:v", "2", output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception:
        return False

def detect_watermark_with_gemini(image_path: str, original_video_path: str = None, retry_mode: bool = False) -> Optional[Dict]:
    """
    Detects watermark using Gemini Vision with 2-step verification.
    retry_mode: If True, uses aggressive prompting.
    """
    gemini_coords = None
    verified = False
    
    target_text = os.getenv("WATERMARK_DETECT_TEXT", "watermark, logo, text overlay")
    
    # 1. Try Dynamic Metadata (Highest Priority)
    if original_video_path:
        try:
            meta_path = os.path.splitext(original_video_path)[0] + '.json'
            if os.path.exists(meta_path):
                import json
                import re
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                keywords = []
                if meta.get('uploader'):
                    u = str(meta['uploader'])
                    keywords.append(u)
                    keywords.append(f"@{u}")
                    keywords.append(u.replace(" ", ""))
                
                if meta.get('caption'):
                    tags = re.findall(r'#(\w+)', str(meta['caption']))
                    keywords.extend([f"#{t}" for t in tags])
                    keywords.extend(tags)
                    
                if meta.get('tags'):
                    keywords.extend([str(t) for t in meta['tags']])
                
                # --- EXPAND KEYWORDS (User Request) ---
                expanded_keywords = set()
                for k in keywords:
                    if not k: continue
                    k = str(k).strip()
                    if not k: continue
                    
                    # 1. Remove Special Characters FIRST
                    k_clean = re.sub(r'^[^a-zA-Z0-9]+', '', k)
                    
                    if not k_clean: continue # Skip if nothing left
                    
                    # 2. Add Cleaned & Case Variants
                    expanded_keywords.add(k_clean)
                    expanded_keywords.add(k_clean.lower())
                    expanded_keywords.add(k_clean.upper())
                        
                unique = list(expanded_keywords)
                if unique:
                    target_text = ", ".join(unique)
                    logger.info(f"🔍 Using Dynamic Watermark Keywords: {target_text[:100]}...")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load metadata: {e}")

    # 2. Fallback to .env List (Medium Priority)
    if target_text == os.getenv("WATERMARK_DETECT_TEXT", "FILMYMANTRA"):
         target_list = os.getenv("WATERMARK_DETECT_LIST", "")
         if target_list:
             target_text = target_list
    
    try:
        from gemini_captions import GeminiCaptionGenerator
        gemini_gen = GeminiCaptionGenerator()
        
        # STEP 1: Detect watermark location
        logger.info(f"🔍 STEP 1: Detecting watermark '{target_text}' with Gemini AI...")
        gemini_coords = gemini_gen.detect_watermark_location(image_path, target_text=target_text)
        
        if gemini_coords:
            logger.info(f"   📍 Gemini detected potential watermark at: {gemini_coords}")
            
            # SAFETY CHECK: Area Size
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    h_img, w_img = img.shape[:2]
                    area_wm = gemini_coords['w'] * gemini_coords['h']
                    area_img = h_img * w_img
                    ratio = area_wm / area_img
                    
                    max_ratio = float(os.getenv("WATERMARK_MAX_AREA_PERCENT", "20")) / 100.0
                    
                    if ratio > max_ratio:
                        logger.warning(f"❌ Watermark too large ({ratio*100:.1f}% > {max_ratio*100:.1f}%). Rejecting.")
                        # LEARN NEGATIVE (False Positive - Too Large)
                        try:
                            import opencv_watermark
                            opencv_watermark.learn(img, gemini_coords, is_positive=False)
                        except: pass
                        return None
            except Exception as e:
                logger.warning(f"⚠️ Area check failed: {e}")

            # 🔍 STEP 2: Verify the coordinates are correct
            verified = True
            if retry_mode:
                logger.info("⏩ Skipping 2-Step Verification (Retry Mode Active - User Confirmed Watermark)")
            elif os.getenv("WATERMARK_2STEP_VERIFICATION", "on").lower() == "on":
                logger.info(f"🔍 STEP 2: Verifying coordinates for '{target_text}'...")
                verified = gemini_gen.verify_watermark_coordinates(image_path, gemini_coords, target_text=target_text)
            else:
                logger.info("⏩ Skipping 2-Step Verification (Disabled in .env)")
            
            if verified:
                # EXPAND BOX IN RETRY MODE (User said "No" -> Likely missed edges)
                if retry_mode:
                    pad_w = int(gemini_coords['w'] * 0.2) # 20% expansion
                    pad_h = int(gemini_coords['h'] * 0.2)
                    
                    gemini_coords['x'] = max(0, gemini_coords['x'] - pad_w // 2)
                    gemini_coords['y'] = max(0, gemini_coords['y'] - pad_h // 2)
                    # Dynamic Clamp
                    h_img, w_img = 1080, 1920
                    try:
                        temp_img = cv2.imread(image_path)
                        if temp_img is not None:
                             h_img, w_img = temp_img.shape[:2]
                    except: pass

                    gemini_coords['w'] = min(gemini_coords['w'] + pad_w, w_img - gemini_coords['x']) # Clamp width
                    gemini_coords['h'] = min(gemini_coords['h'] + pad_h, h_img - gemini_coords['y']) # Clamp height
                    
                    logger.info(f"🔄 Retry Mode: Expanded watermark box by 20% to ensure coverage: {gemini_coords}")

                if os.getenv("WATERMARK_2STEP_VERIFICATION", "on").lower() == "on":
                    logger.info(f"✅ 2-STEP VERIFICATION PASSED! Watermark confirmed at: {gemini_coords}")
                else:
                    logger.info(f"✅ Watermark confirmed at: {gemini_coords}")
            else:
                logger.warning(f"❌ 2-STEP VERIFICATION FAILED! Coordinates rejected by Gemini.")
                
                # LEARN NEGATIVE (False Positive - Rejected by Verification)
                try:
                    import opencv_watermark
                    frame = cv2.imread(image_path)
                    if frame is not None:
                        opencv_watermark.learn(frame, gemini_coords, is_positive=False)
                except: pass
                
                # IMPORTANT: Set verified to False so we fall through
                gemini_coords = None 
                verified = False 

        else:
            logger.info(f"   ℹ️ Gemini did not detect watermark '{target_text}'")

        # FINAL DECISION LOGIC
        final_result = None

        # A. Gemini Success
        if gemini_coords and verified:
            # Create NEW immutable result
            final_result = {
                **gemini_coords,
                "source": "gemini",
                "confidence": 1.0
            }

        # B. Fallback to OpenCV (If Gemini Failed or was Rejected)
        if not final_result:
            try:
                import opencv_watermark
                frame = cv2.imread(image_path)
                if frame is not None:
                    # Log context
                    if gemini_coords:
                         logger.info("   🧠 Gemini Detected given rejected. Attempting OpenCV Fallback...")
                    else:
                         logger.info("   🧠 Gemini Failed. Attempting OpenCV Fallback...")
                    
                    cv_results = opencv_watermark.detect(frame)
                    if cv_results:
                        # Select Best Candidate
                        if isinstance(cv_results, list):
                            cv_coords = max(cv_results, key=lambda b: b.get("confidence", 0))
                        else:
                            cv_coords = cv_results
                        
                        # Create NEW immutable result
                        final_result = {
                            **cv_coords,
                            "source": "opencv",
                            "confidence": cv_coords.get("confidence", 0.5)
                        }
                        
                        logger.info(f"   ✅ OpenCV Fallback Succeeded: {final_result}")
            except Exception as e:
                logger.warning(f"⚠️ OpenCV Fallback failed: {e}")

        return final_result

    except Exception as e:
        logger.warning(f"⚠️ Gemini detection crashed: {e}")
        # FALLBACK: Try Adaptive OpenCV (if Gemini crashed)
        try:
            import opencv_watermark
            frame = cv2.imread(image_path)
            if frame is not None:
                logger.info("   🧠 Attempting Adaptive OpenCV Fallback (Gemini Crash)...")
                cv_results = opencv_watermark.detect(frame)
                if cv_results:
                    if isinstance(cv_results, list):
                        cv_coords = max(cv_results, key=lambda b: b.get("confidence", 0))
                    else:
                        cv_coords = cv_results
                    
                    return {
                        **cv_coords,
                        "source": "opencv",
                        "confidence": cv_coords.get("confidence", 0.5)
                    }
        except: pass
        return None

def process_video_with_watermark(input_path: str, output_path: str, retry_mode: bool = False) -> Dict:
    """
    Main processing function using Hybrid Vision.
    """
    logger.info(f"🛡️ Processing Watermark: {input_path}")
    
    # 1. Extract Frame with Unique Name to prevent caching/stale reads
    import time
    unique_id = f"{int(time.time()*1000)}_{os.path.basename(input_path)}"
    temp_frame = os.path.join(TEMP_DIR, f"frame_{unique_id}.jpg")
    
    if not extract_frame(input_path, temp_frame):
        logger.error("❌ Failed to extract frame")
        return {"success": False, "context": None} # Return valid dict structure
        
    try:
        # 2. Analyze Frame (ALWAYS RUNS)
        frame = cv2.imread(temp_frame)
        if frame is None:
            logger.error("❌ Failed to load frame")
            return {"success": False, "context": None}
            
        gemini_coords = detect_watermark_with_gemini(temp_frame, original_video_path=input_path, retry_mode=retry_mode)
        
        decision = {"action": "none", "logs": []}
        
        # Override with Gemini coordinates ONLY if verified
        if gemini_coords:
            decision["action"] = "replace"
            # Use detected object directly (Perserve Source/Confidence)
            decision["overlay"] = {
                **gemini_coords,
                "text": os.getenv("MY_WATERMARK_TEXT", "swargawasal"),
                "opacity": 0.8
            }
            
            logger.info(f"🎯 Using VERIFIED coordinates (Source: {gemini_coords.get('source', 'unknown')})")
        else:
            logger.info("ℹ️ No watermark detected.")
        
        logger.info(f"🤖 Final Decision: {decision['action']}")
            
        # 3. Execute Action
        action = decision["action"]
        
        # Check for Smart Crop first (if available and preferred)
        enable_smart_crop = os.getenv("ENABLE_SMART_CROP", "no").lower() == "yes"
        
        if enable_smart_crop and "crop_rect" in decision and decision["crop_rect"]:
            # Smart Crop
            rect = decision["crop_rect"]
            logger.info(f"   ✂️ Smart Cropping: {rect}")
            
            crop_filter = f"crop={rect['w']}:{rect['h']}:{rect['x']}:{rect['y']}"
            
            cmd = [
                FFMPEG_BIN, "-y", "-i", input_path,
                "-vf", crop_filter,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy",
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        elif action == "remove":
            # Use delogo
            delogo_filter = decision["ffmpeg_delogo"]
            logger.info(f"   🧹 Removing watermark: {delogo_filter}")
            
            cmd = [
                FFMPEG_BIN, "-y", "-i", input_path,
                "-vf", delogo_filter,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy",
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        elif action == "replace":
            # Use overlay
            overlay = decision["overlay"]
            logger.info(f"   🔄 Replacing watermark with: {overlay['text']}")
            
            # Use text_overlay module for safe application
            from text_overlay import overlay_engine
            font_path = overlay_engine._get_font_path()
            safe_text = overlay_engine._escape_text(overlay["text"])
            
            # Draw text with strong border (No Box, No Blur)
            drawtext = (
                f"drawtext=fontfile='{font_path}':text='{safe_text}':"
                f"fontsize=60:fontcolor=white:alpha={overlay['opacity']}:"
                f"x={overlay['x']}:y={overlay['y']}:"
                f"borderw=3:bordercolor=black:shadowx=2:shadowy=2"
            )
            
            cmd = [
                FFMPEG_BIN, "-y", "-i", input_path,
                "-vf", drawtext,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy",
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Output preserved from overlay command
            pass
            
        # Return success + context
        return {
            "success": os.path.exists(output_path),
            "context": {
                "frame_path": temp_frame,
                "coords": decision.get("overlay", {}) if action == "replace" else gemini_coords,
                "action": action
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Watermark processing failed: {e}")
        # Fallback: copy
        shutil.copy(input_path, output_path)
        return {"success": False, "context": None}
    # NOTE: We do NOT delete temp_frame here anymore. 
    # It is deleted in confirm_learning() or by a cleanup job.

def confirm_learning(context: dict, is_positive: bool):
    """
    Called by main.py when user approves/rejects the result.
    """
    if not context or not context.get("frame_path") or not context.get("coords"):
        return
        
    frame_path = context["frame_path"]
    coords = context["coords"]
    
    if not os.path.exists(frame_path):
        logger.warning(f"⚠️ Frame expired, cannot learn: {frame_path}")
        return

    try:
        import opencv_watermark
        frame = cv2.imread(frame_path)
        if frame is not None:
            # 1. Save Template (Visual Memory)
            opencv_watermark.learn(frame, coords, is_positive=is_positive)
            
            # 2. Log Features (ML Memory)
            # We treat user confirmation as high confidence (1.0)
            features = opencv_watermark.WatermarkLearner.extract_features(
                frame, coords, match_data={"confidence": 1.0, "matches": 100}
            )
            if features:
                opencv_watermark.WatermarkLearner.log_feedback(features, 1 if is_positive else 0)
                
            logger.info(f"✅ User Feedback: Learned as {'POSITIVE' if is_positive else 'NEGATIVE'}")
    except Exception as e:
        logger.error(f"❌ Failed to learn from feedback: {e}")
    finally:
        # Cleanup
        if os.path.exists(frame_path):
            os.remove(frame_path)

# Compatibility Wrappers for existing code
def apply_my_watermark(input_video: str, output_video: str, watermark_path: str, coords: Dict, opacity: float = 0.8) -> bool:
    """
    Overlays a custom watermark image over the detected watermark area.
    """
    try:
        logger.info(f"🔄 Replacing watermark with image: {watermark_path}")
        
        # Scale watermark to fit the detected box
        # overlay=x:y
        # We need to scale the watermark image first? 
        # For simplicity, let's just overlay it at the coordinates.
        # Ideally, we should resize it to coords['w'] x coords['h']
        
        cmd = [
            FFMPEG_BIN, "-y", "-i", input_video, "-i", watermark_path,
            "-filter_complex", 
            f"[1:v]scale={coords['w']}:{coords['h']}[wm];[0:v][wm]overlay={coords['x']}:{coords['y']}:format=auto",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            output_video
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply image watermark: {e}")
        return False

def apply_text_watermark(input_video: str, output_video: str, text: str, coords: Dict, opacity: float = 0.8) -> bool:
    """
    Overlays a custom text watermark over the detected watermark area.
    """
    try:
        logger.info(f"🔄 Replacing watermark with text: {text}")
        
        # Use text_overlay module for font path
        from text_overlay import overlay_engine
        font_path = overlay_engine._get_font_path()
        safe_text = overlay_engine._escape_text(text)
        
        # Calculate font size to fit width?
        # Rough estimate: width / num_chars * 1.5
        font_size = int(coords['w'] / max(1, len(text)) * 1.8)
        font_size = max(20, min(font_size, coords['h'])) # Clamp
        
        # Draw text with strong border (No Box, No Blur)
        drawtext = (
            f"drawtext=fontfile='{font_path}':text='{safe_text}':"
            f"fontsize={font_size}:fontcolor=white:alpha={opacity}:"
            f"borderw=3:bordercolor=black:shadowx=2:shadowy=2:"
            f"x={coords['x']}+(w-text_w)/2:y={coords['y']}+(h-text_h)/2"
        )
        
        # Just use drawtext
        vf = drawtext
        
        cmd = [
            FFMPEG_BIN, "-y", "-i", input_video,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            output_video
        ]
        logger.info(f"   📝 Running Watermark Replacement CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to apply text watermark: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to apply text watermark: {e}")
        return False

def detect_watermark(frame_path: str, template_path: Optional[str] = None) -> Optional[Dict]:
    """
    Legacy wrapper for watermark detection.
    Uses OpenCV heuristics as a fallback when Hybrid Vision is unavailable.
    """
    try:
        import opencv_watermark
        frame = cv2.imread(frame_path)
        if frame is None:
            return None
            
        # Run OpenCV Detection
        # Updated to use the new ORB-based detector which returns a single best match
        results = opencv_watermark.detect(frame)
        
        if results:
            if isinstance(results, list):
                result = max(results, key=lambda b: b.get("confidence", 0))
            else:
                result = results
            logger.info(f"   ✅ Legacy OpenCV Detection found watermark: {result}")
            return result
            
        return None
        
    except Exception as e:
        logger.error(f"Legacy detection failed: {e}")
        return None

def remove_watermark(input_video: str, output_video: str, coords: Dict) -> bool:
    """Legacy wrapper for watermark removal."""
    return process_video_with_watermark(input_video, output_video)
