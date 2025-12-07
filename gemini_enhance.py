"""
Gemini AI Video Orchestrator Module (Hybrid Mode)
Analyzes video frames and outputs JSON instructions for FFmpeg.
Acts as a "Decision Engine" for the Hybrid Enhancement System.
"""

import os
import cv2
import base64
import logging
import json
import re
import numpy as np
import subprocess
from typing import Optional, Dict, Any

logger = logging.getLogger("gemini_orchestrator")

# Try to import Gemini
try:
    import google.generativeai as genai
    from PIL import Image
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("⚠️ google-generativeai not installed. Gemini orchestrator disabled.")

# Check for Torch/GPU availability for AUTO mode
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

# Configuration
ENABLE_GEMINI_ENHANCE = os.getenv("ENABLE_GEMINI_ENHANCE", "auto").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite-preview-09-2025")

gemini_client = None

def init_gemini(api_key: str) -> bool:
    global gemini_client
    if not HAS_GEMINI or not api_key: return False
    
    try:
        genai.configure(api_key=api_key)
        gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"✅ Gemini Orchestrator initialized: {GEMINI_MODEL}")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini init failed: {e}")
        return False

def frame_to_base64(frame: np.ndarray) -> Optional[str]:
    try:
        # Resize for analysis speed (max 1024px width)
        h, w = frame.shape[:2]
        if w > 1024:
            scale = 1024 / w
            frame = cv2.resize(frame, (1024, int(h * scale)))
            
        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success: return None
        return base64.b64encode(buffer).decode('utf-8')
    except:
        return None

def get_hybrid_prompt() -> str:
    return """
You are an Elite Video Enhancement Architect.
Your task is to analyze this video frame and generate a JSON recipe for FFmpeg enhancement.

Output MUST be valid JSON with this EXACT schema:
{
  "enhance": true,
  "sharpness": 0.0 to 1.0,      // Amount of unsharp mask (0.0=none, 1.0=strong)
  "denoise": 0.0 to 1.0,        // Amount of noise reduction (0.0=none, 1.0=strong)
  "contrast": 0.0 to 1.0,       // Contrast adjustment (1.0=neutral, >1.0=boost)
  "brightness": -0.15 to 0.15,  // Brightness shift (0.0=neutral)
  "saturation": 0.0 to 1.0,     // Saturation adjustment (1.0=neutral, >1.0=boost)
  "skin_protect": true/false,   // If true, be conservative with sharpening on faces
  "upscale": "1x" or "2x",      // Recommended upscale factor
  "ffmpeg_filter": "string"     // Optional custom FFmpeg filter string (e.g. "curves=strong_contrast")
}

INSTRUCTIONS:
1. Analyze lighting, noise, and sharpness.
2. If the image is dark, boost brightness slightly (e.g., 0.05).
3. If the image is noisy/grainy, increase denoise (e.g., 0.3-0.6).
4. If the image is soft/blurry, increase sharpness (e.g., 0.5-0.8).
5. If colors are dull, boost saturation (e.g., 1.1-1.3).
6. Be subtle! Over-processing looks bad.
7. Return ONLY JSON. No markdown, no explanations.
"""

def clean_json_response(text: str) -> str:
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Fix: Gemini sometimes outputs JS objects keys without quotes (e.g. { key: "value" })
    # This regex looks for word characters followed by a colon, not inside quotes
    # Note: This is a simple heuristic. For complex nested JSON it might be imperfect but covers simple objects.
    # We find keys like ` key:` and replace with `"key":`
    text = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', text)
    
    return text.strip()

def analyze_frame(frame: np.ndarray) -> Dict[str, Any]:
    """
    Analyze a single frame and return instructions.
    """
    global gemini_client
    if not gemini_client: return None
    
    try:
        b64_frame = frame_to_base64(frame)
        if not b64_frame: return None
        
        prompt = get_hybrid_prompt()
        
        # Define safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Call Gemini
        response = gemini_client.generate_content(
            contents=[
                {'mime_type': 'image/jpeg', 'data': b64_frame},
                prompt
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Low temp for deterministic JSON
                max_output_tokens=1024,
                response_mime_type="application/json" # Force JSON mode
            ),
            safety_settings=safety_settings
        )
        
        # Parse JSON
        try:
            json_str = clean_json_response(response.text)
            instructions = json.loads(json_str)
            return instructions
        except ValueError:
            logger.warning(f"⚠️ Gemini blocked content or invalid JSON.")
            return None
        except Exception as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return None

def detect_watermark(frame: np.ndarray, keywords: str = None) -> Optional[Dict[str, Any]]:
    """
    Detect watermark in frame using Gemini Vision.
    Returns: {'x': int, 'y': int, 'w': int, 'h': int, 'text': str} or None
    """
    global gemini_client
    if not gemini_client:
        init_gemini(os.getenv("GEMINI_API_KEY"))
    
    if not gemini_client:
        logger.warning("⚠️ Gemini client not initialized for watermark detection.")
        return None
    
    try:
        b64_frame = frame_to_base64(frame)
        if not b64_frame: return None
        
        if keywords:
            logger.info(f"   🗝️ Gemini Prompt using Dynamic Keywords: {keywords}")
            target_instruction = f'Specifically check for these likely watermarks: {keywords}.'
        else:
            logger.info("   🗝️ Gemini Prompt using DEFAULT keywords (No metadata found).")
            target_instruction = 'Specifically check for text like "FILMYGYAN" or any other logo/handle.'

        prompt = f"""
        Analyze this image for watermarks, logos, or text overlays.
        
        Specific Targets to Look For: {keywords if keywords else "FILMYGYAN, Brand Logos, @Handles"}
        
        CRITICAL INSTRUCTIONS:
        1. FIND ALL watermarks/logos. There might be MULTIPLE (e.g. Top-Left AND Bottom-Right).
        2. DETECT text handles (like @name, FILMYGYAN, Voompla) even if they overlap the person/dress.
        3. DISTINGUISH between "Content Captions" (subtitles, speech) and "Brand Watermarks" (channel names).
           - Captions are usually centered at the bottom.
           - Watermarks are usually persistent and indicate OWNERSHIP.
           - IF IN DOUBT, and it matches a Target, IT IS A WATERMARK.
        4. IGNORE natural features (eyes, mouths, buttons).
        5. Include the FULL extent of the watermark: Text + Icon/Logo + Background Box.
        
        {target_instruction}
        
        Return JSON ONLY:
        {{
            "watermarks": [
                {{
                    "box_2d": [ymin, xmin, ymax, xmax], // Normalized 0-1 coordinates
                    "text": "detected text or description",
                    "confidence": 0.0 to 1.0
                }}
            ]
        }}
        """
        
        # Define safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Retry loop for robustness
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = gemini_client.generate_content(
                    contents=[
                        {'mime_type': 'image/jpeg', 'data': b64_frame},
                        prompt
                    ],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                        response_mime_type="application/json"
                    ),
                    safety_settings=safety_settings
                )
                
                json_str = clean_json_response(response.text)
                data = json.loads(json_str)
                
                results = []
                wm_list = data.get("watermarks", [])
                if isinstance(wm_list, list):
                    h_img, w_img = frame.shape[:2]
                    
                    for item in wm_list:
                        if item.get("box_2d"):
                            ymin, xmin, ymax, xmax = item["box_2d"]
                            
                            x = int(xmin * w_img)
                            y = int(ymin * h_img)
                            w = int((xmax - xmin) * w_img)
                            h = int((ymax - ymin) * h_img)
                            
                            # Perfect Fit: Reduce padding to ABSOLUTE MINIMAL (Tightest fit possible)
                            # User requested "perfect fit", so strictly 1% or 2px.
                            pad_w = max(2, int(w * 0.01))
                            pad_h = max(2, int(h * 0.01))
                            
                            x = max(0, x - pad_w)
                            y = max(0, y - pad_h)
                            w = min(w_img - x, w + 2*pad_w)
                            h = min(h_img - y, h + 2*pad_h)
                            
                            results.append({
                                "x": x, "y": y, "w": w, "h": h,
                                "text": item.get("text", ""),
                                "confidence": item.get("confidence", 0.9)
                            })
                    
                    if results:
                        return results # Return list
                        
                return None
                
            except Exception as e:
                err_str = str(e)
                if "400" in err_str or "API key not valid" in err_str or "403" in err_str:
                     logger.error(f"❌ Gemini Critical Error (Aborting Retries): {e}")
                     return None
                     
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Gemini JSON Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                    import time
                    time.sleep(1)
                else:
                    logger.error(f"❌ Gemini Watermark Detect failed after retries: {e}")
                    return None
            
    except Exception as e:
        logger.error(f"❌ Gemini Watermark Detect failed: {e}")
        return None

def verify_watermark(frame: np.ndarray, candidate_box: Dict[str, int]) -> bool:
    """
    Stage 1: Validation.
    Asks Gemini if the candidate box actually contains a watermark/logo.
    """
    global gemini_client
    if not gemini_client: return True # Fail open if no Gemini (trust OpenCV)
    
    try:
        # Crop the candidate region with some context
        x, y, w, h = candidate_box['x'], candidate_box['y'], candidate_box['w'], candidate_box['h']
        h_img, w_img = frame.shape[:2]
        
        pad_x = int(w * 0.5)
        pad_y = int(h * 0.5)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        
        roi = frame[y1:y2, x1:x2]
        b64_frame = frame_to_base64(roi)
        if not b64_frame: return True
        
        prompt = """
        Analyze this image crop. The center of this image was detected as a potential watermark.
        Confirm if there is a digital overlay, logo, text handle (e.g. @name, JATIN EDIT), or brand mark present.

        INSTRUCTIONS:
        1. Focus on the CENTER of the crop.
        2. ANY text overlay that looks like a creator name, handle, or channel name IS a watermark.
        3. Ignore natural text (e.g. on street signs or t-shirts) UNLESS it looks like a digital overlay.
        4. If unsure, lean towards TRUE.

        Return JSON ONLY:
        {
            "is_watermark": true/false,
            "confidence": 0.0 to 1.0
        }
        """
        
        # Safety Settings (BLOCK_NONE to avoid skipping on hot/suggestive content)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = gemini_client.generate_content(
            contents=[{'mime_type': 'image/jpeg', 'data': b64_frame}, prompt],
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
            safety_settings=safety_settings
        )
        
        # Safe Access to Response
        safe_text = None
        try:
            safe_text = response.text
        except Exception:
            # Handle Safety Block (Finish Reason 2)
            try:
                if response.candidates and response.candidates[0].finish_reason == 2:
                    logger.warning("⚠️ Gemini Safety Block (Sexual/Harassment filter). Assuming Valid Watermark.")
            except: pass
            
        if safe_text:
            data = json.loads(clean_json_response(safe_text))
            is_wm = data.get("is_watermark", False)
            logger.info(f"🧠 Gemini Verification: {'✅ Valid' if is_wm else '❌ Invalid'} (Conf: {data.get('confidence', 0)})")
            return is_wm
        else:
            logger.warning("⚠️ Gemini Safety Block/Empty. Returning None (Uncertain).")
            return None 
        
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "Quota" in err_msg:
             logger.warning(f"⚠️ Gemini Quota Exceeded. Returning None to trigger Local Check.")
        elif "404" in err_msg:
             logger.warning(f"⚠️ Gemini Model 404 Error (Model not found). Returning None to trigger Local Check.")
        else:
             logger.warning(f"⚠️ Verification failed: {e}")
             
        return None # Return None to indicate failure (neither True nor False)

def refine_watermark(frame: np.ndarray, rough_box: Dict[str, int]) -> Optional[Dict[str, int]]:
    """
    Stage 2: Refinement.
    Uses a high-res crop to find the EXACT pixel coordinates of the watermark.
    """
    global gemini_client
    if not gemini_client: return rough_box
    
    try:
        # Crop with moderate padding to ensure we capture the whole thing but keep resolution high
        x, y, w, h = rough_box['x'], rough_box['y'], rough_box['w'], rough_box['h']
        h_img, w_img = frame.shape[:2]
        
        # 50% padding
        pad_x = int(w * 0.5)
        pad_y = int(h * 0.5)
        
        crop_x = max(0, x - pad_x)
        crop_y = max(0, y - pad_y)
        crop_w = min(w_img - crop_x, w + 2*pad_x)
        crop_h = min(h_img - crop_y, h + 2*pad_y)
        
        roi = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # Don't resize if possible, or resize minimally
        success, buffer = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success: return rough_box
        b64_frame = base64.b64encode(buffer).decode('utf-8')
        
        prompt = """
        Analyze this crop VERY PRECISELY. Find the exact bounding box of the watermark/logo/text.
        Return coordinates relative to this image (0-1).
        DETERMINE THE VISUAL TYPE:
        - TEXT_WHITE: White or light gray text.
        - LOGO_COLORED: Opaque colored logo/icon.
        - LOGO_TRANSPARENT: Semi-transparent/ghostly logo or text.
        
        Return JSON ONLY:
        {
            "box_2d": [ymin, xmin, ymax, xmax],
            "type": "TEXT_WHITE"
        }
        """
        
        # Safety Settings (BLOCK_NONE to avoid skipping on hot/suggestive content)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = gemini_client.generate_content(
            contents=[{'mime_type': 'image/jpeg', 'data': b64_frame}, prompt],
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
            safety_settings=safety_settings
        )
        
        # Safe Access
        safe_text = None
        try:
            safe_text = response.text
        except Exception:
            try:
                 if response.candidates and response.candidates[0].finish_reason == 2:
                     logger.warning("⚠️ Gemini Safety Block (Refinement). Using Rough Box.")
            except: pass
            
        if safe_text:
            data = json.loads(clean_json_response(safe_text))
        if data.get("box_2d"):
            ymin, xmin, ymax, xmax = data["box_2d"]
            wm_type = data.get("type", "TEXT_WHITE")
            
            # Map back to full frame
            refined_x = int(crop_x + (xmin * crop_w))
            refined_y = int(crop_y + (ymin * crop_h))
            refined_w = int((xmax - xmin) * crop_w)
            refined_h = int((ymax - ymin) * crop_h)
            
            # PERFECT FIT: ZERO PADDING
            # We want the exact pixels.
            refined_x = max(0, refined_x)
            refined_y = max(0, refined_y)
            # No extra w/h padding
            refined_w = refined_w 
            refined_h = refined_h
            
            # CLAMP to image bounds (CRITICAL FIX)
            refined_x = max(0, min(w_img - 1, refined_x))
            refined_y = max(0, min(h_img - 1, refined_y))
            refined_w = max(1, min(w_img - refined_x, refined_w))
            refined_h = max(1, min(h_img - refined_y, refined_h))
            
            logger.info(f"🔍 Gemini Refined Box: {refined_x},{refined_y} {refined_w}x{refined_h} Type: {wm_type}")
            
            return {"x": refined_x, "y": refined_y, "w": refined_w, "h": refined_h, "type": wm_type}
            
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "Quota" in err_msg:
             logger.warning(f"⚠️ Gemini Quota Exceeded during Refinement. Using ROUGH BOX.")
        else:
             logger.warning(f"⚠️ Refinement failed: {e}")
        
    return rough_box

def run(input_video: str, output_video: str) -> str:
    """
    Orchestrator Entry Point.
    1. Analyzes representative frame.
    2. Generates FFmpeg command based on JSON.
    3. Executes FFmpeg.
    """
    # Init
    if not gemini_client:
        if not init_gemini(os.getenv("GEMINI_API_KEY")):
            return "GEMINI_FAIL"
            
    try:
        logger.info("🤖 Gemini Hybrid Mode: Analyzing video...")
        
        # Extract frames for analysis
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return "GEMINI_FAIL"
        
        # --- OBJ 1: AUTO MODE LOGIC ---
        # If AUTO, only run if NO GPU is detected
        if ENABLE_GEMINI_ENHANCE == "auto" and HAS_GPU:
            cap.release()
            logger.info("🤖 Gemini Auto: GPU detected, skipping Gemini enhancement to prefer Heavy Editor.")
            return "GEMINI_FAIL"
        
        # --- OBJ 4: MULTI-FRAME ANALYSIS ---
        frames_indices = [
            int(total_frames * 0.1),
            int(total_frames * 0.5),
            int(total_frames * 0.9)
        ]
        
        recipe_list = []
        
        for idx in frames_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f_test = cap.read()
            if ret:
                r = analyze_frame(f_test)
                if r: recipe_list.append(r)
                
        cap.release()
        
        if not recipe_list: return "GEMINI_FAIL"
        
        # Merge Results strictly as requested
        # sharpness = median
        # denoise = max
        # brightness = average
        # saturation = median
        
        sharps = [float(r.get("sharpness", 0)) for r in recipe_list]
        denoises = [float(r.get("denoise", 0)) for r in recipe_list]
        brights = [float(r.get("brightness", 0)) for r in recipe_list]
        sats = [float(r.get("saturation", 1.0)) for r in recipe_list]
        skin_flags = [r.get("skin_protect", False) for r in recipe_list]
        
        # Helper: Median
        def get_median(lst):
            lst = sorted(lst)
            n = len(lst)
            if n == 0: return 0
            if n % 2 == 1: return lst[n//2]
            return (lst[n//2 - 1] + lst[n//2]) / 2.0
            
        final_sharp = get_median(sharps)
        final_denoise = max(denoises) if denoises else 0
        final_bright = sum(brights) / len(brights) if brights else 0
        final_sat = get_median(sats)
        final_skin = any(skin_flags) # Conservative: if any frame needs skin protect, use it.
        final_upscale = "1x"
        if any(r.get("upscale") == "2x" for r in recipe_list):
            final_upscale = "2x"
            
        instructions = {
            "enhance": True, # If we got here, we try to enhance
            "sharpness": final_sharp,
            "denoise": final_denoise,
            "contrast": get_median([float(r.get("contrast", 1.0)) for r in recipe_list]),
            "brightness": final_bright,
            "saturation": final_sat,
            "skin_protect": final_skin,
            "upscale": final_upscale
        }
        
        # Normalize Gemini schema inconsistencies
        if final_sat <= 1.0:
             final_sat = 1.0 + final_sat * 0.5  # map 0–1 → 1.0–1.5
             
        # Need to re-inject normalized sat into instructions for logging consistency
        instructions["saturation"] = final_sat

        logger.info(f"📋 Gemini Recipe (Merged): {json.dumps(instructions, indent=2)}")

        # --- OBJ 2: NUMERIC SANITY GUARD ---
        # Contrast
        cont = float(instructions.get("contrast", 1.0))
        if cont <= 1.0:
            cont = 1.0 + (cont - 1.0) * 0.5 if cont != 1.0 else 1.0 # Normalize schema
            
        if cont < 0.5: cont = 1.0
        cont = max(0.8, min(1.5, cont))
        
        # Brightness
        bright = float(instructions.get("brightness", 0.0))
        bright = max(-0.2, min(0.2, bright))
        
        # Saturation
        sat = float(instructions.get("saturation", 1.0))
        if sat < 0.9: sat = 1.0
        sat = max(1.0, min(2.0, sat))
        
        # --- OBJ 3: SKIN PROTECT ---
        sharp = float(instructions.get("sharpness", 0))
        sharp = max(0.0, min(1.0, sharp))
        
        if instructions.get("skin_protect", False):
            if sharp > 0.6:
                logger.info("🛡️ Skin Protect Active: Reducing sharpness.")
                sharp *= 0.6 # Reduce by 40%
            # Also clamp denoise to preserve texture
            final_denoise = min(final_denoise, 0.4)
            
        # --- OBJ 5: CONFIDENCE GATE ---
        if sharp < 0.1 and final_denoise < 0.1 and sat < 1.05 and abs(bright) < 0.05 and abs(cont - 1.0) < 0.05:
            logger.info(
                f"🛑 Gemini Gate: sharp={sharp:.2f}, denoise={final_denoise:.2f}, "
                f"sat={sat:.2f}, bright={bright:.2f}, cont={cont:.2f}"
            )
            return "GEMINI_FAIL"
            
        # --- CONSTRUCT FFMPEG COMMAND ---
        filters = []
        
        # 1. Sharpness (unsharp)
        if sharp > 0:
            # 5:5 is standard size. Amount 0.0 to 1.5
            amount = sharp * 1.5 
            filters.append(f"unsharp=5:5:{amount:.2f}:5:5:0.0")
            
        # 2. Denoise (hqdn3d)
        denoise = final_denoise
        if denoise > 0:
            # Map 0.0-1.0 to 0-10 strength
            val = denoise * 10
            filters.append(f"hqdn3d={val:.1f}:{val:.1f}:6:6")
            
        # 3. Color/Contrast (eq)
        # Already clamped above
        if cont != 1.0 or bright != 0.0 or sat != 1.0:
            filters.append(f"eq=contrast={cont:.2f}:brightness={bright:.2f}:saturation={sat:.2f}")
            
        # 4. Upscale
        if final_upscale == "2x":
             filters.append("scale=iw*2:ih*2:flags=lanczos")
        else:
             filters.append("scale=1080:1920:flags=lanczos:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2")

        # Build Command
        vf_chain = ",".join(filters) if filters else "null"
        
        # If no filters, just copy (but we usually have scale)
        if not filters: vf_chain = "scale=1080:1920:flags=lanczos:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"

        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", vf_chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-metadata", f"gemini_recipe=sharp={sharp:.2f} denoise={denoise:.2f} sat={sat:.2f} bright={bright:.2f}",
            "-c:a", "copy",
            output_video
        ]
        
        logger.info(f"⚡ Executing Hybrid FFmpeg: {vf_chain}")
        
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return "SUCCESS"
        
    except Exception as e:
        logger.error(f"❌ Hybrid Orchestrator failed: {e}")
        return "GEMINI_FAIL"
