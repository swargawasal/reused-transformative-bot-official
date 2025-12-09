# compiler.py - HIGH-END MULTI-PASS AI EDITOR (DUAL-STAGE ENGINE)
import os
import subprocess
import logging
import shutil
import sys
import random
import json
import glob
import time
import platform
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv(".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("compiler")

from router import run_enhancement
from watermark_auto import (
    extract_frame,
    detect_watermark,
    remove_watermark,
    apply_my_watermark,
    apply_text_watermark
)



# Import Text Overlay
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
if not shutil.which(FFMPEG_BIN):
    FFMPEG_BIN = "ffmpeg"

FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
if not shutil.which(FFPROBE_BIN):
    FFPROBE_BIN = "ffprobe"

try:
    from text_overlay import apply_text_overlay_safe
    HAS_TEXT_OVERLAY = True
except ImportError:
    HAS_TEXT_OVERLAY = False

# Configuration
COMPUTE_MODE = os.getenv("COMPUTE_MODE", "auto").lower()
ENHANCEMENT_LEVEL = os.getenv("ENHANCEMENT_LEVEL", "2x").lower()
TRANSITION_DURATION = float(os.getenv("TRANSITION_DURATION", "1.0"))
TRANSITION_INTERVAL = float(os.getenv("TRANSITION_INTERVAL", "10"))
TARGET_RESOLUTION = os.getenv("TARGET_RESOLUTION", "1080:1920")
REENCODE_CRF = os.getenv("REENCODE_CRF", "23") 
REENCODE_PRESET = os.getenv("REENCODE_PRESET", "veryfast")

# AI Config
FACE_ENHANCEMENT = os.getenv("FACE_ENHANCEMENT", "yes").lower() == "yes"
USE_ADVANCED_ENGINE = os.getenv("USE_ADVANCED_ENGINE", "off").lower() == "on"

# Transformative Features Config
ADD_TEXT_OVERLAY = os.getenv("ADD_TEXT_OVERLAY", "yes").lower() == "yes" # Updated key
ADD_COLOR_GRADING = os.getenv("ADD_COLOR_GRADING", "yes").lower() == "yes"
ADD_SPEED_RAMPING = os.getenv("ADD_SPEED_RAMPING", "yes").lower() == "yes"
FORCE_AUDIO_REMIX = os.getenv("FORCE_AUDIO_REMIX", "yes").lower() == "yes"

# Text Overlay Settings
TEXT_OVERLAY_TEXT = os.getenv("TEXT_OVERLAY_CONTENT", "swargawasal") # Updated key
TEXT_OVERLAY_POSITION = os.getenv("TEXT_OVERLAY_POSITION", "bottom")
TEXT_OVERLAY_SIZE = int(os.getenv("TEXT_OVERLAY_SIZE", "60"))

# Color Grading Settings
COLOR_FILTER = os.getenv("COLOR_FILTER", "cinematic")
COLOR_INTENSITY = float(os.getenv("COLOR_INTENSITY", "0.5"))

# Speed Ramping Settings
SPEED_VARIATION = float(os.getenv("SPEED_VARIATION", "0.15"))

# Audio Remix Settings
ENABLE_HEAVY_REMIX_SHORTS = os.getenv("ENABLE_HEAVY_REMIX_SHORTS", "yes").lower() == "yes"
ENABLE_HEAVY_REMIX_COMPILATION = os.getenv("ENABLE_HEAVY_REMIX_COMPILATION", "yes").lower() == "yes"
AUTO_MUSIC = os.getenv("AUTO_MUSIC", "yes").lower() == "yes"
MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.4"))
ORIGINAL_AUDIO_VOLUME = float(os.getenv("ORIGINAL_AUDIO_VOLUME", "1.0"))

# Watermark Config
WATERMARK_DETECTION = os.getenv("WATERMARK_DETECTION", "yes").lower() == "yes"
# Support 'yes', 'no', 'auto'
WATERMARK_REMOVE_MODE = os.getenv("ENABLE_WATERMARK_REMOVAL", "yes").lower()
WATERMARK_REPLACE_MODE = os.getenv("ENABLE_WATERMARK_REPLACEMENT", "yes").lower()

WATERMARK_REMOVE = WATERMARK_REMOVE_MODE in ["yes", "auto", "true"]
WATERMARK_REPLACE = WATERMARK_REPLACE_MODE in ["yes", "auto", "true"]
MY_WATERMARK_FILE = os.getenv("WATERMARK_REPLACE_PATH", "assets/watermark.png") # Updated key
MY_WATERMARK_TEXT = os.getenv("MY_WATERMARK_TEXT", "swargawasal")
MY_WATERMARK_OPACITY = float(os.getenv("MY_WATERMARK_OPACITY", "0.80"))

TEMP_DIR = "temp"
OUTPUT_DIR = "Processed Shorts"
COMPILATION_DIR = "final_compilations"
TOOLS_DIR = os.path.join(os.getcwd(), "tools")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPILATION_DIR, exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def _run_command(cmd: List[str], check: bool = False, timeout: int = None) -> bool:
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {cmd[0]}")
        return False
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return False

def _get_video_info(path: str) -> Dict:
    try:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json", path
        ]
        result = subprocess.check_output(cmd).decode().strip()
        data = json.loads(result)
        stream = data["streams"][0]
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "duration": float(stream.get("duration", 0))
        }
    except Exception:
        return {"width": 0, "height": 0, "duration": 0}

def verify_video_integrity(file_path: str) -> bool:
    """
    Perform automated QA on the output video.
    Checks:
    1. File existence and non-zero size.
    2. Valid video stream (via ffprobe).
    3. Duration > 0.
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ QA Failed: File not found: {file_path}")
        return False
        
    if os.path.getsize(file_path) == 0:
        logger.error(f"❌ QA Failed: File is empty: {file_path}")
        return False
        
    info = _get_video_info(file_path)
    if info.get("duration", 0) <= 0:
        logger.error(f"❌ QA Failed: Invalid duration ({info.get('duration')}s)")
        return False
        
    if info.get("height", 0) <= 0:
        logger.error(f"❌ QA Failed: Invalid video stream (height=0)")
        return False
        
    logger.info(f"✅ QA Passed: {os.path.basename(file_path)} (Dur: {info['duration']}s)")
    return True

def _upscale_ffmpeg(input_path: str, output_path: str, scale: int):
    vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos" if scale > 1 else "null"
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "copy",
        output_path
    ]
    _run_command(cmd)


def _get_ffmpeg_encoder():
    """
    Detect if NVENC is available and working for hardware acceleration.
    """
    if COMPUTE_MODE == "cpu":
        return "libx264"
        
    try:
        # First check if NVENC is listed
        cmd = [FFMPEG_BIN, "-hide_banner", "-encoders"]
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        if "h264_nvenc" not in result:
            return "libx264"
        
        # Test if NVENC actually works by encoding a dummy frame
        test_cmd = [
            FFMPEG_BIN, "-f", "lavfi", "-i", "color=c=black:s=256x256:d=1",
            "-pix_fmt", "yuv420p",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        subprocess.check_output(test_cmd, stderr=subprocess.STDOUT, timeout=5)
        logger.info("🚀 NVENC (Hardware Acceleration) Detected and Working!")
        return "h264_nvenc"
    except:
        logger.info("ℹ️ NVENC not available or failed test. Using CPU encoding (libx264).")
        return "libx264"

def _get_video_fps(input_path: str) -> float:
    """Get video FPS using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        num, den = map(int, result.stdout.strip().split('/'))
        return num / den
    except Exception as e:
        logger.warning(f"⚠️ Failed to detect FPS: {e}. Defaulting to 30.")
        return 30.0



def normalize_video(input_path: str, output_path: str, target_res: tuple = (1080, 1920)):
    logger.info(f"📏 Normalizing video to {target_res} with Golden Config...")
    encoder = _get_ffmpeg_encoder()
    preset = os.getenv("REENCODE_PRESET", "p4" if encoder == "h264_nvenc" else "superfast")
    
    # Detect FPS
    fps = _get_video_fps(input_path)
    logger.info(f"⏱️ Detected FPS: {fps}")
    
    target_w, target_h = target_res
    
    # Golden Config Filters
    # 1. Scale/Pad to Target Resolution
    # 2. FPS: Dynamic
    # 3. hqdn3d: Light temporal denoise
    # 4. unsharp: Smart sharpen
    # 5. noise: Light film grain (strength 2)
    vf = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"fps={fps},"
        f"hqdn3d=1.5:1.5:6:6,"
        f"unsharp=5:5:0.8:3:3:0.4,"
        f"noise=c0s=2:allf=t"
    )
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", encoder, "-preset", preset,
    ]
    
    if encoder == "libx264":
        cmd.extend(["-crf", "23"])
    else:
        cmd.extend(["-rc", "vbr", "-cq", "19", "-qmin", "19", "-qmax", "19"])

    cmd.extend(["-c:a", "aac", "-ar", "44100", "-ac", "2", output_path])
    
    return _run_command(cmd, check=True)

# ==================== TRANSFORMATIVE FEATURES ====================

def apply_color_grading(input_path: str, output_path: str, filter_type: str, intensity: float):
    """Apply color grading filter to video."""
    logger.info(f"🎨 Applying color grading: {filter_type}")
    
    filters = {
        "cinematic": f"eq=contrast=1.05:brightness=0.0:saturation=1.0,curves=all='0/0 0.5/0.5 1/1'",
        "vintage": f"eq=contrast=1.1:saturation=0.7,colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,curves=all='0/0.1 1/0.9'",
        "vibrant": f"eq=contrast=1.2:saturation={1.0+intensity}:brightness=0.0",
        "dark": f"eq=brightness=-0.1:contrast=1.4:saturation=0.8,curves=all='0/0 0.5/{0.4-intensity*0.1} 1/0.95'",
        "warm": f"colortemperature={6500+int(intensity*2000)}",
        "cool": f"colortemperature={6500-int(intensity*2000)}"
    }
    
    vf = filters.get(filter_type, filters["cinematic"])
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "copy", output_path
    ]
    _run_command(cmd, check=True)


def apply_speed_ramping(input_path: str, output_path: str, variation: float):
    """Apply random speed variations to video."""
    logger.info(f"⚡ Applying speed ramping: ±{variation*100}%")
    
    # Bias towards slow motion to avoid "too fast" feel
    speed = random.uniform(0.92, 1.02)
    
    vf = f"setpts={1/speed}*PTS"
    af = f"atempo={speed}"
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-af", af,
        output_path
    ]
    _run_command(cmd, check=True)

# ==================== TRANSITIONS ====================

def create_transition_clip(seg_a: str, seg_b: str, output_path: str, trans_type: str, duration: float):
    info_a = _get_video_info(seg_a)
    dur_a = info_a['duration']
    start_a = max(0, dur_a - duration)
    
    tail_a = output_path.replace(".mp4", "_tailA.mp4")
    _run_command([FFMPEG_BIN, "-y", "-ss", str(start_a), "-i", seg_a, "-t", str(duration), "-c", "copy", tail_a])
    
    head_b = output_path.replace(".mp4", "_headB.mp4")
    _run_command([FFMPEG_BIN, "-y", "-i", seg_b, "-t", str(duration), "-c", "copy", head_b])
    
    filter_str = f"[0:v][1:v]xfade=transition={trans_type}:duration={duration}:offset=0[v];[0:a][1:a]acrossfade=d={duration}[a]"
    if trans_type == "zoom":
        filter_str = f"[0:v][1:v]xfade=transition=circleopen:duration={duration}:offset=0[v];[0:a][1:a]acrossfade=d={duration}[a]"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", tail_a, "-i", head_b,
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        output_path
    ]
    _run_command(cmd)
    
    if os.path.exists(tail_a): os.remove(tail_a)
    if os.path.exists(head_b): os.remove(head_b)

def compile_with_transitions(input_video: Path, title: str, aggressive_watermark: bool = False) -> Path:
    import audio_processing
    
    input_path = os.path.abspath(str(input_video))
    job_id = f"job_{int(time.time())}"
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        logger.info(f"🚀 Starting Transformative Pipeline for: {title}")
        
        # --- METADATA PROPAGATION ---
        try:
            src_meta = os.path.splitext(input_path)[0] + ".json"
            if os.path.exists(src_meta):
                dst_meta = os.path.join(job_dir, "normalized.json")
                shutil.copy2(src_meta, dst_meta)
                logger.info(f"📝 Propagated metadata to: {dst_meta}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to propagate metadata: {e}")
            
        # Sanitize title for FILENAME only (replace spaces with underscores)
        safe_title = title.replace(" ", "_")
        safe_title = "".join([c for c in safe_title if c.isalnum() or c in ('_', '-')])
        
        # Incremental Renaming Logic
        final_output = os.path.join(OUTPUT_DIR, f"final_{safe_title}.mp4")
        
        for i in range(0, 1000):
            if i == 0:
                candidate_name = f"final_{safe_title}.mp4"
            else:
                candidate_name = f"final_{safe_title}_{i}.mp4"
                
            candidate_path = os.path.join(OUTPUT_DIR, candidate_name)
            if not os.path.exists(candidate_path):
                final_output = candidate_path
                break
        
        # Get video info for smart processing
        video_info = _get_video_info(input_path)
        duration = video_info.get('duration', 0)
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        
        logger.info(f"📊 Video: {width}x{height}, {duration:.1f}s")
        
        wm_context = None
        
        # 1. AI Enhancement
        logger.info("✨ Step 1: AI Enhancement")
        enhanced_video = os.path.join(job_dir, "enhanced.mp4")
        
        # Log resolution
        logger.info(f"   📏 Input Resolution: {width}x{height}")
        
        # Only enhance if resolution is LESS than 1080x1920
        # Skip enhancement if resolution is equal to or greater than 1080x1920
        scale = 2
        if ENHANCEMENT_LEVEL and ENHANCEMENT_LEVEL[0].isdigit():
            scale = int(ENHANCEMENT_LEVEL[0])
        
        # Check for "Native Quality Mode" (1x) or explicit override
        is_quality_mode = (scale == 1) or (ENHANCEMENT_LEVEL.lower() == "quality")
        
        if width >= 1080 and height >= 1920 and not is_quality_mode:
            logger.info("⚡ Resolution is 1080x1920 or higher. Skipping enhancement (already high quality).")
            enhanced_video = input_path
            success = True
        else:
            logger.info(f"📈 Running AI Enhancement (Scale: {scale}x)...")
            success = run_enhancement(input_path, enhanced_video, config=os.environ)
            
        if not success:
            logger.warning("⚠️ Enhancement failed. Using original video.")
            enhanced_video = input_path
        
        # 2. Smart Normalization
        needs_normalization = (width != 1080 or height != 1920)
        
        if needs_normalization:
            logger.info("✨ Step 2: Smart Normalization (9:16)")
            norm_video = os.path.join(job_dir, "normalized.mp4")
            normalize_video(enhanced_video, norm_video, target_res=(1080, 1920))
            current_video = norm_video
        else:
            current_video = enhanced_video
            norm_video = current_video # Ensure norm_video is set
            
        # --- WATERMARK DETECTION & REMOVAL ---
        # Run AFTER normalization so coordinates are consistent (1080x1920)
        if os.getenv("ENABLE_HYBRID_VISION", "yes").lower() == "yes":
            try:
                import hybrid_watermark
                orchestrator = hybrid_watermark.HybridOrchestrator()
                
                # Pass aggressive flag
                wm_result_json = orchestrator.process_video(current_video, aggressive=aggressive_watermark)
                wm_result = json.loads(wm_result_json)
                
                wm_context = wm_result.get('context', {})
                final_watermarks = wm_result.get('watermarks', [])
                
                # Sort: Moving first (complex), then Static (simple)
                if final_watermarks:
                    final_watermarks.sort(key=lambda x: x.get('is_moving', False), reverse=True)
                    # Inject coordinates into context for main.py feedback loop
                    wm_context['coords'] = final_watermarks[0]['coordinates']

                    logger.info(f"💧 Found {len(final_watermarks)} watermarks to process.")
                    
                    masks_to_inpaint = []
                    
                    for i, wm in enumerate(final_watermarks):
                        logger.info(f"   🌊 Processing Watermark #{i+1} (ID: {wm.get('id', 'N/A')})")
                        
                        mask_video = os.path.join(job_dir, f"watermark_mask_{i}.mp4")
                        mask_generated = False
                        
                        try:
                            # Step 2.1: Detect Motion Type
                            trajectory = wm.get('trajectory', [])
                            is_moving = len(trajectory) > 5
                            
                            if is_moving:
                                logger.info("      🏃 Moving Watermark Detected! Generating Dynamic Mask...")
                                
                                # Step 2.2: Template Match Logic
                                template_mask_full = wm.get('template_mask')
                                
                                if template_mask_full is not None:
                                    try:
                                        import hybrid_watermark
                                        # Crop to ROI
                                        tx, ty, tw, th = wm['coordinates']['x'], wm['coordinates']['y'], wm['coordinates']['w'], wm['coordinates']['h']
                                        template_mask_crop = template_mask_full[ty:ty+th, tx:tx+tw].copy()
                                        
                                        mask_generated = hybrid_watermark.generate_dynamic_mask(current_video, trajectory, mask_video, template_mask=template_mask_crop)
                                    except Exception as e:
                                        logger.error(f"      ❌ Failed to prepare template mask: {e}")
                                        mask_generated = hybrid_watermark.generate_dynamic_mask(current_video, trajectory, mask_video)
                                else:
                                     mask_generated = hybrid_watermark.generate_dynamic_mask(current_video, trajectory, mask_video)

                            else:
                                logger.info("      🗿 Static Watermark Detected! Generating Exact-Shape Static Mask...")
                                import hybrid_watermark
                                # Use merged box coordinates
                                box = wm['coordinates']
                                # Inject Type
                                box['type'] = wm.get('type', 'TEXT_WHITE')
                                
                                mask_generated = hybrid_watermark.generate_static_mask(current_video, box, mask_video)
                            
                            # COLLECT SUCCESSFUL MASKS
                            if mask_generated and os.path.exists(mask_video):
                                masks_to_inpaint.append(mask_video)
                                logger.info(f"      ➕ Added to Inpaint Batch: {os.path.basename(mask_video)}")
                            else:
                                # TRIGGER FALLBACK IMMEDIATELY FOR THIS ONE
                                raise Exception("Mask generation failed")

                        except Exception as e:
                            logger.warning(f"      ⚠️ Masking failed ({e}). Running Fallback Delogo...")
                            
                            # FALLBACK: Static Delogo (Rectangular)
                            clean_video_fallback = os.path.join(job_dir, f"clean_static_{i}.mp4")
                            coords = wm['coordinates']
                            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
                            
                            # Clamp & Pad
                            pad = 2
                            max_w, max_h = 1080, 1920
                            x, y = max(2, min(x-pad, max_w-4)), max(2, min(y-pad, max_h-4))
                            w, h = max(4, min(w+pad*2, max_w-x-2)), max(4, min(h+pad*2, max_h-y-2))
                            x, y, w, h = (x//2)*2, (y//2)*2, (w//2)*2, (h//2)*2
                            
                            try:
                                filter_glass = f"[0:v]delogo=x={x}:y={y}:w={w}:h={h}:show=0[out]"
                                cmd_blur = [
                                    FFMPEG_BIN, "-y", "-i", current_video, 
                                    "-filter_complex", filter_glass, "-map", "[out]",
                                    "-c:a", "copy", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                                    clean_video_fallback
                                ]
                                if _run_command(cmd_blur, check=True) and os.path.exists(clean_video_fallback):
                                    current_video = clean_video_fallback
                                    logger.info("      ✅ Fallback Delogo Applied.")
                            except Exception: pass

                        # ... loop ends ...
                        
                    # --- BATCH INPAINTING EXECUTION ---
                    if masks_to_inpaint:
                        logger.info(f"   🎨 Batch Inpainting {len(masks_to_inpaint)} masks...")
                        clean_video_batch = os.path.join(job_dir, "clean_inpainted_batch.mp4")
                        try:
                            import opencv_watermark
                            if opencv_watermark.inpaint_video(current_video, masks_to_inpaint, clean_video_batch):
                                current_video = clean_video_batch
                                logger.info("      ✅ Batch Inpainting Complete.")
                            else:
                                 logger.warning("      ⚠️ Batch Inpaint returned False.")
                        except Exception as e:
                            logger.error(f"      ❌ Batch Inpainting failed: {e}")

                elif wm_result.get('watermark_detected'):
                    logger.info("💧 Single Watermark Detected (Legacy Mode).")
                    pass
                else:
                    logger.info("✅ No watermarks detected.")
            
            except Exception as e:
                logger.error(f"❌ Watermark Detection Failed: {e}")

        # 2.5 Apply Transformative Features
        # current_video is already set correctly (either norm_video or clean_video)
        
        # --- GEMINI CAPTIONS & VOICEOVER ---
        ai_caption_text = None
        # Check AI_CAPTIONS (preferred) or ENABLE_GEMINI_CAPTIONS (legacy)
        enable_ai_captions = os.getenv("AI_CAPTIONS", "yes").lower() == "yes" or \
                             os.getenv("ENABLE_GEMINI_CAPTIONS", "yes").lower() == "yes"
        
        if enable_ai_captions:
            try:
                from gemini_captions import generate_caption_from_video
                logger.info("🧠 Generating AI Caption with Gemini...")
                ai_caption_text = generate_caption_from_video(current_video, style="viral")
                if ai_caption_text:
                    logger.info(f"   ✨ AI Caption: {ai_caption_text}")
            except Exception as e:
                logger.warning(f"⚠️ Gemini Caption failed: {e}")

        # AI Voiceover
        voiceover_path = None
        if os.getenv("ENABLE_MICRO_VOICEOVER", "yes").lower() == "yes" and ai_caption_text:
            try:
                from voiceover import generate_voiceover
                voiceover_path = os.path.join(job_dir, "voiceover.mp3")
                if generate_voiceover(ai_caption_text, voiceover_path):
                    logger.info("   🗣️ AI Voiceover generated.")
                else:
                    voiceover_path = None
            except Exception as e:
                logger.warning(f"⚠️ Voiceover failed: {e}")

        # Text Overlay
        if HAS_TEXT_OVERLAY:
            try:
                # Pass 1: AI Caption (if enabled)
                if ai_caption_text:
                    logger.info(f"✨ Step 2.5a: Applying AI Caption: {ai_caption_text}")
                    text_video_1 = os.path.join(job_dir, "text_overlay_1.mp4")
                    success_1 = apply_text_overlay_safe(
                        current_video,
                        text_video_1,
                        ai_caption_text,
                        lane="caption", # New API: "caption" lane (0.78)
                        size=TEXT_OVERLAY_SIZE
                    )
                    if success_1 and os.path.exists(text_video_1):
                        current_video = text_video_1
                
                # Pass 2: Fixed Text Overlay (if enabled)
                if ADD_TEXT_OVERLAY and TEXT_OVERLAY_TEXT:
                    logger.info(f"✨ Step 2.5b: Applying Fixed Text: {TEXT_OVERLAY_TEXT}")
                    text_video_2 = os.path.join(job_dir, "text_overlay_2.mp4")
                    
                    # Map old logic to new Lanes
                    # If caption exists, use "fixed" (0.88 - Bottom)
                    # If no caption, check env. If env says "bottom", use "fixed". If "center", use "center".
                    
                    target_lane = "fixed" # Default to bottom/branding
                    if TEXT_OVERLAY_POSITION == "center":
                        target_lane = "center"
                    elif TEXT_OVERLAY_POSITION == "top":
                        target_lane = "top"
                        
                    # Force "fixed" if caption is present to prevent overlap (unless top/center explicit)
                    if ai_caption_text and target_lane == "caption":
                        target_lane = "fixed"

                    success_2 = apply_text_overlay_safe(
                        current_video,
                        text_video_2,
                        TEXT_OVERLAY_TEXT,
                        lane=target_lane,
                        size=TEXT_OVERLAY_SIZE
                    )
                    if success_2 and os.path.exists(text_video_2):
                        current_video = text_video_2
                        
            except Exception as e:
                logger.warning(f"⚠️ Text overlay error: {e}, skipping")
        
        # Color Grading
        if ADD_COLOR_GRADING:
            try:
                logger.info("✨ Step 2.6: Color Grading")
                color_video = os.path.join(job_dir, "color_graded.mp4")
                apply_color_grading(current_video, color_video, COLOR_FILTER, COLOR_INTENSITY)
                if os.path.exists(color_video) and os.path.getsize(color_video) > 100:
                    current_video = color_video
                else:
                    logger.warning(f"⚠️ Color grading output invalid. Keeping original.")
            except Exception as e:
                logger.warning(f"⚠️ Color grading error: {e}, skipping")
        
        # Dynamic Motion (Zoom/Pan) - DISABLED by default to prevent distortion
        if os.getenv("DYNAMIC_MOTION", "no").lower() == "yes":
            try:
                logger.info("✨ Step 2.7: Dynamic Motion (Zoom/Pan)")
                motion_video = os.path.join(job_dir, "motion.mp4")
                # Simple slow zoom in
                # zoompan=z='min(zoom+0.0015,1.5)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'
                vf_motion = "zoompan=z='min(zoom+0.0005,1.1)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':fps=30"
                
                cmd_motion = [
                    FFMPEG_BIN, "-y", "-i", current_video,
                    "-vf", vf_motion,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-c:a", "copy",
                    motion_video
                ]
                if os.path.exists(motion_video):
                    current_video = motion_video
            except Exception as e:
                logger.warning(f"⚠️ Dynamic motion failed: {e}")

        # 2.8 Voiceover Mixing (Moved outside Dynamic Motion)
        if voiceover_path and os.path.exists(voiceover_path):
            try:
                logger.info("✨ Step 2.8: Mixing Voiceover")
                mixed_vo_video = os.path.join(job_dir, "mixed_vo.mp4")
                
                # Simple mix: [1:a]volume=2.0[vo];[0:a]volume=0.5[bg];[bg][vo]amix=inputs=2:duration=first
                cmd_mix = [
                    FFMPEG_BIN, "-y", "-i", current_video, "-i", voiceover_path,
                    "-filter_complex", "amix=inputs=2:duration=first",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    mixed_vo_video
                ]
                if _run_command(cmd_mix, check=True):
                    if os.path.exists(mixed_vo_video) and os.path.getsize(mixed_vo_video) > 1000:
                         current_video = mixed_vo_video
                    else:
                         logger.warning("      ⚠️ Voiceover Mix output invalid/empty.")
                else:
                    logger.warning("      ⚠️ Voiceover Mix command failed.")
            except Exception as e:
                logger.warning(f"⚠️ Voiceover mixing failed: {e}")
        
        norm_video = current_video
        
        # 3. Smart Transitions
        skip_transitions = duration < 30
        
        if skip_transitions:
            logger.info("⚡ Step 3: Transitions (skipped - video too short)")
            merged_video = norm_video
        else:
            logger.info("✨ Step 3: Segmentation")
            seg_pattern = os.path.join(job_dir, "seg_%03d.mp4")
            cmd_split = [
                FFMPEG_BIN, "-y", "-i", norm_video,
                "-c", "copy", "-f", "segment", "-segment_time", str(TRANSITION_INTERVAL),
                "-reset_timestamps", "1", seg_pattern
            ]
            _run_command(cmd_split, check=True)
            segments = sorted(glob.glob(os.path.join(job_dir, "seg_*.mp4")))
            
            if len(segments) < 2:
                logger.info("   Video too short for transitions.")
                merged_video = norm_video
            else:
                # 4. Transitions
                logger.info("✨ Step 4: Transitions")
                final_segments = []
                transitions = ["fade", "slideleft", "slideright", "wipeleft", "wiperight", "circleopen", "circleclose", "zoom"]
                
                seg0 = segments[0]
                dur0 = _get_video_info(seg0)['duration']
                trim0 = os.path.join(job_dir, "final_seg_000.mp4")
                _run_command([FFMPEG_BIN, "-y", "-i", seg0, "-t", str(max(0, dur0 - TRANSITION_DURATION)), "-c", "copy", trim0])
                final_segments.append(trim0)
                
                for i in range(len(segments) - 1):
                    seg_curr = segments[i]
                    seg_next = segments[i+1]
                    trans_type = random.choice(transitions)
                    trans_path = os.path.join(job_dir, f"final_trans_{i}.mp4")
                    create_transition_clip(seg_curr, seg_next, trans_path, trans_type, TRANSITION_DURATION)
                    final_segments.append(trans_path)
                    
                    is_last = (i + 1) == (len(segments) - 1)
                    dur_next = _get_video_info(seg_next)['duration']
                    start_trim = TRANSITION_DURATION
                    end_trim = 0 if is_last else TRANSITION_DURATION
                    keep_dur = max(0, dur_next - start_trim - end_trim)
                    
                    body_next = os.path.join(job_dir, f"final_seg_{i+1:03d}.mp4")
                    _run_command([
                        FFMPEG_BIN, "-y", "-ss", str(start_trim), "-i", seg_next,
                        "-t", str(keep_dur), "-c", "copy", body_next
                    ])
                    final_segments.append(body_next)

                # 5. Merge
                logger.info("✨ Step 5: Merging")
                
                # Filter out missing segments (e.g. if transition generation failed)
                valid_segments = [p for p in final_segments if os.path.exists(p) and os.path.getsize(p) > 0]
                
                if not valid_segments:
                    logger.error("❌ No valid segments to merge.")
                    merged_video = norm_video # Fallback
                else:
                    list_file = os.path.join(job_dir, "merge_list.txt")
                    with open(list_file, "w") as f:
                        for p in valid_segments:
                            f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
                    
                    merged_video = os.path.join(job_dir, "merged_video.mp4")
                    try:
                        _run_command([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", merged_video], check=True)
                    except Exception:
                        logger.error("❌ Merge failed. Using original video.")
                        merged_video = norm_video

        # 6. Smart Audio Remix (Shorts)
        skip_audio_remix = (duration < 15) and not ENABLE_HEAVY_REMIX_SHORTS
        
        if skip_audio_remix:
            logger.info("⚡ Step 6: Audio Remix (skipped - video too short)")
            _run_command([
                FFMPEG_BIN, "-y", "-i", merged_video,
                "-c", "copy", final_output
            ], check=True)
        else:
            if ENABLE_HEAVY_REMIX_SHORTS:
                logger.info("✨ Step 6: Heavy Audio Remix for Shorts (enabled)")
            else:
                logger.info("✨ Step 6: Audio Remix")
            remixed_audio = os.path.join(job_dir, "remixed.wav")
            audio_processing.heavy_remix(merged_video, remixed_audio, original_volume=ORIGINAL_AUDIO_VOLUME)
            
            # --- BACKGROUND MUSIC / AUTO-GEN ---
            if AUTO_MUSIC:
                bg_mixed_video = os.path.join(job_dir, "bg_mixed.mp4")
                vol = MUSIC_VOLUME
                
                # Create temp video with remixed audio first
                temp_remixed = os.path.join(job_dir, "temp_remixed.mp4")
                cmd_merge_audio = [
                    FFMPEG_BIN, "-y", "-i", merged_video, "-i", remixed_audio,
                    "-map", "0:v", "-map", "1:a", "-c", "copy", temp_remixed
                ]
                _run_command(cmd_merge_audio, check=True)
                
                music_success = False
                
                # A. Try Existing Tracks
                if audio_processing.mix_background_music(temp_remixed, bg_mixed_video, volume=vol):
                    merged_video = bg_mixed_video
                    music_success = True
                
                # B. Auto-Generate if no tracks found and enabled
                elif os.getenv("ENABLE_AUTO_MUSIC_GEN", "yes").lower() == "yes":
                    logger.info("🎹 No tracks found. Auto-generating transformative music for Short...")
                    generated_music = os.path.join(job_dir, "auto_gen_music.wav")
                    
                    # Get duration
                    dur = _get_video_info(merged_video)['duration']
                    
                    # Pass as list
                    if audio_processing.generate_transformative_music([merged_video], generated_music, duration=dur):
                        # Mix generated music
                        cmd_mix_gen = [
                            FFMPEG_BIN, "-y", "-i", temp_remixed, "-i", generated_music,
                            "-map", "0:v", "-map", "1:a",
                            "-c:v", "copy", "-c:a", "aac", "-shortest",
                            bg_mixed_video
                        ]
                        if _run_command(cmd_mix_gen, check=True):
                            merged_video = bg_mixed_video
                            music_success = True

            # Final Encode
            encoder = _get_ffmpeg_encoder()
            preset = "p4" if encoder == "h264_nvenc" else REENCODE_PRESET
            
            # Dynamic Metadata for Anti-Reused Content
            unique_id = str(uuid.uuid4())
            creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cmd_final = [
                FFMPEG_BIN, "-y", "-i", merged_video,
                "-c:v", encoder, "-preset", preset,
                "-c:a", "aac", 
                "-map_metadata", "-1",
                "-metadata", "title=Transformative Short",
                "-metadata", f"comment=Unique ID: {unique_id}",
                "-metadata", f"creation_time={creation_time}",
                "-metadata", "copyright=2025",
                "-shortest", final_output
            ]
            
            if encoder == "libx264":
                cmd_final.extend(["-crf", REENCODE_CRF])
            else:
                cmd_final.extend(["-rc", "vbr", "-cq", "19", "-qmin", "19", "-qmax", "19"])
                
            _run_command(cmd_final, check=True)
        
        if verify_video_integrity(final_output):
            logger.info(f"✅ Transformative Pipeline Complete: {final_output}")
            return Path(final_output), wm_context
        else:
            raise Exception("Final integrity check failed")
            
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        return None, None
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

# Legacy function for compilation support
def compile_batch_with_transitions(video_files: List[str], output_filename: str) -> Optional[str]:
    """
    Compiles multiple video files into one with transitions, normalization, and audio remixing.
    """
    import audio_processing
    
    if os.path.isabs(output_filename):
        final_output = output_filename
    else:
        final_output = os.path.abspath(os.path.join(COMPILATION_DIR, output_filename))
        
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    
    job_id = f"batch_{int(time.time())}"
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        logger.info(f"🚀 Starting Batch Compilation for {len(video_files)} videos...")
        
        # 2. Normalize Clips (Inputs are ALREADY PROCESSED)
        # User Request: "take the already proccessed shorts... so the clips are already ai captioned and text overlay"
        # So we SKIP re-captioning and re-overlaying.
        
        normalized_clips = []
        logger.info("✨ Step 2: Normalizing Pre-Processed Clips...")
        
        for i, video in enumerate(video_files):
            try:
                # Just normalize (resize/FPS) to ensure consistency for concatenation
                norm_path = os.path.join(job_dir, f"norm_{i:03d}.mp4")
                
                if normalize_video(video, norm_path):
                    if os.path.getsize(norm_path) > 0:
                        normalized_clips.append(norm_path)
                        logger.info(f"   ✅ Normalized clip {i+1}/{len(video_files)}")
                    else:
                        logger.error(f"   ❌ Normalized clip {i+1} is empty.")
                else:
                    logger.error(f"   ❌ Failed to normalize clip {i+1}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error processing clip {i+1}: {e}")

        if not normalized_clips:
            logger.error("❌ No valid clips to compile.")
            return None

        # 3. Smart Transitions
        # If total duration is short, skip transitions to avoid errors
        duration = 0
        for p in normalized_clips:
            duration += _get_video_info(p)['duration']
            
        skip_transitions = duration < 30
        
        final_segments = []
        
        if skip_transitions:
            logger.info("⚡ Step 3: Transitions (skipped - video too short)")
            final_segments = normalized_clips
        else:
            logger.info("✨ Step 3: Generating Transitions...")
            TRANSITION_DURATION = 0.5 # seconds
            
            try:
                for i in range(len(normalized_clips)):
                    curr = normalized_clips[i]
                    
                    # Get info
                    info = _get_video_info(curr)
                    dur = info['duration']
                    
                    # Calculate trim points
                    start_trim = TRANSITION_DURATION if i > 0 else 0
                    end_trim = TRANSITION_DURATION if i < len(normalized_clips) - 1 else 0
                    
                    keep_dur = dur - start_trim - end_trim
                    
                    if keep_dur <= 0:
                        logger.warning(f"⚠️ Clip {i} too short for transitions. Skipping.")
                        final_segments.append(curr)
                        continue
                        
                    # Extract Body
                    body_next = os.path.join(job_dir, f"body_{i:03d}.mp4")
                    _run_command([
                        FFMPEG_BIN, "-y", "-ss", str(start_trim), "-i", curr,
                        "-t", str(keep_dur), "-c", "copy", body_next
                    ])
                    final_segments.append(body_next)
                    
                    # Generate Transition to Next
                    if i < len(normalized_clips) - 1:
                        next_clip = normalized_clips[i+1]
                        trans_path = os.path.join(job_dir, f"trans_{i:03d}.mp4")
                        
                        # Create Xfade
                        tail_a = os.path.join(job_dir, f"tail_{i:03d}.mp4")
                        _run_command([
                            FFMPEG_BIN, "-y", "-ss", str(dur - TRANSITION_DURATION), "-i", curr,
                            "-t", str(TRANSITION_DURATION), "-c:v", "libx264", "-preset", "ultrafast", tail_a
                        ])
                        
                        head_b = os.path.join(job_dir, f"head_{i:03d}.mp4")
                        _run_command([
                            FFMPEG_BIN, "-y", "-ss", "0", "-i", next_clip,
                            "-t", str(TRANSITION_DURATION), "-c:v", "libx264", "-preset", "ultrafast", head_b
                        ])
                        
                        import random
                        t_type = random.choice(["fade", "wipeleft", "wiperight", "circleopen", "rectcrop"])
                        
                        cmd_trans = [
                            FFMPEG_BIN, "-y", "-i", tail_a, "-i", head_b,
                            "-filter_complex", f"[0:v][1:v]xfade=transition={t_type}:duration={TRANSITION_DURATION}:offset=0[v]",
                            "-map", "[v]", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                            "-c:a", "copy", 
                            trans_path
                        ]
                        _run_command(cmd_trans)
                        
                        if os.path.exists(trans_path):
                            final_segments.append(trans_path)
                            
            except Exception as e:
                logger.error(f"⚠️ Transition generation failed: {e}. Falling back to simple concat.")
                final_segments = normalized_clips

        # 4. Merge Segments
        logger.info("✨ Step 4: Merging Segments")
        valid_segments = [p for p in final_segments if os.path.exists(p) and os.path.getsize(p) > 0]
        if not valid_segments:
            logger.error("❌ No valid segments to merge.")
            return None
            
        merged_video = os.path.join(job_dir, "merged_video.mp4")
        merge_success = False
        
        try:
            list_file = os.path.join(job_dir, "merge_list.txt")
            with open(list_file, "w") as f:
                for p in valid_segments:
                    f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
            
            if _run_command([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", merged_video], check=True):
                merge_success = True
        except Exception as e:
            logger.warning(f"⚠️ Concat demuxer failed: {e}")

        if not merge_success:
            try:
                concat_str = "concat:" + "|".join([os.path.abspath(p).replace(os.sep, '/') for p in valid_segments])
                encoder = _get_ffmpeg_encoder()
                preset = "p4" if encoder == "h264_nvenc" else REENCODE_PRESET
                
                cmd_fallback = [
                    FFMPEG_BIN, "-y", "-i", concat_str,
                    "-c:v", encoder, "-preset", preset
                ]
                if encoder == "libx264":
                    cmd_fallback.extend(["-crf", REENCODE_CRF])
                else:
                    cmd_fallback.extend(["-rc", "vbr", "-cq", "19"])
                
                cmd_fallback.extend(["-c:a", "aac", merged_video])
                
                if _run_command(cmd_fallback, check=True):
                    merge_success = True
            except Exception as e:
                logger.error(f"❌ All merge methods failed: {e}")

        if not merge_success or not os.path.exists(merged_video):
            logger.error("❌ Merge failed completely.")
            return None

        # 5. Audio Remix (User Request: Remove Original + Add Remix/Music)
        logger.info("✨ Step 5: Audio Replacement & Remix")
        final_audio_video = os.path.join(job_dir, "final_audio.mp4")
        
        # A. Generate Remix Audio (Unique)
        remix_audio_path = os.path.join(job_dir, "remix_track.wav")
        has_remix = False
        
        try:
            # Try to generate a unique remix using audio_processing
            # We want a track that matches the duration of merged_video
            duration = _get_video_info(merged_video)['duration']
            
            # 1. Try Auto-Gen Music (Transformative) - Controlled by ENABLE_HEAVY_REMIX_COMPILATION
            if ENABLE_HEAVY_REMIX_COMPILATION and os.getenv("ENABLE_AUTO_MUSIC_GEN", "yes").lower() == "yes":
                 logger.info("   🎹 Generating unique AI remix...")
                 if audio_processing.generate_transformative_music(normalized_clips, remix_audio_path, duration=duration):
                     has_remix = True
            
            # 2. Fallback: Music Folder Mix (Multi-Song Stitching)
            if not has_remix:
                logger.info("   📂 Remix failed, falling back to 'music/' folder mix...")
                music_dir = os.path.join(os.getcwd(), "music")
                
                # Use the new continuous mix function
                if audio_processing.create_continuous_music_mix(remix_audio_path, duration, music_dir):
                     has_remix = True
                     logger.info("   ✅ Created continuous music mix from folder.")
                else:
                    logger.warning("   ⚠️ Failed to create music mix (folder empty?).")

        except Exception as e:
            logger.error(f"   ❌ Audio generation failed: {e}")

        # B. Mux Audio (Replace Original)
        if has_remix and os.path.exists(remix_audio_path):
            logger.info("   🔄 Replacing original audio with remix...")
            # Map video from 0, audio from 1. Shortest ensures we don't overrun.
            cmd_mux = [
                FFMPEG_BIN, "-y", "-i", merged_video, "-i", remix_audio_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                final_audio_video
            ]
            _run_command(cmd_mux, check=True)
            
            # Use this as final
            merged_video = final_audio_video
        else:
            logger.warning("   ⚠️ No remix audio available. Keeping original audio (fallback).")

        # [Safe-Audit] Removed matched duplicate block (Merge + Remix + Final Assembly)
        # The pipeline continues here with the result from the first pass 'merged_video'


        # 6.5 Intro Voiceover (Applied AFTER Audio Remix to ensure it's not overwritten)
        if os.getenv("ENABLE_COMPILATION_INTRO", "yes").lower() == "yes":
            try:
                logger.info("✨ Step 6.5: Generating Intro Voiceover")
                intro_vo_path = os.path.join(job_dir, "intro_vo.mp3")
                
                # Use first clip for context
                first_clip = normalized_clips[0]
                
                # Generate Intro Text
                from gemini_captions import generate_caption_from_video
                intro_text = generate_caption_from_video(first_clip, style="compilation_intro")
                
                if intro_text:
                    logger.info(f"   🎤 Intro Hook: '{intro_text}'")
                    
                    # Generate Audio
                    from voiceover import generate_voiceover
                    if generate_voiceover(intro_text, intro_vo_path):
                        logger.info("   ✅ Intro voiceover generated.")
                        
                        # Mix into start of video (Duck background audio)
                        intro_mixed_video = os.path.join(job_dir, "intro_mixed.mp4")
                        
                        # Filter: [1:a]volume=2.0[intro];[0:a]volume=0.5[bg];[bg][intro]amix=inputs=2:duration=first
                        # We apply this to the ALREADY REMIXED video (merged_video)
                        
                        cmd_intro_mix = [
                            FFMPEG_BIN, "-y", "-i", merged_video, "-i", intro_vo_path,
                            "-filter_complex", "[1:a]volume=2.5[vo];[0:a]volume=0.4[bg];[bg][vo]amix=inputs=2:duration=first",
                            "-c:v", "copy", "-c:a", "aac",
                            intro_mixed_video
                        ]
                        
                        if _run_command(cmd_intro_mix, check=True):
                            merged_video = intro_mixed_video
                            logger.info("   ✅ Intro voiceover mixed into compilation.")
            except Exception as e:
                logger.warning(f"⚠️ Intro voiceover failed: {e}")

        # 6. Final Assembly (Metadata & Container)
        logger.info("✨ Step 6: Final Assembly")
        
        # Dynamic Metadata for Anti-Reused Content
        unique_id = str(uuid.uuid4())
        creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cmd_final = [
            FFMPEG_BIN, "-y", "-i", merged_video,
            "-c:v", "copy", "-c:a", "copy",
            "-map_metadata", "-1",
            "-metadata", "title=Transformative Compilation",
            "-metadata", f"comment=Unique ID: {unique_id}",
            "-metadata", f"creation_time={creation_time}",
            "-metadata", "copyright=2025",
            final_output
        ]
        _run_command(cmd_final, check=True)
        
        if verify_video_integrity(final_output):
            logger.info(f"✅ Batch Compilation Complete: {final_output}")
        else:
            logger.info(f"✅ Transformative Pipeline Complete: {final_output}")
        return final_output

    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        return None
    finally:
        # Cleanup temp dir (except watermark frame if context exists)
        # shutil.rmtree(job_dir, ignore_errors=True) 
        pass
def reprocess_watermark_step(input_video: str, retry_mode: bool = False) -> tuple[str, Optional[Dict]]:
    """
    Reprocesses an existing video for watermark removal/replacement.
    Used for the Retry Loop when user rejects the initial result.
    """
    logger.info(f"🔄 Reprocessing Watermark (Retry: {retry_mode})...")
    
    # Create a new output path to avoid overwriting
    dir_name = os.path.dirname(input_video)
    base_name = os.path.basename(input_video)
    name, ext = os.path.splitext(base_name)
    
    # If already reprocessed, strip the suffix to avoid stacking
    if "_reprocessed" in name:
        name = name.split("_reprocessed")[0]
        
    timestamp = int(time.time())
    output_video = os.path.join(dir_name, f"{name}_reprocessed_{timestamp}{ext}")
    
    # Run watermark processing
    from watermark_auto import process_video_with_watermark
    
    result = process_video_with_watermark(input_video, output_video, retry_mode=retry_mode)
    
    if result["success"] and os.path.exists(output_video):
        logger.info(f"✅ Reprocessing complete: {output_video}")
        return output_video, result.get("context")
    else:
        logger.error("❌ Reprocessing failed.")
        return input_video, None
