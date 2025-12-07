from dotenv import load_dotenv
load_dotenv()

import os
import glob
import logging
import asyncio
import shutil
import sys
import re
import time
import subprocess
import csv
import json
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import threading
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from telegram.error import NetworkError, TimedOut

import compiler
import uploader
import downloader

# Constants
ALLOWED_DOMAINS = ["instagram.com", "youtube.com", "youtu.be"]

# Logging Setup
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN not found in .env! Exiting.")
    sys.exit(1)

# Global Activity State (Smart Idle Tracking)
class GlobalState:
    is_busy = False
    last_activity = time.time()
    _lock = threading.Lock()
    
    @classmethod
    def set_busy(cls, busy: bool):
        with cls._lock:
            cls.is_busy = busy
            cls.last_activity = time.time()
    
    @classmethod
    def get_idleness(cls):
        with cls._lock:
            if cls.is_busy: return 0
            return time.time() - cls.last_activity

# Global State
user_sessions = {}
COMPILATION_BATCH_SIZE = int(os.getenv("COMPILATION_BATCH_SIZE", "5"))

# ==================== AUTO-INSTALL & SETUP ====================

# ==================== AUTO-INSTALL & SETUP ====================

def detect_hardware_capabilities():
    """
    Detect hardware capabilities for smart auto-selection.
    Returns: dict with 'has_gpu', 'gpu_name', 'vram_gb', 'cuda_available'
    """
    hardware_info = {
        'has_gpu': False,
        'gpu_name': 'CPU',
        'vram_gb': 0,
        'cuda_available': False
    }
    
    try:
        # Try to detect NVIDIA GPU without importing torch
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split(',')
            hardware_info['has_gpu'] = True
            hardware_info['gpu_name'] = gpu_info[0].strip()
            hardware_info['vram_gb'] = int(gpu_info[1].strip().split()[0]) / 1024
            logger.info(f"🎮 GPU Detected: {hardware_info['gpu_name']} ({hardware_info['vram_gb']:.1f} GB VRAM)")
    except:
        logger.info("ℹ️ No NVIDIA GPU detected via nvidia-smi.")
    
    # Check PyTorch CUDA availability if GPU detected or just to be sure
    try:
        import torch
        if torch.cuda.is_available():
            hardware_info['cuda_available'] = True
            hardware_info['has_gpu'] = True # Confirm GPU presence
            if hardware_info['gpu_name'] == 'CPU':
                hardware_info['gpu_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
        
    return hardware_info

def resolve_compute_mode():
    """
    Resolve the final compute mode based on CPU_MODE, GPU_MODE settings and Hardware.
    Returns: 'gpu' or 'cpu'
    """
    cpu_mode = os.getenv("CPU_MODE", "auto").lower()
    gpu_mode = os.getenv("GPU_MODE", "auto").lower()
    
    # 1. Forced Modes
    if cpu_mode == "on":
        logger.info("🖥️ CPU_MODE is ON. Forcing CPU.")
        return "cpu"
    
    if gpu_mode == "on":
        logger.info("🎮 GPU_MODE is ON. Forcing GPU.")
        return "gpu"
        
    # 2. Auto Logic
    hardware = detect_hardware_capabilities()
    
    if gpu_mode == "auto":
        if hardware['cuda_available']:
            logger.info(f"🤖 GPU_MODE=auto: CUDA detected ({hardware['gpu_name']}). Selecting GPU.")
            return "gpu"
        elif hardware['has_gpu']:
             logger.info(f"🤖 GPU_MODE=auto: GPU detected but CUDA not ready. Falling back to CPU.")
             return "cpu"
        else:
            logger.info("🤖 GPU_MODE=auto: No GPU detected. Selecting CPU.")
            return "cpu"
            
    # Default fallback
    return "cpu"

def check_and_update_env():
    """
    Auto-updates .env file with missing keys and smart defaults.
    """
    env_path = ".env"
    if not os.path.exists(env_path):
        logger.warning("⚠️ .env file not found. Creating template...")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("""# ==================== CORE SETTINGS ====================
# REQUIRED: Get your bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE

# REQUIRED: Get your API key from https://aistudio.google.com/app/apikey
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# ==================== PERFORMANCE ====================
# Modes: auto, on, off
CPU_MODE=auto
GPU_MODE=auto
REENCODE_PRESET=fast
REENCODE_CRF=25

# ==================== ENHANCEMENT ====================
ENHANCEMENT_LEVEL=medium
TARGET_RESOLUTION=1080:1920

# ==================== TRANSFORMATIVE FEATURES ====================
ADD_TEXT_OVERLAY=yes
TEXT_OVERLAY_TEXT=🔥 VIRAL
TEXT_OVERLAY_POSITION=bottom
TEXT_OVERLAY_STYLE=modern

ADD_COLOR_GRADING=yes
COLOR_FILTER=cinematic
COLOR_INTENSITY=0.5

ADD_SPEED_RAMPING=yes
SPEED_VARIATION=0.15

FORCE_AUDIO_REMIX=yes

# ==================== COMPILATION ====================
COMPILATION_BATCH_SIZE=6
SEND_TO_YOUTUBE=off
DEFAULT_HASHTAGS_SHORTS=#shorts #viral #trending
DEFAULT_HASHTAGS_COMPILATION=#compilation #funny #viral

# ==================== TRANSITIONS ====================
TRANSITION_DURATION=0.5
TRANSITION_INTERVAL=5
""")
        logger.info("✅ Created .env template. Please update TELEGRAM_BOT_TOKEN and GEMINI_API_KEY!")
        
    # Load current env content
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    updates = []
    
    # Define required keys and defaults
    required_keys = {
        "CPU_MODE": "auto",
        "GPU_MODE": "auto",
        "ENHANCEMENT_LEVEL": "medium",
        "TRANSITION_INTERVAL": "5",
        "TRANSITION_DURATION": "0.5",
        "FORCE_AUDIO_REMIX": "yes",
        "ADD_TEXT_OVERLAY": "yes",
        "ADD_SPEED_RAMPING": "yes"
    }
    
    for key, default in required_keys.items():
        if key not in os.environ and f"{key}=" not in content:
            logger.info(f"➕ Auto-adding missing key: {key}={default}")
            updates.append(f"\n# Auto-added by Smart Installer\n{key}={default}")
            os.environ[key] = default 
            
    if updates:
        with open(env_path, "a", encoding="utf-8") as f:
            f.writelines(updates)
        logger.info(f"✅ Auto-added {len(updates)} missing keys to .env")

# Conditional imports
compute_mode = os.environ.get("COMPUTE_MODE", "cpu")

try:
    import downloader
    import uploader
    from compiler import compile_with_transitions, compile_batch_with_transitions
    from router import run_enhancement
    
    # Only import audio_processing if we have full dependencies (GPU mode)
    if compute_mode == "gpu":
        try:
            import audio_processing
        except ImportError:
            logger.warning("⚠️ audio_processing not found (likely CPU mode).")
            audio_processing = None
    else:
        logger.info("ℹ️ Skipping audio_processing import (CPU mode)")
        audio_processing = None
        
except ImportError as e:
    logger.error(f"Critical Import Error: {e}")
    sys.exit(1)

# ==================== UTILS ====================

UPLOAD_LOG = "upload_log.csv"

def _ensure_log_header():
    if not os.path.exists(UPLOAD_LOG):
        with open(UPLOAD_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "file_path", "yt_link", "title"])

def log_video(file_path: str, yt_link: str, title: str):
    _ensure_log_header()
    with open(UPLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), file_path, yt_link, title])

def total_uploads() -> int:
    if not os.path.exists(UPLOAD_LOG):
        return 0
    with open(UPLOAD_LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        return max(0, len(rows) - 1)

def last_n_filepaths(n: int) -> list:
    """Get the last N video file paths from the upload log, filtered by recency."""
    if not os.path.exists(UPLOAD_LOG):
        return []
    
    with open(UPLOAD_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter by timestamp - only videos from last 24 hours
    from datetime import datetime, timedelta
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    
    recent_rows = []
    for r in rows:
        try:
            timestamp_str = r.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp > cutoff_time:
                    recent_rows.append(r)
        except:
            # If timestamp parsing fails, skip this row
            continue
    
    # Get last N from recent rows
    subset = recent_rows[-n:]
    paths = [r.get("file_path") for r in subset if r.get("file_path")]
    
    # Return only paths that exist
    valid_paths = [p for p in paths if p and os.path.exists(p)]
    
    logger.info(f"📊 Found {len(valid_paths)} recent videos for compilation (last 24h)")
    return valid_paths

async def safe_reply(update: Update, text: str):
    for attempt in range(1, 4):
        try:
            if update.message:
                await update.message.reply_text(
                    text,
                    read_timeout=30,
                    write_timeout=30,
                    connect_timeout=30,
                    pool_timeout=30
                )
            return
        except (NetworkError, TimedOut) as e:
            logger.warning(f"🛑 Reply failed (Attempt {attempt}/3): {e}. Retrying in 5s...")
            await asyncio.sleep(5)
    logger.error("❌ Failed to send message after retries.")

async def safe_video_reply(update: Update, video_path: str, caption: str = None):
    """
    Robustly reply with a video, handling timeouts and retries.
    """
    for attempt in range(1, 4):
        try:
            if update.message:
                # read_timeout/write_timeout kwargs are supported in send_video (which reply_video wraps)
                # We set a very high timeout for large file uploads
                await update.message.reply_video(
                    video_path, 
                    caption=caption, 
                    read_timeout=600, 
                    write_timeout=600,
                    connect_timeout=60,
                    pool_timeout=60
                )
            return
        except (NetworkError, TimedOut) as e:
            logger.warning(f"🛑 Video reply failed (Attempt {attempt}/3): {e}. Retrying in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"❌ Video reply error: {e}")
            break
            
    logger.error("❌ Failed to send video after retries.")
    await safe_reply(update, "❌ Failed to send video due to network timeout.")

def _validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(allowed in domain for allowed in ALLOWED_DOMAINS)
    except: return False

def _sanitize_title(title: str) -> str:
    # Allow spaces but remove other special characters
    clean = re.sub(r'[^\w\s-]', '', title)
    # clean = clean.replace(' ', '_')  <-- REMOVED: Keep spaces for YouTube title
    return clean[:100]  # Increased limit slightly for better titles

def _get_hashtags(text: str) -> str:
    link_count = len(re.findall(r'https?://', text))
    if link_count > 1:
        return os.getenv("DEFAULT_HASHTAGS_COMPILATION", "").strip()
    return os.getenv("DEFAULT_HASHTAGS_SHORTS", "").strip()



# ==================== COMPILATION LOGIC ====================

async def maybe_compile_and_upload(update: Update):
    count = total_uploads()
    n = COMPILATION_BATCH_SIZE
    if n <= 0 or count == 0 or count % n != 0:
        return

    await safe_reply(update, f"⏳ Creating compilation of last {n} shorts...📦")
    files = last_n_filepaths(n)
    if len(files) < n:
        await safe_reply(update, "⚠️ Not enough local files to compile. Skipping.")
        return

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_name = f"compilation_{n}_{stamp}.mp4"
    await safe_reply(update, f"🔨 Merging {len(files)} videos now...🛸")

    try:
        await safe_reply(update, "✨ Running full AI pipeline for batch compilation…")

        # --- Single Stage: Batch Compile with Transitions ---
        # This replaces the old 2-stage process (raw merge -> enhance)
        # Now we normalize -> transition -> merge -> remix -> assemble in one go
        
        output_filename = f"compilation_{n}_{stamp}.mp4"
        
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            files,
            output_filename
        )
        
        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Failed to create compilation.")
            return

        # Check if we should send to YouTube or Telegram
        send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "off").lower() == "on"
        
        if not send_to_youtube:
            await safe_reply(update, "📤 Sending compilation for testing...")
            if os.path.getsize(merged) < 50 * 1024 * 1024:
                await safe_video_reply(update, merged)
            else:
                await safe_reply(update, "⚠️ Compilation too large for Telegram.")
            return

        comp_title = f"🎬 {n} Videos Compilation #{count // n}"  # Changed from "Shorts" to "Videos"
        
        # Use compilation hashtags WITHOUT #Shorts to ensure it's uploaded as regular video
        comp_hashtags = os.getenv("DEFAULT_HASHTAGS_COMPILATION", "").replace("#Shorts", "").replace("#shorts", "").strip()
        
        comp_link = await uploader.upload_to_youtube(merged, comp_hashtags, comp_title)

        if comp_link:
            await safe_reply(update, f"🎉 Compilation uploaded!\n🔗 {comp_link}")
            log_video(merged, comp_link, comp_title)
        else:
            await safe_reply(update, "❌ Failed to upload compilation.")

    except Exception as e:
        logger.exception("Compilation/upload failed: %s", e)
        await safe_reply(update, f"❌ Compilation failed: {e}")

async def compile_last(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Compiles the last N downloaded videos from the downloads/ folder.
    Usage: 
      /compile_last <number> (default 6)
      /compile_last <number> <name_prefix> (e.g. /compile_last 6 reem hot)
    """
    try:
        # 1. Parse arguments
        n = 6
        name_query = None
        
        if context.args:
            try:
                n = int(context.args[0])
            except ValueError:
                await safe_reply(update, "⚠️ Invalid number. Using default: 6")
            
            if len(context.args) > 1:
                name_query = " ".join(context.args[1:])
        
        if n <= 1:
            await safe_reply(update, "⚠️ Please specify at least 2 videos.")
            return

        # Source from Processed Shorts
        source_dir = "Processed Shorts"
        if not os.path.exists(source_dir):
             await safe_reply(update, f"❌ Directory '{source_dir}' not found.")
             return

        selected_files = []
        
        if name_query:
            # --- NAMED SORT COMPILATION ---
            # User wants specific named clips (e.g. reem_hot_1, reem_hot_2...)
            clean_query = _sanitize_title(name_query) # Use same sanitizer as downloader/main
            clean_query = clean_query.replace(' ', '_') # Ensure underscores if sanitizer kept spaces
            
            logger.info(f"🔍 Searching for clips matching: {clean_query}")
            await safe_reply(update, f"🔍 Searching for {n} clips matching '{clean_query}'...")
            
            # Find all files matching the pattern
            # We look for: base_name.mp4, base_name_1.mp4, base_name_2.mp4...
            # Or just any file starting with base_name
            all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            
            # Filter by name prefix
            matching_files = []
            for f in all_files:
                fname = os.path.basename(f)
                if fname.startswith(clean_query):
                    matching_files.append(f)
            
            # Sort them naturally (reem_hot.mp4, reem_hot_1.mp4, reem_hot_2.mp4...)
            # We need smart sorting to handle _1, _2, _10 correctly
            def natural_keys(text):
                return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
                
            matching_files.sort(key=lambda f: natural_keys(os.path.basename(f)))
            
            if len(matching_files) < n:
                await safe_reply(update, f"⚠️ Not enough clips found matching '{clean_query}'. Found {len(matching_files)}, need {n}.")
                return
                
            # Take the first N (assuming user wants the sequence 1..N)
            # Or should we take the last N? 
            # User said: "reem_hot_1 + reem_hot_2 ... reem_hot_6"
            # This implies the first 6 of that sequence.
            # But if they downloaded 12, and ask for 6, maybe they want the latest?
            # "compile_last" usually means latest.
            # However, with named clips, usually you download a batch and want to compile THAT batch.
            # Let's take the LAST N to be consistent with command name.
            selected_files = matching_files[-n:]
            
        else:
            # --- DEFAULT: TIME BASED ---
            all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            files = [f for f in all_files if not os.path.basename(f).startswith("compile_")]
            
            if not files:
                await safe_reply(update, f"❌ No processed videos found in '{source_dir}' folder.")
                return
    
            # Sort by modification time (newest first)
            files.sort(key=os.path.getmtime, reverse=True)
            
            # Take top N
            selected_files = files[:n]
        
        if len(selected_files) < 2:
            await safe_reply(update, f"⚠️ Found {len(selected_files)} videos, but need at least 2 to compile.")
            return

        # Log selected files for user confirmation
        msg = f"✅ Found {len(selected_files)} videos:\n"
        for f in selected_files:
            msg += f"- {os.path.basename(f)}\n"
        await safe_reply(update, msg)

        # 4. Compile
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"compile_last_{n}_{stamp}.mp4"
        if name_query:
            output_filename = f"compile_{clean_query}_{n}_{stamp}.mp4"
        
        await safe_reply(update, "🚀 Starting batch compilation with transitions...")
        GlobalState.set_busy(True)
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            selected_files,
            output_filename
        )
        GlobalState.set_busy(False)

        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Compilation failed (check logs).")
            return

        # 5. Send Result
        await safe_reply(update, "📤 Sending compiled video...")
        caption = f"🎬 Last {len(selected_files)} Videos Compilation"
        if name_query:
            caption = f"🎬 Compilation: {name_query} ({len(selected_files)} clips)"
            
        if os.path.getsize(merged) < 50 * 1024 * 1024:
            await safe_video_reply(update, merged, caption=caption)
        else:
            await safe_reply(update, "⚠️ Video too large for Telegram, but saved locally.")
            
        logger.info(f"✅ /compile_last finished: {merged}")

        # 6. Optional YouTube Upload
        if os.getenv("SEND_TO_YOUTUBE", "off").lower() == "on":
            await safe_reply(update, "📤 Uploading compilation to YouTube...")
            
            # Prepare Metadata
            comp_hashtags = os.getenv("DEFAULT_HASHTAGS_COMPILATION", "#compilation #viral").replace("#Shorts", "").strip()
            
            link = await uploader.upload_to_youtube(merged, title=caption, hashtags=comp_hashtags)
            
            if link:
                await safe_reply(update, f"🎉 Compilation uploaded!\n🔗 {link}")
                log_video(merged, link, caption)
            else:
                await safe_reply(update, "❌ YouTube upload failed.")

    except Exception as e:
        logger.exception(f"/compile_last failed: {e}")
        await safe_reply(update, f"❌ Error: {e}")

async def compile_first(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Compiles the FIRST N downloaded videos from the downloads/ folder.
    Usage: 
      /compile_first <number> (default 6)
      /compile_first <number> <name_prefix> (e.g. /compile_first 6 reem hot)
    """
    try:
        # 1. Parse arguments
        n = 6
        name_query = None
        
        if context.args:
            try:
                n = int(context.args[0])
            except ValueError:
                await safe_reply(update, "⚠️ Invalid number. Using default: 6")
            
            if len(context.args) > 1:
                name_query = " ".join(context.args[1:])
        
        if n <= 1:
            await safe_reply(update, "⚠️ Please specify at least 2 videos.")
            return

        # Source from Processed Shorts
        source_dir = "Processed Shorts"
        if not os.path.exists(source_dir):
             await safe_reply(update, f"❌ Directory '{source_dir}' not found.")
             return

        selected_files = []
        
        if name_query:
            # --- NAMED SORT COMPILATION ---
            clean_query = _sanitize_title(name_query)
            clean_query = clean_query.replace(' ', '_')
            
            logger.info(f"🔍 Searching for clips matching: {clean_query}")
            await safe_reply(update, f"🔍 Searching for {n} clips matching '{clean_query}'...")
            
            all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            
            # Filter by name prefix
            matching_files = []
            for f in all_files:
                fname = os.path.basename(f)
                if fname.startswith(clean_query):
                    matching_files.append(f)
            
            # Sort them naturally (1, 2, 3...)
            def natural_keys(text):
                return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
                
            matching_files.sort(key=lambda f: natural_keys(os.path.basename(f)))
            
            if len(matching_files) < n:
                await safe_reply(update, f"⚠️ Not enough clips found matching '{clean_query}'. Found {len(matching_files)}, need {n}.")
                return
                
            # Take the FIRST N (1..N)
            selected_files = matching_files[:n]
            
        else:
            # --- DEFAULT: TIME BASED ---
            all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            files = [f for f in all_files if not os.path.basename(f).startswith("compile_")]
            
            if not files:
                await safe_reply(update, f"❌ No processed videos found in '{source_dir}' folder.")
                return
    
            # Sort by modification time (OLDEST first)
            files.sort(key=os.path.getmtime, reverse=False)
            
            # Take top N (which are now the oldest)
            selected_files = files[:n]
        
        if len(selected_files) < 2:
            await safe_reply(update, f"⚠️ Found {len(selected_files)} videos, but need at least 2 to compile.")
            return

        # Log selected files for user confirmation
        msg = f"✅ Found {len(selected_files)} videos:\n"
        for f in selected_files:
            msg += f"- {os.path.basename(f)}\n"
        await safe_reply(update, msg)

        # 4. Compile
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"compile_first_{n}_{stamp}.mp4"
        if name_query:
            output_filename = f"compile_{clean_query}_first_{n}_{stamp}.mp4"
        
        await safe_reply(update, "🚀 Starting batch compilation with transitions...")
        GlobalState.set_busy(True)
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            selected_files,
            output_filename
        )
        GlobalState.set_busy(False)

        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Compilation failed (check logs).")
            return

        # 5. Send Result
        await safe_reply(update, "📤 Sending compiled video...")
        caption = f"🎬 First {len(selected_files)} Videos Compilation"
        if name_query:
            caption = f"🎬 Compilation: {name_query} (First {len(selected_files)} clips)"
            
        if os.path.getsize(merged) < 50 * 1024 * 1024:
            await safe_video_reply(update, merged, caption=caption)
        else:
            await safe_reply(update, "⚠️ Video too large for Telegram, but saved locally.")
            
        logger.info(f"✅ /compile_first finished: {merged}")

        # 6. Optional YouTube Upload
        if os.getenv("SEND_TO_YOUTUBE", "off").lower() == "on":
            await safe_reply(update, "📤 Uploading compilation to YouTube...")
            
            # Prepare Metadata
            comp_hashtags = os.getenv("DEFAULT_HASHTAGS_COMPILATION", "#compilation #viral").replace("#Shorts", "").strip()
            
            link = await uploader.upload_to_youtube(merged, title=caption, hashtags=comp_hashtags)
            
            if link:
                await safe_reply(update, f"🎉 Compilation uploaded!\n🔗 {link}")
                log_video(merged, link, caption)
            else:
                await safe_reply(update, "❌ YouTube upload failed.")

    except Exception as e:
        logger.exception(f"/compile_first failed: {e}")
        await safe_reply(update, f"❌ Error: {e}")

# ==================== HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, "❓ Please send an Instagram reel or YouTube link to begin.")

async def getbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, f"Current compilation batch size: {COMPILATION_BATCH_SIZE}")

async def setbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global COMPILATION_BATCH_SIZE
    try:
        if not context.args:
            await safe_reply(update, "Usage: /setbatch <number>")
            return
        n = int(context.args[0])
        if n <= 0:
            await safe_reply(update, "Please provide a positive integer.")
            return
        COMPILATION_BATCH_SIZE = n
        await safe_reply(update, f"✅ Compilation batch size set to {n}.")
    except Exception:
        await safe_reply(update, "Usage: /setbatch <number>")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    load_dotenv(override=True)
    send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "off").lower() == "on"
    
    text = update.message.text.strip()
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    state = session.get('state')

    # Case 1: New URL
    if _validate_url(text):
        # Store URL and wait for title
        user_sessions[user_id] = {
            'state': 'WAITING_FOR_TITLE',
            'pending_url': text
        }
        
        default_hashtags = os.getenv("DEFAULT_HASHTAGS_SHORTS", "#shorts")
        
        await safe_reply(update, f"✅ Got the link!\n\n📌 Hashtags:\n{default_hashtags}\n\n✏️ Now send the title.")
        return

    # Case 2: Waiting for Title
    if state == 'WAITING_FOR_TITLE':
        pending_url = session.get('pending_url')
        if not pending_url:
            await safe_reply(update, "❌ Error: No pending URL found. Please send the link again.")
            return
            
        custom_title = text
        await safe_reply(update, f"✅ Title set: '{custom_title}'\n📥 Downloading...")
        
        try:
            GlobalState.set_busy(True)
            video_path = await asyncio.to_thread(downloader.download_video, pending_url, custom_title=custom_title)
            
            if not video_path:
                GlobalState.set_busy(False)
                await safe_reply(update, "❌ Download failed.")
                user_sessions.pop(user_id, None)
                return
            
            # Load metadata (for hashtags etc)
            metadata = {}
            try:
                meta_path = os.path.splitext(video_path)[0] + ".json"
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                
            # Use user title, but sanitize it for display/files
            title = custom_title
            
            # Combine Metadata Tags + Default Hashtags
            meta_tags = metadata.get('tags', [])
            default_hashtags = os.getenv("DEFAULT_HASHTAGS_SHORTS", "#shorts #viral #trending")
            
            if meta_tags:
                 # Take top 5 meta tags
                 meta_tag_str = " ".join([f"#{t}" for t in meta_tags[:5]])
                 hashtags = f"{default_hashtags} {meta_tag_str}"
            else:
                 hashtags = default_hashtags
            
            await safe_reply(update, f"✅ Downloaded: {title}\n✨ Processing...")
            
            # Compile/Process
            final_path, wm_context = await asyncio.to_thread(compiler.compile_with_transitions, Path(video_path), title)
            GlobalState.set_busy(False)
            
            if not final_path or not os.path.exists(final_path):
                await safe_reply(update, "❌ Processing failed.")
                return
                
            final_str = str(final_path)
            
            # QA: Send for Review
            user_sessions[user_id] = {
                'state': 'WAITING_FOR_APPROVAL',
                'video_path': video_path, # Store original for retry
                'final_path': final_str,
                'title': title,
                'hashtags': hashtags,
                'watermark_context': wm_context # Store for feedback
            }
            
            await safe_reply(update, "✅ Video processed! Sending preview...")
            
            # Dynamic Caption based on detection
            wm_msg = "(No watermark detected - reply 'no' if missed)"
            if wm_context and wm_context.get('coords'):
                 wm_msg = "(Watermark detected - please verify removal- yes/no)"
            
            caption = f"✨ {title}\n\n{hashtags}\n\nReply /approve to upload or /reject to discard.\n\n{wm_msg}"
            
            if os.path.getsize(final_str) < 50 * 1024 * 1024:
                await safe_video_reply(update, final_str, caption=caption)
            else:
                await safe_reply(update, "⚠️ Video too large for Telegram preview.\nReply /approve to upload blindly or /reject.")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            await safe_reply(update, "❌ Error occurred.")
        return

    # Case 3: Approval
    if state == 'WAITING_FOR_APPROVAL':
        if text.lower() in ['approve', '/approve']:
            await approve_upload(update, context)
        elif text.lower() in ['yes', 'y']:
            await verify_watermark(update, context, is_positive=True)
        elif text.lower() in ['no', 'n']:
            await verify_watermark(update, context, is_positive=False)
        elif text.lower() in ['reject', '/reject']:
            await reject_upload(update, context)
        else:
            await safe_reply(update, "⚠️ Options:\n• 'yes'/'no' - Verify watermark removal (Training Data)\n• '/approve' - Upload to YouTube\n• '/reject' - Discard Video")
        return

async def approve_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("DEBUG: Entered approve_upload")
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"DEBUG: Entered approve_upload at {datetime.now()}\n")
        
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    print(f"DEBUG: Session state: {session.get('state')}")
    if session.get('state') != 'WAITING_FOR_APPROVAL':
        await safe_reply(update, "⚠️ No video waiting for approval.")
        return

    final_path = session.get('final_path')
    title = session.get('title')
    hashtags = session.get('hashtags')
    
    print(f"DEBUG: Final path: {final_path}")
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"DEBUG: Final path: {final_path}\n")
        
    if not final_path or not os.path.exists(final_path):
        print(f"DEBUG: File missing: {final_path}")
        with open("debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"DEBUG: File missing: {final_path}\n")
        await safe_reply(update, "❌ Video file expired or missing.")
        user_sessions.pop(user_id, None)
        return

    await safe_reply(update, "📤 Uploading to YouTube...")
    logger.info(f"🚀 Calling uploader for: {final_path}")
    try:
        print("DEBUG: Calling uploader.upload_to_youtube")
        link = await uploader.upload_to_youtube(final_path, title=title, hashtags=hashtags)
        print(f"DEBUG: Upload result: {link}")
        with open("debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"DEBUG: Upload result: {link}\n")
            
        if link:
            await safe_reply(update, f"🎉 Uploaded successfully!\n🔗 {link}")
            log_video(final_path, link, title)
            
            # Check for compilation trigger
            await maybe_compile_and_upload(update)
        else:
            await safe_reply(update, "❌ Upload failed.")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        print(f"DEBUG: Upload exception: {e}")
        with open("debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"DEBUG: Upload exception: {e}\n")
        await safe_reply(update, f"❌ Upload error: {e}")
        
    # Clear session
    print("DEBUG: Clearing session")
    user_sessions.pop(user_id, None)

async def verify_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE, is_positive: bool = True):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    if session.get('state') != 'WAITING_FOR_APPROVAL':
        await safe_reply(update, "⚠️ No video waiting for approval.")
        return

    wm_context = session.get('watermark_context')
    
    # Logic:
    # If wm_context exists and has coords -> System detected something.
    #   User "Yes" -> Correct Detection (Positive)
    #   User "No" -> Wrong Detection (Negative)
    # If wm_context exists but NO coords -> System detected NOTHING.
    #   User "Yes" -> Correct Non-Detection (Positive)
    #   User "No" -> Missed Watermark (Negative) -> RETRY
    
    # Fallback if wm_context is missing (e.g. crash or error)
    # If User says "No", assume they mean "Missed Watermark" and retry.
    
    feedback_type = "POSITIVE" if is_positive else "NEGATIVE"
    
    if wm_context and wm_context.get('coords'):
        # Case 1: System detected something
        try:
            import hybrid_watermark
            hybrid_watermark.confirm_learning(wm_context, is_positive)
            
            if is_positive:
                await safe_reply(update, f"✅ Success! Watermark removal verified. ({feedback_type})")
                await safe_reply(update, "Reply /approve to upload or /reject to discard.")
            else:
                await safe_reply(update, f"❌ Failure recorded. I will learn to ignore this. ({feedback_type})")
                await safe_reply(update, "🔄 Retrying with AGGRESSIVE detection mode...")
                
                # RETRY LOGIC (Copied from Case 2)
                video_path = session.get('video_path')
                title = session.get('title')
                
                if video_path and os.path.exists(video_path):
                    try:
                        # Re-run compiler with aggressive flag
                        final_path, new_wm_context = await asyncio.to_thread(
                            compiler.compile_with_transitions, 
                            Path(video_path), 
                            title, 
                            aggressive_watermark=True
                        )
                        
                        # Update session
                        session['final_path'] = str(final_path)
                        session['watermark_context'] = new_wm_context
                        
                        await safe_reply(update, "✅ Retry complete! Sending new version...")
                        
                        caption = f"✨ {title} (Aggressive Retry)\n\nReply /approve to upload or /reject to discard.\n(Watermark detected? - yes/no)"
                        
                        if os.path.getsize(str(final_path)) < 50 * 1024 * 1024:
                            await safe_video_reply(update, str(final_path), caption=caption)
                        else:
                            await safe_reply(update, "⚠️ Video too large for Telegram.")
                            
                    except Exception as e:
                        logger.error(f"Retry failed: {e}")
                        await safe_reply(update, f"❌ Retry failed: {e}")
                else:
                    await safe_reply(update, "⚠️ Original video not found. Cannot retry.")

            session['watermark_context'] = None
        except Exception as e:
            logger.warning(f"Failed to confirm learning: {e}")
            await safe_reply(update, f"ℹ️ Feedback noted. ({feedback_type})")
            
    else:
        # Case 2: System detected NOTHING (or context missing)
        if is_positive:
            # User says "Yes" -> Correctly detected nothing
            await safe_reply(update, f"✅ Success! Correctly identified NO watermark. ({feedback_type})")
            session['watermark_context'] = None
            await safe_reply(update, "Reply /approve to upload or /reject to discard.")
        else:
            # User says "No" -> Missed watermark
            if wm_context:
                try:
                    import hybrid_watermark
                    hybrid_watermark.save_missed_detection(wm_context)
                except Exception as e:
                    logger.error(f"Failed to save missed detection: {e}")
                
            await safe_reply(update, f"❌ Failure recorded. I missed the watermark. ({feedback_type})")
            await safe_reply(update, "🔄 Retrying with AGGRESSIVE detection mode...")
            
            # RETRY LOGIC
            video_path = session.get('video_path')
            title = session.get('title')
            
            if video_path and os.path.exists(video_path):
                try:
                    # Re-run compiler with aggressive flag
                    final_path, new_wm_context = await asyncio.to_thread(
                        compiler.compile_with_transitions, 
                        Path(video_path), 
                        title, 
                        aggressive_watermark=True
                    )
                    
                    # Update session
                    if final_path and os.path.exists(str(final_path)):
                        session['final_path'] = str(final_path)
                        session['watermark_context'] = new_wm_context
                        
                        await safe_reply(update, "✅ Retry complete! Sending new version...")
                        
                        caption = f"✨ {title} (Aggressive Retry)\n\nReply /approve to upload or /reject to discard.\n(Watermark detected? - yes/no)"
                        
                        if os.path.getsize(str(final_path)) < 50 * 1024 * 1024:
                            await safe_video_reply(update, str(final_path), caption=caption)
                        else:
                            await safe_reply(update, "⚠️ Video too large for Telegram.")
                    else:
                        await safe_reply(update, "❌ Retry failed: Compilation pipeline returned no output.")
                        
                except Exception as e:
                    logger.error(f"Retry failed: {e}")
                    await safe_reply(update, f"❌ Retry failed: {e}")
            else:
                await safe_reply(update, "⚠️ Original video not found. Cannot retry.")

async def reject_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("DEBUG: Entered reject_upload")
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"DEBUG: Entered reject_upload at {datetime.now()}\n")
        
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"DEBUG: Session state: {session.get('state')}\n")
    
    if session.get('state') == 'WAITING_FOR_APPROVAL':
        final_path = session.get('final_path')
        print(f"DEBUG: Rejecting file: {final_path}")
        with open("debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"DEBUG: Rejecting file: {final_path}\n")
            f.write(f"DEBUG: CWD: {os.getcwd()}\n")
            f.write(f"DEBUG: File exists?: {os.path.exists(final_path) if final_path else 'None'}\n")
        
        logger.info(f"🗑️ Attempting to delete rejected file: {final_path}")
        
        # Delete the file to prevent it from being used in compilations
        if final_path and os.path.exists(final_path):
            deleted = False
            for i in range(3):
                try:
                    os.remove(final_path)
                    logger.info(f"🗑️ Deleted rejected file: {final_path}")
                    print(f"DEBUG: File deleted successfully")
                    with open("debug_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"DEBUG: File deleted successfully\n")
                    await safe_reply(update, "🗑️ Video rejected and permanently deleted.")
                    deleted = True
                    break
                except PermissionError:
                    print(f"DEBUG: PermissionError deleting file (Attempt {i+1}/3)")
                    with open("debug_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"DEBUG: PermissionError deleting file (Attempt {i+1}/3)\n")
                    logger.warning(f"⚠️ PermissionError deleting file (Attempt {i+1}/3). Retrying...")
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"DEBUG: Error deleting file: {e}")
                    with open("debug_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"DEBUG: Error deleting file: {e}\n")
                    logger.error(f"❌ Failed to delete file: {e}")
                    break
            
            if not deleted:
                await safe_reply(update, "⚠️ Video rejected but failed to delete file (File locked?).")
        else:
            print("DEBUG: File already missing")
            with open("debug_log.txt", "a", encoding="utf-8") as f:
                f.write(f"DEBUG: File already missing\n")
            await safe_reply(update, "🗑️ Video discarded (File already missing).")
            
        print("DEBUG: Clearing session after reject")
        user_sessions.pop(user_id, None)
    else:
        print("DEBUG: Nothing to reject")
        with open("debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"DEBUG: Nothing to reject. Session state: {session.get('state')}\n")
        await safe_reply(update, "⚠️ Nothing to reject.")

import signal
import sys

def signal_handler(sig, frame):
    logger.info("🛑 KeyboardInterrupt received. Force Shutting down...")
    os._exit(0)

def main():
    # Register Signal Handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).connect_timeout(30).read_timeout(30).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("getbatch", getbatch))
    app.add_handler(CommandHandler("setbatch", setbatch))
    app.add_handler(CommandHandler("compile_last", compile_last))
    app.add_handler(CommandHandler("compile_first", compile_first))
    app.add_handler(CommandHandler("approve", approve_upload))
    app.add_handler(CommandHandler("reject", reject_upload))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    logger.info("🤖 Bot is running...")
    
    # Run polling
    # stop_signals=None prevents it from overwriting our signal handler (unlikely, but safe)
    app.run_polling()

# ==================== AUTO-TRAINING ====================
# ==================== AUTO-TRAINING ====================
class AutoTrainer(threading.Thread):
    def __init__(self, interval_minutes=60): # interval_minutes argument kept for compatibility but ignored
        super().__init__()
        self.daemon = True
        self.running = True
        self.last_train_time = time.time()

    def run(self):
        logger.info("⏳ Smart AutoTrainer started (Check: 30s, Trigger: >3m Idle).")
        
        # 1. Startup Training: ALWAYS run on boot
        logger.info("🚀 Triggering startup model training...")
        self._trigger_retrain()
        self.last_train_time = time.time()
            
        while self.running:
            time.sleep(30) # Check status every 30 seconds
            
            # Smart Idle Check
            idle_seconds = GlobalState.get_idleness()
            time_since_last = time.time() - self.last_train_time
            
            # Rule: Train if Idle > 3 mins AND (It's been > 3 mins since last train)
            if idle_seconds > 180 and time_since_last > 180:
                logger.info(f"💤 System Idle for {int(idle_seconds)}s. Triggering Maintenance Training...")
                self._trigger_retrain()
                self.last_train_time = time.time()

    def _trigger_retrain(self):
        try:
            from scripts import nightly_retrain
            
            # Only run if enabled
            if os.getenv("NIGHTLY_RETRAIN", "yes").lower() == "yes":
                nightly_retrain.retrain()
        except Exception as e:
            logger.error(f"❌ AutoTrainer Error: {e}")

class AutoCleanup(threading.Thread):
    def __init__(self, interval_minutes=60, age_days=2):
        super().__init__()
        self.interval = interval_minutes * 60
        self.age_days = age_days
        self.daemon = True
        self.running = True
        self.target_dir = "downloads"
        self.state_file = "cleanup_state.json"
        self.last_run = self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('last_run', 0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cleanup state: {e}")
        return 0

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump({'last_run': self.last_run}, f)
        except Exception as e:
            logger.error(f"❌ Failed to save cleanup state: {e}")

    def run(self):
        logger.info("🧹 AutoCleanup started (Persistent Mode).")
        
        while self.running:
            # Calculate time since last run
            elapsed = time.time() - self.last_run
            wait_time = max(0, self.interval - elapsed)
            
            if wait_time > 0:
                logger.info(f"⏳ Next cleanup in {int(wait_time/60)} minutes ({int(wait_time)}s)...")
                # Sleep in chunks to allow faster shutdown if needed (though daemon thread handles kill)
                # But for simplicity, simple sleep is fine as it's a daemon thread.
                time.sleep(wait_time)
            
            # Perform cleanup
            self._cleanup()
            
            # Update state
            self.last_run = time.time()
            self._save_state()
            
            # Wait for next interval (full interval now)
            # Actually, the loop logic above handles this naturally:
            # Next iteration: elapsed will be ~0, so wait_time will be ~interval.
            # So we don't need an extra sleep here.

    def _cleanup(self):
        try:
            if not os.path.exists(self.target_dir):
                return

            cutoff = time.time() - (self.age_days * 86400)
            count = 0
            
            for item in os.listdir(self.target_dir):
                item_path = os.path.join(self.target_dir, item)
                
                if "Processed Shorts" in item:
                    continue
                    
                if os.path.isfile(item_path):
                    try:
                        if os.path.getmtime(item_path) < cutoff:
                            os.remove(item_path)
                            logger.info(f"🗑️ Auto-Cleanup: Deleted old file {item}")
                            count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Auto-Cleanup failed to delete {item}: {e}")
            
            if count > 0:
                logger.info(f"✅ Auto-Cleanup finished. Deleted {count} old files.")
            else:
                logger.info("✅ Auto-Cleanup finished. No old files to delete.")
                
        except Exception as e:
            logger.error(f"❌ AutoCleanup Error: {e}")

if __name__ == '__main__':
    # Start AutoTrainer (Checks every 60 minutes + runs on startup)
    trainer = AutoTrainer(interval_minutes=60)
    trainer.start()
    
    # Start AutoCleanup (Checks every 60 minutes, deletes files > 2 days old)
    cleanup = AutoCleanup(interval_minutes=60, age_days=2)
    cleanup.start()
    
    main()

