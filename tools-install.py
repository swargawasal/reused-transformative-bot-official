import os
import sys
import requests
import logging
import shutil
import subprocess
from tqdm import tqdm

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] tools-install: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tools-install")

MODELS_DIR = os.path.join(os.getcwd(), "models", "heavy")
os.makedirs(MODELS_DIR, exist_ok=True)

# Model URLs (Direct Links)
MODELS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
}

def is_gpu_available():
    """Check if NVIDIA GPU is available via nvidia-smi or torch."""
    # Method 1: Check nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            subprocess.check_output("nvidia-smi", stderr=subprocess.STDOUT)
            return True
        except:
            pass
            
    # Method 2: Check torch (if installed)
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
        
    return False

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logger.info(f"✅ {os.path.basename(dest_path)} already exists.")
        return True
        
    logger.info(f"📥 Downloading {os.path.basename(dest_path)}...")
    temp_path = dest_path + ".part"
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(temp_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        
        # Atomic Move
        shutil.move(temp_path, dest_path)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False

def main():
    logger.info("🚀 Starting Smart Tools Installer...")
    
    # 1. Check FAST_MODE
    fast_mode = os.getenv("FAST_MODE", os.getenv("AI_FAST_MODE", "no")).lower() == "yes"
    if fast_mode:
        logger.info("⚡ FAST_MODE is enabled. Skipping heavy AI model downloads.")
        return

    # 2. Check Hardware
    gpu_available = is_gpu_available()
    if not gpu_available:
        # Check if user explicitly wants to force CPU AI (not recommended but possible)
        force_cpu = os.getenv("FORCE_CPU_AI", "no").lower() == "yes"
        
        if not force_cpu:
            logger.warning("⚠️ No NVIDIA GPU detected. Skipping heavy AI model downloads to save space/time.")
            logger.info("💡 To force download anyway, set FORCE_CPU_AI=yes in .env")
            logger.info("💡 Bot will run in 'Basic Mode' (FFmpeg scaling) which is faster for CPU.")
            return
        else:
            logger.info("⚠️ CPU Mode forced. Downloading models (this might be slow to run)...")
    else:
        logger.info("✅ GPU Detected. Downloading heavy AI models for enhancement...")

    # 3. Download Models
    success = True
    for name, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if not download_file(url, dest):
            success = False
            
    if success:
        logger.info("✨ All required models installed successfully.")
    else:
        logger.error("⚠️ Some models failed to download.")
        # Don't exit(1) here, allow bot to continue in basic mode
        
if __name__ == "__main__":
    main()
