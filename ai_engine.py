# ai_engine.py - OPTIMIZED AI ENHANCEMENT ENGINE
import os
import logging
import cv2
import torch
import numpy as np
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

logger = logging.getLogger("ai_engine")

class HeavyEditor:
    """AI Video Enhancement Engine using RealESRGAN + GFPGAN (Optimized)"""
    def __init__(self, model_dir="models/heavy", scale=2, face_enhance=True):
        self.model_dir = model_dir
        self.scale = scale
        self.face_enhance = face_enhance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths to models
        # Paths to models
        self.realesrgan_x4_path = os.path.join(model_dir, 'RealESRGAN_x4plus.pth')
        self.realesrgan_x2_path = os.path.join(model_dir, 'RealESRGAN_x2plus.pth')
        self.gfpgan_model_path = os.path.join(model_dir, 'GFPGANv1.4.pth')
        
        self.face_enhancer = None
        
        self._load_models()

    def _ensure_model(self, path, url):
        """Check if model exists, if not download it."""
        # STRICT RULE: Only download if running on GPU (CUDA).
        # If on CPU, we should not be using Heavy Engine anyway, so skip download.
        if self.device.type != 'cuda':
            logger.warning(f"⚠️ CPU detected. Skipping auto-download of {path} (GPU only feature).")
            return

        if not os.path.exists(path):
            logger.info(f"⬇️ Model not found: {path}. Downloading from {url}...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                torch.hub.download_url_to_file(url, path)
                logger.info(f"✅ Downloaded: {path}")
            except Exception as e:
                logger.error(f"❌ Failed to download model: {e}")
                if os.path.exists(path): # Cleanup partial download
                    os.remove(path)
                raise e

    def _get_device_config(self):
        """Detect VRAM and return optimal settings based on strict thresholds."""
        config = {
            "tile": 0,
            "half": False,
            "face_enhance": True,
            "device": self.device
        }
        
        if self.device.type == 'cuda':
            config["half"] = True
            try:
                vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                config["vram_gb"] = vram
                logger.info(f"🎮 GPU Detected: {torch.cuda.get_device_name(self.device)} ({vram:.2f} GB VRAM)")
                
                # Strict VRAM Thresholds (Orchestrator Rules)
                # LEVEL 1: < 6GB -> NO HEAVY AI (Fallback to CPU)
                if vram < 6:
                    logger.warning(f"⚠️ GPU Level 1 Detected ({vram:.2f} GB). Insufficient VRAM for Heavy AI.")
                    logger.warning("🚫 Orchestrator Rule: Blocking AI installation on < 6GB GPU.")
                    raise RuntimeError("Insufficient VRAM for Heavy AI (Level 1 GPU). Fallback to CPU required.")

                # LEVEL 2: 6GB - 8GB -> MID GPU (Stable Mode)
                elif vram < 8:
                    logger.info("ℹ️ GPU Level 2 Detected (Mid Range). Enabling Stable Mode.")
                    config["tile"] = 400 # Safety tiling
                    config["face_enhance"] = True
                    config["auto_scale"] = 2
                    
                # LEVEL 3: > 8GB -> HEAVY GPU (Full Power)
                else:
                    logger.info("🚀 GPU Level 3 Detected (High Performance). Full Enhancement Enabled.")
                    config["tile"] = 0 # No tiling needed for T4 (16GB)
                    config["face_enhance"] = True
                    config["auto_scale"] = 4
                    
            except RuntimeError as e:
                raise e # Re-raise our strict error
            except Exception as e:
                logger.warning(f"⚠️ Failed to detect VRAM: {e}. Defaulting to safe mode (Level 2).")
                config["tile"] = 400
                config["face_enhance"] = True
                config["auto_scale"] = 2
        else:
            logger.warning("⚠️ CPU Mode detected. Basic mode only (no face enhancement).")
            config["tile"] = 200
            config["half"] = False
            config["face_enhance"] = False
            config["auto_scale"] = 2
            
        return config

    def _load_models(self):
        logger.info(f"🚀 Loading Heavy Engine Models on {self.device}...")
        
        config = self._get_device_config()
        
        # AUTO SCALE LOGIC
        # If ENHANCEMENT_LEVEL is 'auto', use the scale determined by VRAM
        enhancement_env = os.getenv("ENHANCEMENT_LEVEL", "2x").lower()
        if enhancement_env == "auto":
            if "auto_scale" in config:
                self.scale = config["auto_scale"]
                vram_info = f"{config.get('vram_gb', 0):.1f}GB" if "vram_gb" in config else "CPU/Unknown"
                logger.info(f"🧠 Auto-scale resolved: VRAM={vram_info} → scale={self.scale}")
        elif enhancement_env != "auto":
            logger.info(f"🔒 Manual scale enforced: scale={self.scale}")
            
        # Final scale sanity
        if self.scale not in (2, 4):
            logger.warning(f"⚠️ Invalid scale {self.scale}, falling back to x2")
            self.scale = 2
        
        # Override face enhance if config says no
        if not config["face_enhance"]:
            if self.face_enhance: # Only log if user wanted it but we disabled it
                logger.info("🔧 Face enhancement disabled due to VRAM constraints.")
            self.face_enhance = False
        
        # Determine which model to load based on requested scale
        # If scale is 1 or 2, use x2plus (faster, cleaner). If 3 or 4, use x4plus.
        if self.scale <= 2:
            model_path = self.realesrgan_x2_path
            model_scale = 2
            # Auto-download x2plus
            self._ensure_model(model_path, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")
        else:
            model_path = self.realesrgan_x4_path
            model_scale = 4
            # Auto-download x4plus
            self._ensure_model(model_path, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RealESRGAN model not found at {model_path}")
            
        logger.info(f"⚡ Using RealESRGAN Model: {os.path.basename(model_path)} (Internal Scale: {model_scale}x)")
            
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_scale)
        self.upsampler = RealESRGANer(
            scale=model_scale,
            model_path=model_path,
            model=model,
            tile=config["tile"],
            tile_pad=10,
            pre_pad=0,
            half=config["half"],
            device=self.device
        )
        
        if self.face_enhance:
            # Auto-download GFPGAN
            try:
                self._ensure_model(self.gfpgan_model_path, "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth")
            except Exception as e:
                logger.warning(f"⚠️ Failed to download GFPGAN: {e}. Face enhancement disabled.")
                self.face_enhance = False

            if not os.path.exists(self.gfpgan_model_path):
                logger.warning(f"⚠️ GFPGAN model not found at {self.gfpgan_model_path}. Face enhancement disabled.")
                self.face_enhance = False
            else:
                self.face_enhancer = GFPGANer(
                    model_path=self.gfpgan_model_path,
                    upscale=self.scale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=self.upsampler
                )
                logger.info("✅ Face Enhancement (GFPGAN) Loaded.")
        
        logger.info("✅ Models Loaded Successfully.")

    def enhance_frame(self, img):
        """Enhance a single frame using RealESRGAN + GFPGAN."""
        try:
            with torch.no_grad():
                if self.face_enhance and self.face_enhancer:
                    _, _, output = self.face_enhancer.enhance(
                        img, 
                        has_aligned=False, 
                        only_center_face=False, 
                        paste_back=True
                    )
                else:
                    output, _ = self.upsampler.enhance(img, outscale=self.scale)
                
                # Skin Protection
                enable_skin_protect = os.getenv("ENABLE_SKIN_PROTECT", "yes").lower() == "yes"
                skin_max_brightness = int(os.getenv("SKIN_MAX_BRIGHTNESS", 175))
                
                if enable_skin_protect:
                    # Detect skin (Simple YCrCb)
                    img_ycrcb = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
                    # Skin range: Cr [133, 173], Cb [77, 127]
                    lower = np.array([0, 133, 77], dtype=np.uint8)
                    upper = np.array([255, 173, 127], dtype=np.uint8)
                    skin_mask = cv2.inRange(img_ycrcb, lower, upper)
                    
                    # Dilate to cover edges
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
                    
                    # Convert to LAB for brightness clamping
                    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Create a mask where skin is too bright
                    # Note: LAB L channel is 0-255 in OpenCV
                    bright_skin_mask = cv2.bitwise_and(cv2.threshold(l, skin_max_brightness, 255, cv2.THRESH_BINARY)[1], skin_mask)
                    
                    # Clamp L channel in those areas
                    l = np.where(bright_skin_mask > 0, skin_max_brightness, l)
                    
                    # Merge back
                    lab_clamped = cv2.merge([l, a, b])
                    output = cv2.cvtColor(lab_clamped, cv2.COLOR_LAB2BGR)
            
            return output
        except Exception as e:
            logger.error(f"Frame enhancement failed: {e}")
            return img

    def process_video(self, input_path, output_path, progress_callback=None):
        logger.info(f"🎬 Starting Video Enhancement: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("❌ Could not open input video.")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        target_width = width * self.scale
        target_height = height * self.scale
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        frame_buffer = []
        # Batch size depends on VRAM, but 4 is usually safe for T4
        batch_size = 4 if self.device.type == 'cuda' else 1
        
        try:
            logger.info(f"⚡ Processing with batch size: {batch_size}")
            processed_count = 0
            
            for i in tqdm(range(total_frames), desc="Enhancing"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add to batch
                frame_buffer.append(frame)
                
                # Process batch when full
                if len(frame_buffer) >= batch_size:
                    enhanced_batch = self._process_batch(frame_buffer)
                    for enhanced in enhanced_batch:
                        writer.write(enhanced)
                        processed_count += 1
                    
                    frame_buffer = []
                
                if progress_callback and i % 10 == 0:
                    progress_callback(i / total_frames)
            
            # Process remaining frames
            if frame_buffer:
                enhanced_batch = self._process_batch(frame_buffer)
                for enhanced in enhanced_batch:
                    writer.write(enhanced)
                    processed_count += 1
            
            logger.info(f"✅ Video Enhancement Complete. Processed {processed_count}/{total_frames} frames")
            return True
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False
        finally:
            cap.release()
            writer.release()
    
    def _process_batch(self, frames):
        """Process multiple frames at once for GPU efficiency."""
        enhanced = []
        for frame in frames:
            enhanced.append(self.enhance_frame(frame))
        return enhanced
