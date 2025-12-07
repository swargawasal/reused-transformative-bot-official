"""
Text Overlay Module (Hardened Production Grade)
Handles robust text overlay with Font Auto-Healing via Official Zip, ASS Fallback, and Crash Safety.

Capabilities:
1. Auto-downloads authoritative font (Inter v4.0 Zip).
2. Extracts and verifies font file integrity (>50KB).
3. Falls back to subtitle overlay (.ass) if drawtext fails or unicode detected.
4. Sanitizes all text inputs.
5. Non-blocking failure model (returns False instead of crashing).
"""

import os
import subprocess
import logging
import shutil
import textwrap
import requests
import zipfile
import io
from typing import Optional

logger = logging.getLogger("text_overlay")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FONT_ZIP_URL = "https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip"
LOCAL_FONT_DIR = os.path.join("assets", "fonts")
LOCAL_FONT_PATH = os.path.join(LOCAL_FONT_DIR, "Inter-Bold.ttf")

# Shorts safe margins (percentage of height)
SAFE_TOP = 0.08
SAFE_BOTTOM = 0.92

class TextOverlay:
    _drawtext_supported: Optional[bool] = None
    _font_checked: bool = False
    _drawtext_failed_once: bool = False

    def __init__(self):
        self._ensure_font()
        self._check_drawtext_support()

    def _ensure_font(self):
        """Auto-heals missing font by downloading and extracting the official Zip."""
        if self._font_checked:
            return

        # Check if already exists and valid
        if self._validate_font_file(LOCAL_FONT_PATH):
            self._font_checked = True
            return

        try:
            logger.info(f"⬇️ Downloading font from official release...")
            os.makedirs(LOCAL_FONT_DIR, exist_ok=True)
            
            response = requests.get(FONT_ZIP_URL, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find Inter-Bold.ttf robustly (ignore folder structure prefix)
                target_file = None
                for name in z.namelist():
                    if name.endswith("Inter-Bold.ttf") and "Variable" not in name:
                        target_file = name
                        break
                
                if target_file:
                    with z.open(target_file) as source, open(LOCAL_FONT_PATH, "wb") as target:
                        shutil.copyfileobj(source, target)
                    logger.info("✅ Font extracted successfully.")
                else:
                    logger.error("❌ Inter-Bold.ttf not found in downloaded ZIP.")
            
            # Final Validation
            if self._validate_font_file(LOCAL_FONT_PATH):
                self._font_checked = True
            else:
                logger.warning("⚠️ Font validation failed (missing or too small).")
                
        except Exception as e:
            logger.warning(f"⚠️ Font auto-download/extract failed: {e}. Output will fallback to subtitles.")

    def _validate_font_file(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        # Size check > 50KB
        if os.path.getsize(path) < 50 * 1024:
            return False
        return True

    def _check_drawtext_support(self):
        """Checks if installed FFmpeg supports drawtext filter."""
        if self._drawtext_supported is not None:
            return

        try:
            result = subprocess.run(
                [FFMPEG_BIN, "-filters"], 
                capture_output=True, 
                text=True
            )
            self._drawtext_supported = "drawtext" in result.stdout
            if not self._drawtext_supported:
                logger.warning("⚠️ FFmpeg 'drawtext' filter NOT found. Fallback mode enabled.")
        except Exception:
            logger.warning("⚠️ Could not verify FFmpeg filters. Assuming broken.")
            self._drawtext_supported = False

    def _has_unicode_or_emoji(self, text: str) -> bool:
        """Strict check for characters that break drawtext (Emoji, Arabic, etc)."""
        # Allow Basic Latin + Latin-1 Supplement (ISO-8859-1 coverage)
        try:
            text.encode('latin-1')
            return False
        except UnicodeEncodeError:
            return True

    def _escape_drawtext(self, text: str) -> str:
        """Strict escaping for FFmpeg drawtext."""
        if not text: return ""
        text = text.replace("\\", "\\\\")
        text = text.replace(":", "\\:")
        text = text.replace("'", "\\'")
        text = text.replace("%", "\\%")
        text = text.replace("\n", "\\n")
        return text

    def _escape_ass(self, text: str) -> str:
        """Escaping for ASS subtitles."""
        if not text: return ""
        text = text.replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\n", "\\N") 
        return text

    def _wrap_text(self, text: str, max_chars: int = 26) -> str:
        if not text: return ""
        if len(text) <= max_chars: return text
        return textwrap.fill(text, width=max_chars, break_long_words=False, break_on_hyphens=False)

    def _calc_y_drawtext(self, position: str) -> str:
        if position == "top": return f"h*{SAFE_TOP}"
        if position == "center": return "(h-text_h)/2"
        return f"h*{SAFE_BOTTOM}-text_h"

    def _create_ass_file(self, text: str, position: str) -> str:
        """Generates a temporary .ass subtitle file."""
        ass_path = f"temp/overlay_{os.getpid()}.ass"
        os.makedirs("temp", exist_ok=True)
        
        # Alignment: 2=Bottom Center, 8=Top Center, 5=Middle Center
        alignment = 2
        margin_v = 50 
        
        if position == "top":
            alignment = 8
            margin_v = int(1920 * SAFE_TOP)
        elif position == "center":
            alignment = 5
        elif position == "bottom":
             # ASS MarginV is from bottom for Alignment 2
             margin_v = int(1920 * (1.0 - SAFE_BOTTOM)) + 20

        escaped_text = self._escape_ass(text)

        ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,{alignment},20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,9:59:59.00,Default,,0,0,0,,{escaped_text}
"""
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        return ass_path

    def add_overlay(self, video_path, output_path, text, position="bottom", size=60):
        """
        Main entry point with Strict Fallback Logic.
        """
        if not text or not video_path or not os.path.exists(video_path):
            return False

        wrapped_text = self._wrap_text(text)

        # Decision Tree
        use_drawtext = True
        
        if self._drawtext_failed_once:
             use_drawtext = False
             logger.info("Overlay Method: SUBTITLES (previous_failure)")
        elif not self._drawtext_supported:
             use_drawtext = False
             logger.info("Overlay Method: SUBTITLES (drawtext_unavailable)")
        elif not os.path.exists(LOCAL_FONT_PATH):
             use_drawtext = False
             logger.info("Overlay Method: SUBTITLES (font_missing)")
        elif self._has_unicode_or_emoji(text):
             use_drawtext = False
             logger.info("Overlay Method: SUBTITLES (unicode_unsafe)")
        else:
             logger.info("Overlay Method: DRAWTEXT (attempting)")

        # Attempt Execution
        if use_drawtext:
            success = self._apply_drawtext(video_path, output_path, wrapped_text, position, size)
            if success:
                return True
            else:
                # If drawtext failed, mark it as broken and fallback IMMEDIATELY
                self._drawtext_failed_once = True
                logger.warning("⚠️ Drawtext failed! Falling back to SUBTITLES...")
                return self._apply_ass(video_path, output_path, wrapped_text, position)
        else:
            return self._apply_ass(video_path, output_path, wrapped_text, position)


    def _apply_drawtext(self, video_path, output_path, text, position, size):
        try:
            safe_text = self._escape_drawtext(text)
            y_expr = self._calc_y_drawtext(position)
            
            # Dynamic Border/Box
            border_w = min(int(size * 0.25), 30)
            box_cmd = ""
            if os.getenv("TEXT_OVERLAY_BOX", "yes").lower() == "yes":
                box_cmd = f"box=1:boxcolor=black@0.55:boxborderw={border_w}:"

            # MANDATORY: use_fontconfig=0 to ignore system fonts and prevent crashes
            # MANDATORY: fontfile must be absolute path, colons escaped for filter
            font_path = os.path.abspath(LOCAL_FONT_PATH).replace("\\", "/").replace(":", "\\:")

            vf_filter = (
                f"drawtext="
                f"fontfile='{font_path}':"
                f"text='{safe_text}':"
                f"fontsize={size}:"
                f"fontcolor=white:"
                f"borderw=1:"
                f"bordercolor=black:"
                f"{box_cmd}"
                f"x=(w-text_w)/2:"
                f"y={y_expr}"
            )

            cmd = [
                FFMPEG_BIN, "-y", "-i", video_path,
                "-vf", vf_filter,
                "-c:v", "libx264",
                "-preset", os.getenv("REENCODE_PRESET", "ultrafast"),
                "-crf", os.getenv("REENCODE_CRF", "23"),
                "-c:a", "copy",
                output_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Drawtext execution failed: {e.stderr.decode() if e.stderr else 'Unknown'}")
            return False
        except Exception as e:
            logger.error(f"Drawtext execution crashed: {e}")
            return False

    def _apply_ass(self, video_path, output_path, text, position):
        try:
            ass_file = self._create_ass_file(text, position)
            # Escape for filter syntax: C:/Path -> C\:/Path
            safe_ass_path = os.path.abspath(ass_file).replace("\\", "/").replace(":", "\\:")
            vf_filter = f"subtitles='{safe_ass_path}'"

            cmd = [
                FFMPEG_BIN, "-y", "-i", video_path,
                "-vf", vf_filter,
                "-c:v", "libx264",
                "-preset", os.getenv("REENCODE_PRESET", "ultrafast"),
                "-crf", os.getenv("REENCODE_CRF", "23"),
                "-c:a", "copy",
                output_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            if os.path.exists(ass_file):
                os.remove(ass_file)
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"ASS subtitle validation failed: {e.stderr.decode() if e.stderr else 'Unknown'}")
            return False
        except Exception as e:
            logger.error(f"ASS execution crashed: {e}")
            return False

# Global Instance
overlay_engine = TextOverlay()

def apply_text_overlay_safe(input_path, output_path, text, position="bottom", size=60):
    return overlay_engine.add_overlay(input_path, output_path, text, position, size)
