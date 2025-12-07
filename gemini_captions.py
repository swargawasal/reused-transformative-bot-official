# gemini_captions.py - AI-Powered Caption Generator using Gemini Vision API
import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("gemini_captions")

# Try to import Gemini
try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️ google-generativeai not installed. Run: pip install google-generativeai")


class GeminiCaptionGenerator:
    """
    AI-powered caption generator using Google Gemini Vision API.
    Analyzes video frames and generates engaging, context-aware captions.
    """
    
    def __init__(self):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        if "YOUR_" in api_key or len(api_key) < 20:
            raise ValueError("GEMINI_API_KEY not configured properly. Get one from https://aistudio.google.com/app/apikey")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.5 Flash (stable, supports vision + text input, text output)
        # Supports: Text, images, video, audio inputs
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite-preview-09-2025"))
        
        # Define safety settings to prevent blocking (List format for compatibility)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Initialize Caption Cache
        self.cache_file = "captions_cache.json"
        self.caption_cache = self._load_cache()

    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                import json
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_cache(self):
        try:
            import json
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.caption_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def generate_caption(self, image_path: str, style: str = "viral") -> str:
        """
        Generate AI caption from video frame.
        """
        # ... (rest of function setup is same until the try block)
        prompts = {
            "viral": (
                "Generate a short, engaging caption (3-6 words) for this video frame. "
                "Describe the scene or emotion accurately. "
                "CRITICAL: Do NOT use single words like 'VIRAL', 'TRENDING', 'HOT', or 'SEXY'. "
                "Do NOT use meta-labels. Write a proper sentence fragment. "
                "Example: 'Golden hour glow looks amazing' or 'That outfit is absolutely stunning'."
            ),
            "descriptive": (
                "Describe what's happening in this image in ONE short, clear sentence (max 8 words). "
                "Be specific and accurate."
            ),
            "question": (
                "Generate an ENGAGING QUESTION (max 8 words) about this image that makes people want to watch. "
                "Make it intriguing and clickable!"
            ),
            "emoji": (
                "Generate a short caption with emojis (max 6 words) that's perfect for social media. "
                "Use 2-3 relevant emojis. Make it fun and eye-catching!"
            ),
            "clickbait": (
                "Generate a CLICKBAIT-style caption (max 7 words) that creates curiosity. "
                "Use phrases like 'YOU WON'T BELIEVE', 'WAIT FOR IT', 'WATCH TILL END', etc."
            ),
            "motivational": (
                "Generate a SHORT motivational caption (max 6 words) that inspires viewers. "
                "Be positive and uplifting!"
            ),
            "compilation_intro": (
                "Generate a SHORT, EXCITING intro hook (max 8 words) for a compilation video starting with this clip. "
                "Example: 'Get ready for the ultimate viral compilation!' or 'You won't believe these clips!'. "
                "Make it hype and energetic!"
            )
        }
        
        try:
            # Load image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # --- RESTORED DEFINITIONS (Fix UnboundLocalError) ---
            full_prompt_override = os.getenv("AI_CAPTION_PROMPT")
            custom_content = os.getenv("AI_CAPTION_TOPIC") or os.getenv("CAPTION_CONTENT_GEMINI")
            
            # Check for style override from .env
            env_style = os.getenv("AI_CAPTION_STYLE")
            if env_style:
                style = env_style

            # Check for custom prompt file from .env
            prompt_file = os.getenv("CAPTION_PROMPT_FILE")
            
            if prompt_file and os.path.exists(prompt_file):
                try:
                    import json
                    logger.info(f"🤖 Loading custom prompt from file: {prompt_file}")
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        p_data = json.load(f)
                    
                    # Construct Prompt from JSON components
                    system_msg = p_data.get("system", "")
                    rules = "\n".join([f"- {r}" for r in p_data.get("rules", [])])
                    style_guides = "\n".join([f"- {s}" for s in p_data.get("style", [])])
                    template = p_data.get("template", "")
                    
                    # Inject Variables
                    topic = os.getenv("AI_CAPTION_TOPIC", "") or custom_content or "viral aesthetics"
                    final_template = template.replace("{{AI_CAPTION_TOPIC}}", topic)
                    final_template = final_template.replace("{{VISUAL_HINT}}", "The attached image") # Default placeholder
                    
                    prompt = (
                        f"{system_msg}\n\n"
                        f"RULES:\n{rules}\n\n"
                        f"STYLE:\n{style_guides}\n\n"
                        f"TASK:\n{final_template}"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to parse prompt file: {e}")
                    # Fallback to normal logic below if file read fails
                    prompt_file = None

            if not prompt_file:
                # --- STANDARD LEGACY LOGIC ---
                if full_prompt_override:
                    logger.info(f"🤖 Using custom FULL prompt from .env")
                    prompt = full_prompt_override
                elif custom_content:
                    logger.info(f"🤖 Using custom caption topic: '{custom_content}'")
                    prompt = (
                        f"Generate a single, short, engaging caption (max 5 words) based on this topic/style: '{custom_content}'. "
                        "The caption must be relevant to the image provided. "
                        "CRITICAL: Return ONLY the caption text. Do NOT say 'Here is a caption' or provide multiple options. "
                        "Just the text."
                    )
                else:
                    # Get prompt for style
                    base_prompt = prompts.get(style, prompts["viral"])
                    prompt = f"{base_prompt} CRITICAL: Return ONLY the caption text. Do NOT provide options or conversational filler."
            
            logger.info(f"🤖 Generating caption with prompt length: {len(prompt)} chars")
            
            # Retry logic for 500 errors
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with Image.open(image_path) as img:
                        # Generate content with Gemini Vision
                        response = self.model.generate_content([prompt, img], safety_settings=self.safety_settings)
                        
                        # Extract text
                        caption = response.text.strip()
                        break # Success
                except Exception as e:
                    if "500" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"⚠️ Gemini 500 Error (Attempt {attempt+1}/{max_retries}). Retrying in 2s...")
                        time.sleep(2)
                    else:
                        raise e # Re-raise if not 500 or out of retries
            
            # Clean up caption
            caption = caption.replace('"', '').replace("'", '').replace('\n', ' ')
            
            # Limit length (Fix: Skip legacy truncation if JSON prompt is active)
            if not prompt_file:
                max_lengths = {
                    "viral": 50,
                    "descriptive": 60,
                    "question": 60,
                    "emoji": 50,
                    "clickbait": 60,
                    "motivational": 50
                }
                max_len = max_lengths.get(style, 50)
                
                if len(caption) > max_len:
                    caption = caption[:max_len].rsplit(' ', 1)[0]  # Cut at last word
            
            logger.info(f"✨ Generated caption: '{caption}'")
            
            # Succes! Save to Cache
            if caption not in self.caption_cache and len(caption) > 5:
                self.caption_cache.append(caption)
                self._save_cache()
                
            return caption
            
        except Exception as e:
            logger.error(f"❌ Gemini caption generation failed: {e}")
            
            # Fallback: Try Cache First (Memory)
            if self.caption_cache:
                import random
                cached_caption = random.choice(self.caption_cache)
                logger.info(f"♻️ Verification Failed/Quota Exceeded. Using Cached Fallback: '{cached_caption}'")
                return cached_caption
            
            # Fallback captions by style (Hardcoded Last Resort)
            fallbacks = {
                "viral": "Best viral moments daily",
                "descriptive": "Amazing video content",
                "question": "Can you believe this?",
                "emoji": "🤯 WATCH THIS 🔥",
                "clickbait": "YOU WON'T BELIEVE THIS",
                "motivational": "NEVER GIVE UP"
            }
            
            return fallbacks.get(style, "Best viral moments daily")
    
    def generate_hashtags(self, image_path: str, count: int = 5) -> str:
        """
        Generate relevant hashtags based on video content.
        
        Args:
            image_path: Path to video frame image
            count: Number of hashtags to generate
        
        Returns:
            Space-separated hashtags string
        """
        
        prompt = (
            f"Analyze this image and generate {count} relevant, popular hashtags "
            f"that would work well on YouTube Shorts or Instagram Reels. "
            f"Return ONLY the hashtags separated by spaces, starting with #. "
            f"Focus on trending, viral topics."
        )
        
        try:
            img = Image.open(image_path)
            response = self.model.generate_content([prompt, img], safety_settings=self.safety_settings)
            hashtags = response.text.strip()
            
            # Clean up
            hashtags = ' '.join([tag for tag in hashtags.split() if tag.startswith('#')])
            
            logger.info(f"✨ Generated hashtags: {hashtags}")
            return hashtags
            
        except Exception as e:
            logger.error(f"❌ Hashtag generation failed: {e}")
            return "#viral #trending #shorts"
    
    def generate_title(self, image_path: str) -> str:
        """
        Generate a YouTube-ready title based on video content.
        
        Args:
            image_path: Path to video frame image
        
        Returns:
            Generated title string
        """
        
        prompt = (
            "Generate a CATCHY YouTube title (max 60 characters) for this video. "
            "Make it clickable, engaging, and optimized for YouTube algorithm. "
            "Use capitalization strategically. Be creative!"
        )
        
        try:
            img = Image.open(image_path)
            response = self.model.generate_content([prompt, img], safety_settings=self.safety_settings)
            title = response.text.strip().replace('"', '').replace("'", '')
            
            if len(title) > 60:
                title = title[:60].rsplit(' ', 1)[0]
            
            logger.info(f"✨ Generated title: '{title}'")
            return title
            
        except Exception as e:
            logger.error(f"❌ Title generation failed: {e}")
            return "Amazing Video You Need To See!"

    def detect_watermark_location(self, image_path: str, target_text: str = None) -> Optional[dict]:
        """
        Detect watermark bounding box using Gemini Vision.
        Returns: {'x': int, 'y': int, 'w': int, 'h': int} or None
        """
        if not target_text:
            target_list = os.getenv("WATERMARK_DETECT_LIST", "")
            if target_list:
                target_text = target_list
            else:
                target_text = os.getenv("WATERMARK_DETECT_TEXT", "FILMYMANTRA")

        if "," in target_text:
            prompt = (
                f"Analyze this image for text overlays. "
                f"Task: Find the bounding box of the watermark. "
                f"Primary Targets (Keywords): [{target_text}]. "
                f"Secondary Targets: Any other text that looks like a watermark (e.g. @handles, URLs, 'posted by', or small floating text in corners). "
                f"Instructions: "
                f"1. Prioritize the keywords, but if you see OTHER obvious watermarks, detect them too. "
                f"2. The text is usually small, semi-transparent, and near edges/corners. "
                f"3. IGNORE: Large central captions, subtitles, faces, or essential content. "
                f"4. Return the bounding box in this EXACT format: [xmin, ymin, xmax, ymax] (0-1000 scale). "
                f"Example: [700, 800, 950, 900] "
                f"If NO watermark is found, return: None"
            )
        else:
            prompt = (
                f"Detect the bounding box of the watermark text '{target_text}' in this image. "
                f"It is a text overlay, usually small and located in a corner or along the edge. "
                f"CRITICAL: Include any ICONS, LOGOS, or SYMBOLS associated with the text (above/below/next to it). "
                f"Do NOT detect the person, face, body, or any large central captions/subtitles. "
                f"ONLY detect the specific text '{target_text}' and its logo. "
                f"Return the bounding box in this EXACT format: [xmin, ymin, xmax, ymax] "
                f"where coordinates are normalized from 0 to 1000. "
                f"Example: [700, 800, 950, 900] "
                f"If the text '{target_text}' is not found, return: None"
            )
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                response = self.model.generate_content([prompt, img], safety_settings=self.safety_settings)
                text = response.text.strip()
                
                logger.info(f"🤖 RAW GEMINI RESPONSE: {text}") # DEBUG LOGGING
                
                # Parse [xmin, ymin, xmax, ymax]
                import re
                match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
                
                if match:
                    xmin, ymin, xmax, ymax = map(int, match.groups())
                    
                    # Convert to pixels
                    x = int((xmin / 1000) * width)
                    y = int((ymin / 1000) * height)
                    w = int(((xmax - xmin) / 1000) * width)
                    h = int(((ymax - ymin) / 1000) * height)
                    
                    # Padding
                    pad = 10
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    w = min(width - x, w + 2*pad)
                    h = min(height - y, h + 2*pad)
                    
                    return {'x': x, 'y': y, 'w': w, 'h': h}
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Gemini watermark detection failed: {e}")
            return None

    def verify_watermark_coordinates(self, image_path: str, coords: dict, target_text: str = None) -> bool:
        """
        STEP 2: Verify that the detected coordinates actually contain the watermark.
        This is a critical verification step to prevent false positives.
        
        Args:
            image_path: Path to the video frame
            coords: Dictionary with 'x', 'y', 'w', 'h' keys
            target_text: Text to verify
            
        Returns:
            True if Gemini confirms watermark is at these coordinates, False otherwise
        """
        if not target_text:
            target_list = os.getenv("WATERMARK_DETECT_LIST", "")
            if target_list:
                target_text = target_list
            else:
                target_text = os.getenv("WATERMARK_DETECT_TEXT", "FILMYMANTRA")

        if "," in target_text:
            prompt = (
                f"Analyze the region at x={coords['x']}, y={coords['y']}, width={coords['w']}, height={coords['h']} in this image. "
                f"Task: Verify if any of these watermark keywords are present: [{target_text}]. "
                f"Instructions: "
                f"1. Look STRICTLY within the specified coordinates. "
                f"2. Answer 'YES' ONLY if you clearly see text that matches or partially matches the keywords. "
                f"3. Answer 'NO' if the region is empty, blurry, contains only a person/object, or contains text unrelated to the keywords. "
                f"4. Be critical. Do not hallucinate text. "
                f"Response: YES or NO."
            )
        else:
            prompt = (
                f"Analyze the region at x={coords['x']}, y={coords['y']}, width={coords['w']}, height={coords['h']} in this image. "
                f"Task: Verify if the watermark text '{target_text}' is present. "
                f"Instructions: "
                f"1. Look STRICTLY within the specified coordinates. "
                f"2. Answer 'YES' ONLY if the text '{target_text}' is clearly visible. "
                f"3. Answer 'NO' if the region is empty, blurry, contains only a person/object, or contains text unrelated to '{target_text}'. "
                f"4. Be critical. Do not hallucinate text. "
                f"Response: YES or NO."
            )
        
        try:
            with Image.open(image_path) as img:
                response = self.model.generate_content([prompt, img], safety_settings=self.safety_settings)
                answer = response.text.strip().upper()
                
                # Check for affirmative response
                is_confirmed = "YES" in answer and "NO" not in answer
                
                if is_confirmed:
                    logger.info(f"✅ Gemini VERIFIED watermark at coordinates: {coords}")
                else:
                    logger.warning(f"❌ Gemini REJECTED coordinates: {coords}. Response: {answer}")
                
                return is_confirmed
                
        except Exception as e:
            logger.error(f"❌ Gemini verification failed: {e}")
            return False



# Convenience function for quick caption generation
def generate_caption_from_video(video_path: str, style: str = "viral", timestamp: str = "00:00:01") -> Optional[str]:
    """
    Extract frame from video and generate caption.
    
    Args:
        video_path: Path to video file
        style: Caption style
        timestamp: Timestamp to extract frame from (HH:MM:SS)
    
    Returns:
        Generated caption or None if failed
    """
    import subprocess
    import tempfile
    
    try:
        # Extract frame
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            frame_path = tmp.name
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", timestamp,
            "-vframes", "1",
            frame_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Generate caption
        generator = GeminiCaptionGenerator()
        caption = generator.generate_caption(frame_path, style)
        
        return caption
        
    except Exception as e:
        logger.error(f"❌ Failed to generate caption from video: {e}")
        return None
    finally:
        # Robust cleanup
        if 'frame_path' in locals() and os.path.exists(frame_path):
            import time
            for _ in range(3):
                try:
                    os.remove(frame_path)
                    break
                except PermissionError:
                    time.sleep(0.5)
            if os.path.exists(frame_path):
                logger.warning(f"⚠️ Could not delete temp frame: {frame_path}")
        



# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 Gemini Caption Generator Test")
    print("=" * 50)
    
    if not GEMINI_AVAILABLE:
        print("❌ google-generativeai not installed")
        print("Run: pip install google-generativeai")
        exit(1)
    
    try:
        generator = GeminiCaptionGenerator()
        print("✅ Gemini initialized successfully!")
        print("\nTo test, provide an image path:")
        print("Example: python gemini_captions.py path/to/image.jpg")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("\nMake sure:")
        print("1. GEMINI_API_KEY is set in .env")
        print("2. API key is valid (get from https://aistudio.google.com/app/apikey)")
