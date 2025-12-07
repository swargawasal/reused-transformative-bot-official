"""
AI Voiceover Generator
Uses gTTS (Google Text-to-Speech) to generate micro-commentary.
"""

import os
import logging
import random
from typing import Optional

logger = logging.getLogger("voiceover")

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    logger.warning("⚠️ gTTS not installed. Voiceover will be disabled.")

class VoiceoverGenerator:
    def __init__(self):
        self.enabled = os.getenv("ENABLE_MICRO_VOICEOVER", "yes").lower() == "yes"
        self.lang = "en"
        self.tld = "com" # 'co.uk', 'com.au', 'co.in' for accents
        
        # Randomize accent slightly for uniqueness if desired
        self.accents = ["com", "co.uk", "us", "ca", "co.in"]
        
    def generate_voiceover(self, text: str, output_path: str) -> bool:
        """
        Generate MP3 voiceover from text.
        """
        if not self.enabled or not HAS_GTTS:
            return False
            
        if not text or len(text) < 5:
            return False
            
        try:
            # Ensure directory exists
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            
            logger.info(f"🗣️ Generating Voiceover: '{text}'")
            
            # Select accent (Determinism check)
            seed = os.getenv("VOICEOVER_SEED")
            if seed:
                random.seed(seed)
            tld = random.choice(self.accents)
            
            tts = gTTS(text=text, lang=self.lang, tld=tld, slow=False)
            tts.save(output_path)
            
            return os.path.exists(output_path)
            
        except Exception as e:
            logger.error(f"❌ Voiceover generation failed: {e}")
            return False

# Global Instance
voice_engine = VoiceoverGenerator()

def generate_voiceover(text: str, output_path: str) -> bool:
    return voice_engine.generate_voiceover(text, output_path)
