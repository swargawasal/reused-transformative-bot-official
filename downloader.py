import os
import logging
import yt_dlp
import glob
import time
from datetime import datetime
import re
import json

logger = logging.getLogger("downloader")

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def _sanitize_filename(name: str) -> str:
    """Sanitize filename."""
    clean = re.sub(r'[^\w\s-]', '', name)
    return clean.replace(' ', '_')

def _get_next_filename(base_name: str, ext: str) -> str:
    """
    Find the next available filename with an incrementing number.
    Example: base_name="foo" -> foo_1.mp4, foo_2.mp4, etc.
    """
    pattern = f"{base_name}_*.{ext}"
    existing = glob.glob(os.path.join(DOWNLOAD_DIR, pattern))
    
    max_num = 0
    for f in existing:
        # Extract number from filename: foo_1.mp4 -> 1
        name = os.path.basename(f)
        # Regex to match base_name_(\d+).ext
        match = re.match(rf"{re.escape(base_name)}_(\d+)\.{ext}", name)
        if match:
            try:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
                
    next_num = max_num + 1
    return f"{base_name}_{next_num}.{ext}"

def download_video(url: str, custom_title: str = None) -> str:
    """
    Download video from URL synchronously.
    Strategy:
    1. Download with temporary unique name (ID/Timestamp).
    2. Rename to sanitized_title_N.mp4.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract unique identifier from URL (Instagram post ID)
    url_id = ""
    if "instagram.com" in url:
        match = re.search(r'/(?:reel|p)/([A-Za-z0-9_-]+)', url)
        if match:
            url_id = match.group(1)
            logger.info(f"📌 Extracted Instagram ID: {url_id}")
    
    # Temporary filename for download
    if url_id:
        temp_filename = f"temp_{url_id}_{timestamp}.mp4"
    else:
        temp_filename = f"temp_{timestamp}.mp4"
        
    output_path = os.path.join(DOWNLOAD_DIR, temp_filename)
    absolute_path = os.path.abspath(output_path)
    
    logger.info(f"💾 Downloading to temp: {temp_filename}")
    
    # Base options
    ydl_opts = {
        'outtmpl': absolute_path,
        'format': 'bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    metadata_holder = {}

    def _save_metadata(info_dict, video_path):
        if not info_dict: return
        try:
            # Store metadata in holder for renaming logic
            metadata_holder.update(info_dict)
            
            meta_path = video_path.rsplit('.', 1)[0] + '.json'
            metadata = {
                'uploader': info_dict.get('uploader'),
                'uploader_id': info_dict.get('uploader_id'),
                'title': info_dict.get('title'),
                'caption': info_dict.get('description'),
                'tags': info_dict.get('tags'),
                'webpage_url': info_dict.get('webpage_url')
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"📝 Saved metadata to: {meta_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save metadata: {e}")

    # --- DOWNLOAD ATTEMPTS ---
    success = False
    
    # Attempt 1: No Cookies
    if not success:
        try:
            logger.info(f"⬇️ Downloading (Attempt 1 - No Cookies): {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                _save_metadata(info, absolute_path)
            if os.path.exists(absolute_path): success = True
        except Exception as e:
            logger.warning(f"⚠️ Attempt 1 failed: {e}")

    # Attempt 2: With Cookies File
    if not success:
        cookies_path = os.getenv("COOKIES_FILE", "cookies.txt").strip('"').strip("'")
        if os.path.exists(cookies_path):
            logger.info(f"🔄 Retrying with cookies from file: {cookies_path}")
            ydl_opts['cookiefile'] = cookies_path
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    _save_metadata(info, absolute_path)
                if os.path.exists(absolute_path): success = True
            except Exception as e:
                logger.error(f"❌ Download error (Attempt 2 - File): {e}")

    # Attempt 3: With Username/Password (Instagram)
    if not success:
        ig_username = os.getenv("IG_USERNAME", "").strip()
        ig_password = os.getenv("IG_PASSWORD", "").strip()
        if ig_username and ig_password and "instagram.com" in url:
            logger.info("🔄 Retrying with Instagram credentials...")
            ydl_opts.pop('cookiefile', None)
            ydl_opts['username'] = ig_username
            ydl_opts['password'] = ig_password
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    _save_metadata(info, absolute_path)
                if os.path.exists(absolute_path): success = True
            except Exception as e:
                logger.error(f"❌ Download error (Attempt 3 - Credentials): {e}")

    # Attempt 4: With Browser Cookies (Fallback)
    if not success:
        logger.info("🔄 Retrying with cookies from browser (Chrome)...")
        ydl_opts.pop('cookiefile', None)
        ydl_opts.pop('username', None)
        ydl_opts.pop('password', None)
        ydl_opts['cookiesfrombrowser'] = ('chrome',) 
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                _save_metadata(info, absolute_path)
            if os.path.exists(absolute_path): success = True
        except Exception as e:
            logger.warning(f"⚠️ Browser cookies failed: {e}")

    if not success:
        logger.error("❌ All download attempts failed.")
        return None

    # --- RENAME LOGIC ---
    try:
        # Determine title
        title = custom_title
        if not title:
            title = metadata_holder.get('title', 'video')
            
        clean_title = _sanitize_filename(title)
        # Truncate if too long
        clean_title = clean_title[:50]
        
        # Robust Rename Loop
        # Try up to 100 times to find a free name
        final_path = absolute_path # Default fallback
        
        for i in range(0, 1000):
            if i == 0:
                candidate_name = f"{clean_title}.mp4"
            else:
                candidate_name = f"{clean_title}_{i}.mp4"
                
            candidate_path = os.path.join(DOWNLOAD_DIR, candidate_name)
            candidate_abs_path = os.path.abspath(candidate_path)
            
            if not os.path.exists(candidate_abs_path):
                try:
                    os.rename(absolute_path, candidate_abs_path)
                    final_path = candidate_abs_path
                    logger.info(f"✅ Renamed video to: {candidate_name}")
                    
                    # Rename JSON if exists
                    old_json = absolute_path.rsplit('.', 1)[0] + '.json'
                    new_json = candidate_abs_path.rsplit('.', 1)[0] + '.json'
                    if os.path.exists(old_json):
                        # Ensure JSON target doesn't exist either
                        if os.path.exists(new_json):
                            try: os.remove(new_json)
                            except: pass
                        os.rename(old_json, new_json)
                        logger.info(f"✅ Renamed metadata to: {os.path.basename(new_json)}")
                    
                    return final_path
                    
                except OSError as e:
                    # WinError 183 or FileExistsError
                    if e.errno == 17 or getattr(e, 'winerror', 0) == 183:
                        logger.warning(f"⚠️ Collision for {candidate_name}, trying next...")
                        continue
                    else:
                        raise e # Other error
            
        logger.error("❌ Could not find a free filename after 100 attempts.")
        return absolute_path
        
    except Exception as e:
        logger.error(f"❌ Rename failed: {e}. Returning temp path.")
        return absolute_path