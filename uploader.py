import os
import time
import logging
from typing import Optional
import asyncio
import subprocess
import sys
import json
import shutil

FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
if not shutil.which(FFPROBE_BIN):
    FFPROBE_BIN = "ffprobe"

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET_FILE = os.environ.get("CLIENT_SECRET_FILE", "client_secret.json")
TOKEN_FILE = os.environ.get("YOUTUBE_TOKEN_FILE", "token.json")

logger = logging.getLogger("uploader")
logger.setLevel(logging.INFO)


def _get_service_sync():
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            logger.warning("Failed to read token file, will run auth flow.")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                logger.warning("Refresh failed; performing new auth flow.")
                creds = None
        if not creds:
            logger.warning("🔄 Token expired or missing. Launching auto-auth...")
            try:
                # Auto-run the auth script
                subprocess.check_call([sys.executable, "scripts/auth_youtube.py"])
                
                # Reload credentials after script finishes
                if os.path.exists(TOKEN_FILE):
                    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            except Exception as e:
                logger.error(f"❌ Auto-auth failed: {e}")

            if not creds or not creds.valid:
                logger.error("❌ Authentication failed: Token expired or missing.")
                raise Exception("YouTube Authentication Failed. Please run 'python scripts/auth_youtube.py' locally to refresh credentials.")

    service = build("youtube", "v3", credentials=creds)
    return service


def verify_metadata(file_path: str) -> bool:
    """
    Checks if the video file has fresh metadata (Unique ID, Creation Time).
    Returns True if fresh, False otherwise.
    """
    try:
        cmd = [
            FFPROBE_BIN, "-v", "quiet", 
            "-print_format", "json", 
            "-show_format", 
            file_path
        ]
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        data = json.loads(result)
        tags = data.get("format", {}).get("tags", {})
        
        comment = tags.get("comment", "")
        creation_time = tags.get("creation_time", "")
        
        is_fresh = False
        if "Unique ID:" in comment:
            logger.info(f"✅ Metadata Verified: Found Unique ID in comments.")
            is_fresh = True
        else:
            logger.warning(f"⚠️ Metadata Warning: No 'Unique ID' found in file comments.")
            
        if creation_time:
            logger.info(f"✅ Metadata Verified: Creation Time = {creation_time}")
        else:
            logger.warning(f"⚠️ Metadata Warning: No 'creation_time' found.")
            
        return is_fresh
    except Exception as e:
        logger.warning(f"⚠️ Failed to verify metadata: {e}")
        return False


def _upload_sync(
    file_path: str,
    hashtags: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy: str = "public",
) -> Optional[str]:
    # Enforce .mp4 extension
    if not file_path.lower().endswith(".mp4"):
        logger.error("❌ Upload rejected: File must be .mp4")
        return None

    service = _get_service_sync()
    final_title = (title or "Untitled").strip()
    final_description = ((description or "").strip() + ("\n\n" + hashtags if hashtags else "")).strip()

    body = {
        "snippet": {
            "title": final_title,
            "description": final_description,
            "categoryId": "22",  # People & Blogs
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        file_path,
        chunksize=1024 * 1024 * 16,  # 16 MB
        resumable=True,
        mimetype="video/mp4"
    )

    logger.info("🚀 Starting upload request to YouTube API...")
    
    # Verify Metadata Freshness
    verify_metadata(file_path)
    
    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )
    logger.info("✅ Upload request created. Starting chunk upload loop...")

    logger.info("🚀 Starting upload: %s", file_path)
    retry = 0
    while True:
        try:
            status, response = request.next_chunk()
            if response is not None:
                video_id = response.get("id")
                if video_id:
                    logger.info("✅ Upload complete: %s", video_id)
                    return f"https://youtube.com/watch?v={video_id}"
                return None
            if status and hasattr(status, "progress"):
                progress = int(status.progress() * 100)
                logger.info("Upload progress: %d%%", progress)
        except HttpError as e:
            logger.warning("YouTube API HttpError on chunk: %s", e)
            retry += 1
            if retry > 5:
                logger.error("Max retries reached for upload due to HttpError.")
                return None
            time.sleep(2 ** retry)
        except Exception as e:
            logger.exception("Upload error: %s", e)
            retry += 1
            if retry > 5:
                logger.error("Max retries reached for upload due to Exception.")
                return None
            time.sleep(2 ** retry)


async def upload_to_youtube(
    file_path: str,
    hashtags: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy: str = "public",
) -> Optional[str]:
    print(f"DEBUG: uploader.upload_to_youtube called for {file_path}")
    return await asyncio.to_thread(_upload_sync, file_path, hashtags, title, description, privacy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_file = os.environ.get("TEST_UPLOAD_FILE", "downloads/final_highres_output.mp4")
    # Create a dummy file for testing if it doesn't exist
    if not os.path.exists(test_file):
        with open(test_file, "wb") as f:
            f.write(b"dummy content")
            
    link = _upload_sync(test_file, hashtags="#example", title="High-Quality Test Upload")
    print("Uploaded:", link)
