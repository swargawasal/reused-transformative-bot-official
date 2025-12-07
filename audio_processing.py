import os
import logging
import subprocess

logger = logging.getLogger("audio_processing")

# Configuration
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
ENABLE_TREMOLO = os.getenv("ENABLE_TREMOLO", "no").lower() == "yes"
COMPILATION_MASTER_MODE = os.getenv("COMPILATION_MASTER_MODE", "heavy")
def heavy_remix(input_path: str, output_path: str, original_volume: float = 1.15) -> bool:
    """
    AUDIO REMIX FIX — Clean Beat Preset
    Removes noise, adds bass/treble boost, compression, and volume.
    
    Args:
        input_path: Input audio/video file
        output_path: Output audio file
        original_volume: Volume multiplier for original audio (default 1.15)
    """
    logger.info(f"🎛️ Audio Remix Fix: {input_path} (Volume: {original_volume}x)")
    
    # Effects Chain
    effects = [
        "atempo=1.03",
        "equalizer=f=80:t=h:w=100:g=3",
        "equalizer=f=12000:t=h:w=2000:g=2",
        "acompressor=threshold=-14dB:ratio=2.5:attack=20:release=200",
        f"volume={original_volume}",
    ]
    
    if ENABLE_TREMOLO:
        effects.append("tremolo=f=1:d=0.4")
    
    effects.append("alimiter=limit=0.95")
    
    af_filter = ",".join(effects)
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-af", af_filter,
        "-vn", # No video
        "-ac", "2", "-ar", "44100",
        output_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("✅ Audio Remix Complete (Clean Beat Preset)")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Audio Remix (Complex) failed: {e}. Trying simple volume boost...")
        
        # Fallback: Simple Volume Boost (No fancy filters)
        cmd_fallback = [
            FFMPEG_BIN, "-y", "-i", input_path,
            "-af", f"volume={original_volume}",
            "-vn", 
            "-ac", "2", "-ar", "44100",
            output_path
        ]
        try:
            subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            logger.info("✅ Audio Remix Complete (Volume Only Fallback)")
            return True
        except subprocess.CalledProcessError as e2:
            logger.error(f"❌ Audio remix fallback also failed: {e2}")
            return False

def mix_background_music(input_video: str, output_video: str, volume: float = 0.15) -> bool:
    """
    Mixes a random background track from 'music/' folder into the video.
    """
    import random
    import glob
    
    music_dir = "music"
    if not os.path.exists(music_dir):
        logger.warning(f"⚠️ Music directory '{music_dir}' not found.")
        return False
        
    tracks = glob.glob(os.path.join(music_dir, "*.mp3"))
    if not tracks:
        logger.warning(f"⚠️ No mp3 tracks found in '{music_dir}'.")
        return False
        
    bg_track = random.choice(tracks)
    logger.info(f"🎵 Mixing background music: {os.path.basename(bg_track)} (Vol: {volume})")
    
    # Mix: [1:a]volume=0.15[bg];[0:a]volume=1.0[main];[main][bg]amix=inputs=2:duration=first[a]
    # We use duration=first to match video length
    # We also loop the music if it's shorter? amix doesn't loop. 
    # Better to use -stream_loop -1 for music input.
    
    # Mix with Ducking: Music volume lowers when main audio speaks
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_video,
        "-stream_loop", "-1", "-i", bg_track,
        "-filter_complex", f"[1:a]volume={volume}[bg];[0:a]volume=1.0,sidechaincompress=threshold=0.05:ratio=8:attack=50:release=300[main];[main][bg]amix=inputs=2:duration=first[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        output_video
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Background music mix failed: {e}")
        return False

def apply_compilation_mastering(input_path: str, output_path: str, original_volume: float = 1.2) -> bool:
    """
    HEAVY REMIX for Compilations.
    Adds 'Stadium' reverb, heavier bass, and compression for a transformative feel.
    
    Args:
        input_path: Input audio/video file
        output_path: Output audio file
        original_volume: Volume multiplier for original audio (default 1.2)
    """
    logger.info(f"🏟️ Applying Compilation Mastering (Heavy Remix): {input_path} (Volume: {original_volume}x)")
    
    # Effects Chain:
    # 1. Bass Boost (Stronger)
    # 2. Exciter (Highs)
    # 3. Stadium Reverb (aecho)
    # 4. Compression (Glue)
    # 5. Limiter
    # 6. Pitch Up + Tempo Fix (Transformative)
    
    filter_chain = ""
    if COMPILATION_MASTER_MODE == "lite":
        filter_chain = (
            "asetrate=44100*1.03,atempo=1/1.03,"
            "equalizer=f=100:t=h:w=120:g=3,"
            "acompressor=threshold=-14dB:ratio=2.5,"
            f"volume={original_volume}"
        )
    else:
        # Heavily Distanced (Default)
        filter_chain = (
            "asetrate=44100*1.05," 
            "atempo=1/1.05,"
            "aecho=0.8:0.88:60:0.4,"
            "equalizer=f=60:t=h:w=100:g=5,"
            "equalizer=f=12000:t=h:w=2000:g=3,"
            "acompressor=threshold=-12dB:ratio=4:attack=5:release=50,"
            f"volume={original_volume}"
        )
    
    # Add random pitch noise at final stage
    filter_chain += ",rubberband=pitch=random(0.997,1.003)"
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-af", filter_chain,
        "-vn", # Audio only first
        "-ac", "2", "-ar", "44100",
        output_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("✅ Compilation Mastering Complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Compilation mastering failed: {e}")
        return False

def generate_transformative_music(input_paths: list, output_music: str, duration: float) -> bool:
    """
    Generates a unique, royalty-free ambient background track by sampling multiple source videos.
    
    Features:
    - Multi-Clip Sampling: Takes samples from different input clips.
    - Chaos Engine: Applies randomized effect chains to each sample.
    - Layering: Creates a complex, evolving soundscape.
    """
    import random
    import shutil
    
    # Handle single string input
    if isinstance(input_paths, str):
        input_paths = [input_paths]
        
    logger.info(f"🎹 Generating Advanced Transformative Music from {len(input_paths)} sources...")
    
    # Derive ffprobe from ffmpeg bin if possible, or default to ffprobe
    FFPROBE_BIN = "ffprobe"
    if "ffmpeg" in FFMPEG_BIN.lower():
        FFPROBE_BIN = FFMPEG_BIN.replace("ffmpeg", "ffprobe")
    
    temp_dir = os.path.join(os.path.dirname(output_music), "temp_music_gen")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        samples = []
        
        # We want to generate enough samples to fill the duration roughly, 
        # but since we loop, we just need a good "base loop" of maybe 10-15 seconds.
        # Let's create a 15s base loop from ~10 samples of 1.5s each.
        
        target_samples = 10
        
        # Effect Pool
        effect_pool = [
            "areverse", 
            "asetrate=44100*0.5", # Slow down
            "asetrate=44100*0.75", # Moderate slow
            "aecho=0.8:0.9:1000:0.3", # Heavy Reverb
            "lowpass=f=500", # Underwater
            "highpass=f=2000", # Thin/Radio
            "tremolo=f=5:d=0.5", # Choppy
            "vibrato=f=6:d=0.5", # Wobbly
            "aphaser=in_gain=0.4:out_gain=0.5:delay=3.0:decay=0.4:speed=0.5:type=t" # Phaser
        ]
        
        for i in range(target_samples):
            # 1. Pick a random source clip
            src_video = random.choice(input_paths)
            if not os.path.exists(src_video):
                logger.warning(f"⚠️ Source video not found: {src_video}")
                continue
            
            # 2. Extract a random 1.5s chunk
            # Need duration of this clip
            try:
                dur_cmd = [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", src_video]
                src_dur_str = subprocess.check_output(dur_cmd).decode().strip()
                if not src_dur_str:
                    logger.warning(f"⚠️ Could not determine duration for {src_video}")
                    continue
                src_dur = float(src_dur_str)
            except Exception as e:
                logger.warning(f"⚠️ Failed to get duration for {src_video}: {e}")
                continue
                
            if src_dur < 2.0:
                logger.warning(f"⚠️ Source too short ({src_dur}s): {src_video}")
                continue
            
            start = random.uniform(0, max(0, src_dur - 1.6))
            sample_raw = os.path.join(temp_dir, f"raw_{i}.wav")
            sample_processed = os.path.join(temp_dir, f"proc_{i}.wav")
            
            # Extract raw audio
            try:
                subprocess.run([FFMPEG_BIN, "-y", "-ss", str(start), "-t", "1.5", "-i", src_video, "-vn", "-ac", "1", sample_raw],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.decode() if e.stderr else str(e)
                logger.warning(f"⚠️ Failed to extract audio sample from {src_video}: {err_msg}")
                continue
            
            # 3. Build Random Effect Chain (Chaos Engine)
            # Always have some reverb/ambience, but mix others
            chain = []
            
            # 50% chance of reverse
            if random.random() > 0.5: chain.append("areverse")
            
            # Always slow down for ambient feel (random rate)
            rate = random.choice(["0.5", "0.6", "0.75", "0.8"])
            chain.append(f"asetrate=44100*{rate}")
            
            # 1 or 2 random effects
            chain.extend(random.sample(effect_pool[3:], k=random.randint(1, 2)))
            
            # Always end with some reverb to smooth cuts
            chain.append("aecho=0.8:0.88:60:0.4")
            
            af_filter = ",".join(chain)
            
            # Apply effects
            try:
                subprocess.run([FFMPEG_BIN, "-y", "-i", sample_raw, "-af", af_filter, sample_processed],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                samples.append(sample_processed)
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.decode() if e.stderr else str(e)
                logger.warning(f"⚠️ Failed to process sample: {err_msg}")
                continue
            
        if not samples:
            logger.error("❌ Failed to generate any samples. (Check if input videos have audio)")
            return False
            
        # 4. Concatenate Samples to create the Loop
        concat_list = os.path.join(temp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for s in samples:
                f.write(f"file '{os.path.abspath(s).replace(os.sep, '/')}'\n")
                # Add a small crossfade? No, hard cuts with reverb sound glitchy/cool.
                
        loop_segment = os.path.join(temp_dir, "loop.wav")
        subprocess.run([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", loop_segment],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                      
        # 5. Loop to full duration
        cmd_final = [
            FFMPEG_BIN, "-y", "-stream_loop", "-1", "-i", loop_segment,
            "-t", str(duration),
            "-t", str(duration),
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,volume=0.3,acompressor=threshold=-12dB:ratio=2:attack=5:release=50", # Master + Loudness
            output_music
        ]
        subprocess.run(cmd_final, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        logger.info("✅ Generated Advanced Transformative Track")
        return True
        
    except Exception as e:
        logger.error(f"❌ Music Generation Failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def create_continuous_music_mix(output_path: str, target_duration: float, music_dir: str = "music") -> bool:
    """
    Creates a continuous music mix by stitching multiple different songs from the music directory
    until the target duration is reached.
    """
    import random
    import glob
    
    if not os.path.exists(music_dir):
        logger.warning(f"⚠️ Music directory '{music_dir}' not found.")
        return False
        
    music_files = glob.glob(os.path.join(music_dir, "*.mp3")) + glob.glob(os.path.join(music_dir, "*.wav"))
    if not music_files:
        logger.warning(f"⚠️ No music files found in '{music_dir}'.")
        return False
        
    logger.info(f"🎵 Creating continuous mix for {target_duration:.1f}s from {len(music_files)} tracks...")
    
    # Shuffle tracks for randomness
    random.shuffle(music_files)
    
    # Select tracks until we have enough duration
    selected_tracks = []
    current_dur = 0.0
    
    # We might need to loop the playlist if total duration of all songs is less than target
    playlist = music_files.copy()
    
    while current_dur < target_duration:
        if not playlist:
            # Refill playlist if empty (looping the set of songs)
            playlist = music_files.copy()
            random.shuffle(playlist)
            
        track = playlist.pop(0)
        
        # Get track duration
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", track
            ]
            # Use shell=True on Windows if needed, or just standard run
            # Assuming ffprobe is in path or we use the global FFMPEG_BIN logic if we had access to it here.
            # We'll assume 'ffprobe' is available since this is a helper.
            # Better: use the one defined in this file if possible, but FFMPEG_BIN is defined at top.
            # Let's try to use the FFMPEG_BIN logic from top of file
            ffprobe_bin = FFMPEG_BIN.replace("ffmpeg", "ffprobe") if "ffmpeg" in FFMPEG_BIN.lower() else "ffprobe"
            
            cmd[0] = ffprobe_bin
            dur_str = subprocess.check_output(cmd).decode().strip()
            track_dur = float(dur_str)
            
            selected_tracks.append(track)
            current_dur += track_dur
            
        except Exception as e:
            logger.warning(f"⚠️ Could not read duration of {track}: {e}")
            continue
            
    if not selected_tracks:
        return False
        
    # Create concat list
    temp_list = os.path.join(os.path.dirname(output_path), "mix_list.txt")
    try:
        with open(temp_list, "w") as f:
            for track in selected_tracks:
                f.write(f"file '{os.path.abspath(track).replace(os.sep, '/')}'\n")
        
        # Concat and Trim
        # -safe 0 is needed for absolute paths
        cmd = [
            FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", temp_list,
            "-t", str(target_duration),
            "-c", "copy", # Try copy first? Might fail if codecs differ.
            output_path
        ]
        
        # If copy fails (different codecs), we re-encode
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            logger.info("⚠️ Stream copy failed (different codecs?), re-encoding mix...")
            cmd = [
                FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", temp_list,
                "-t", str(target_duration),
                "-ac", "2", "-ar", "44100", # Standardize audio
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Music mix failed: {e}")
        return False
    finally:
        if os.path.exists(temp_list):
            os.remove(temp_list)

def detect_silence(audio_file: str) -> bool:
    """
    Checks if the audio file has significant silence using ffmpeg silencedetect.
    Returns True if silence is detected.
    """
    cmd = [
        FFMPEG_BIN, "-i", audio_file,
        "-af", "silencedetect=noise=-40dB:d=2",
        "-f", "null", "-"
    ]
    try:
        p = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        return "silence_start" in p.stderr
    except Exception as e:
        logger.warning(f"⚠️ Silence detection failed: {e}")
        return False
