
import os
import logging
import platform
import time
import json

# Setup logger
logger = logging.getLogger("health")

# Try imports
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

# ==================== CONFIGURATION ====================
def get_env_float(key, default):
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return float(default)

CPU_USAGE_WARN_PERCENT = get_env_float("CPU_USAGE_WARN_PERCENT", 85.0)
CPU_TEMP_WARN_C = get_env_float("CPU_TEMP_WARN_C", 85.0)
MIN_RAM_FREE_MB = get_env_float("MIN_RAM_FREE_MB", 600.0)
MIN_VRAM_FREE_MB = get_env_float("MIN_VRAM_FREE_MB", 500.0)

# ==================== HEALTH CHECKS ====================

def check_cpu_health() -> dict:
    """
    Checks CPU usage and temperature.
    Returns: {"safe": bool, "reason": str}
    """
    if not psutil:
        return {"safe": True, "reason": "psutil_missing_assumed_safe"}
    
    # 1. CPU Usage (Rolling average would be better, but snapshot is okay for gate)
    # We take a small sample if needed, but blocking is bad. 
    # psutil.cpu_percent(interval=None) returns immediate since last call.
    # First call is 0.0.
    cpu_usage = psutil.cpu_percent(interval=None) 
    
    if cpu_usage > CPU_USAGE_WARN_PERCENT:
        return {"safe": False, "reason": f"CPU Load High: {cpu_usage}% > {CPU_USAGE_WARN_PERCENT}%"}

    # 2. CPU Temperature (Linux Only usually)
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current and entry.current > CPU_TEMP_WARN_C:
                            return {"safe": False, "reason": f"CPU Overheating: {entry.current}C > {CPU_TEMP_WARN_C}C"}
        except Exception as e:
            logger.warning(f"Failed to read temperatures: {e}")

    return {"safe": True, "reason": "optimal"}

def check_ram_health() -> dict:
    """
    Checks System RAM.
    Returns: {"safe": bool, "free_mb": float, "reason": str}
    """
    if not psutil:
        return {"safe": True, "free_mb": 0, "reason": "psutil_missing"}
    
    try:
        mem = psutil.virtual_memory()
        free_mb = mem.available / (1024 * 1024)
        
        if free_mb < MIN_RAM_FREE_MB:
            return {"safe": False, "free_mb": free_mb, "reason": f"Low RAM: {int(free_mb)}MB < {MIN_RAM_FREE_MB}MB"}
            
        return {"safe": True, "free_mb": free_mb, "reason": "optimal"}
    except Exception as e:
        logger.error(f"RAM check failed: {e}")
        return {"safe": True, "free_mb": 0, "reason": "check_failed"}

def check_gpu_health() -> dict:
    """
    Checks GPU VRAM (Advisory).
    Returns: {"safe": bool, "free_mb": float, "reason": str}
    """
    if not torch or not torch.cuda.is_available():
        return {"safe": True, "free_mb": 0, "reason": "no_gpu", "available": False}
        
    try:
        device = torch.device("cuda")
        # torch.cuda.mem_get_info returns (free, total) in bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_mb = free_bytes / (1024 * 1024)
        
        if free_mb < MIN_VRAM_FREE_MB:
            # We strictly only flag unsafe if we really can't fit a model
            return {"safe": False, "free_mb": free_mb, "reason": f"Low VRAM: {int(free_mb)}MB < {MIN_VRAM_FREE_MB}MB", "available": True}
            
        return {"safe": True, "free_mb": free_mb, "reason": "optimal", "available": True}
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
        return {"safe": True, "free_mb": 0, "reason": "check_failed", "available": False}

# ==================== PUBLIC API ====================

def check_health() -> dict:
    """
    Master Health Gate.
    Returns structured dict with final verdict.
    """
    cpu = check_cpu_health()
    ram = check_ram_health()
    gpu = check_gpu_health()
    
    # Aggregated Safety
    # NOTE: GPU unsafe is usually fine for CPU fallback, so we might not want to hard block 
    # unless strictly GPU_MODE=on. For now, let's treat GPU low VRAM as 'Degraded' but Safe?
    # User prompt says: "GPU is advisory only... Never hard-block job unless explicitly configured"
    
    is_safe = cpu["safe"] and ram["safe"]
    
    # Construct Verdict
    reason = []
    if not cpu["safe"]: reason.append(cpu["reason"])
    if not ram["safe"]: reason.append(ram["reason"])
    if not gpu["safe"]: reason.append(f"GPU Degraded ({gpu['reason']})")
    
    verdict = {
        "safe": is_safe,
        "timestamp": time.time(),
        "platform": platform.system(),
        "cpu_safe": cpu["safe"],
        "ram_safe": ram["safe"],
        "gpu_safe": gpu["safe"],
        "gpu_available": gpu.get("available", False),
        "cpu_reason": cpu["reason"],
        "ram_free_mb": ram["free_mb"],
        "vram_free_mb": gpu["free_mb"],
        "summary": " | ".join(reason) if reason else "System Healthy ✅"
    }
    
    return verdict

if __name__ == "__main__":
    # Manual Test
    logging.basicConfig(level=logging.INFO)
    status = check_health()
    print(json.dumps(status, indent=2))
    if status["safe"]:
        print("✅ SYSTEM READY")
    else:
        print("⛔ SYSTEM UNSAFE")
