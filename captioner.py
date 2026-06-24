#!/usr/bin/env python3
"""
Dataset Turbo Captioner — Image Captioning Edition


Launch: python3 dataset_turbo_captioner.py
Opens:  http://127.0.0.1:7860
"""

import base64
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
import json
import queue
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template_string, request
from PIL import Image as PILImage

# ─────────────────────────────────────────────────────────────────────────────
# EARLY ENVIRONMENT & DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def _diag(msg):
    print(f"[DIAG] {msg}", flush=True)

os.makedirs("./models", exist_ok=True)

def _check_llama_binary():
    """Check if llama-server is accessible."""
    raw = "llama-server"
    if shutil.which(raw):
        return True
    if sys.platform == "win32" and shutil.which(raw + ".exe"):
        return True
    return False

if _check_llama_binary():
    _diag("llama-server found in PATH")
else:
    _diag("llama-server not in PATH. Will auto-detect or install on Models tab.")

# ─────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_FILE = "config.json"

class AppState:
    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    persisted = json.load(f)
                    self.config.update(persisted)
            except Exception as e:
                print(f"Warning: Could not load config.json: {e}")

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save config.json: {e}")

    def __init__(self):
        self.config = {
            "llama_server_bin": os.environ.get("LLAMA_SERVER_BIN", "llama-server"),
            "model_path": os.environ.get("LLAMA_MODEL", ""),
            "mmproj_path": os.environ.get("LLAMA_MMPROJ", ""),
            "model_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
            "port": int(os.environ.get("LLAMA_PORT", 8080)),
            "ctx_size": 16384,
            "gpu_layers": 99
        }
        self.stats_lock = threading.Lock()
        self._load_config()
        if not os.path.exists(CONFIG_FILE):
            self.save_config()

        self.is_server_running = False
        self.server_process = None
        self.active_model_path = None
        self.active_mmproj_path = None

        self.log_lines = []
        self.log_lock = threading.Lock()
        self.batch_stop = False
        self.batch_running = False
        self.batch_thread = None
        self.batch_queue = None
        self.batch_config = {}
        self.batch_progress = {"current": 0, "total": 0, "start_time": None, "eta": "Calculating..."}
        self.current_image_b64 = None
        self.download_progress = {"active": False, "current_file": "", "files_done": 0, "files_total": 0, "error": None, "message": ""}

state = AppState()

def _log(msg: str):
    with state.log_lock:
        state.log_lines.append(msg)
        print(msg, flush=True)

def _safe_get(cfg, key, default=None):
    val = cfg.get(key, default)
    return val if val is not None else default

# ─────────────────────────────────────────────────────────────────────────────
# MODEL UTILS
# ─────────────────────────────────────────────────────────────────────────────
def _scan_available_models(model_dir=None):
    if model_dir is None:
        model_dir = state.config.get("model_dir", "./models")
    ggufs = []
    mmprojs = []

    if os.path.exists(model_dir):
        try:
            for f in sorted(os.listdir(model_dir)):
                fp = os.path.join(model_dir, f)
                if not os.path.isfile(fp):
                    continue
                lower_f = f.lower()
                if not lower_f.endswith(".gguf"):
                    continue
                if 'mmproj' in lower_f or 'projector' in lower_f:
                    mmprojs.append({"name": f, "path": fp})
                else:
                    ggufs.append({"name": f, "path": fp})
        except Exception as e:
            _log(f"Error scanning model_dir {model_dir}: {e}")

    return {
        "gguf": sorted(ggufs, key=lambda x: x['name']),
        "mmproj": sorted(mmprojs, key=lambda x: x['name'])
    }

# ─────────────────────────────────────────────────────────────────────────────
# LLAMA-SERVER LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────
def _llama_running():
    try:
        r = requests.get(f"http://127.0.0.1:{state.config['port']}/health", timeout=3)
        return r.status_code == 200 and "ok" in r.text.lower()
    except Exception:
        return False

def _resolve_llama_server_bin():
    """Resolve the llama-server binary path."""
    raw = state.config.get("llama_server_bin", "llama-server")

    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw

    found = shutil.which(raw)
    if found:
        return found

    if sys.platform == "win32":
        if not raw.lower().endswith(".exe"):
            found = shutil.which(raw + ".exe")
            if found:
                return found

        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, raw),
            os.path.join(script_dir, raw + ".exe"),
            os.path.join(script_dir, "llama", raw),
            os.path.join(script_dir, "llama", raw + ".exe"),
            os.path.join(script_dir, "bin", raw),
            os.path.join(script_dir, "bin", raw + ".exe"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    return raw

def _start_llama_server(config_override=None):
    global state

    if config_override:
        state.config.update(config_override)

    model_path = state.config.get("model_path", "")
    mmproj_path = state.config.get("mmproj_path", "")

    if model_path:
        model_path = os.path.abspath(model_path)
    if mmproj_path:
        mmproj_path = os.path.abspath(mmproj_path)

    state.config["model_path"] = model_path
    state.config["mmproj_path"] = mmproj_path

    if not model_path or not os.path.isfile(model_path):
        return f"Model file not found or invalid: {model_path}"

    if mmproj_path:
        if not os.path.isfile(mmproj_path):
            return f"MMProj file not found: {mmproj_path}. Please select a valid projector file."
        _log(f"  Loading Vision Projector: {os.path.basename(mmproj_path)}")
    else:
        _log("  No MMProj provided. Vision/Image inputs will fail.")

    if state.is_server_running and state.active_model_path == model_path and state.active_mmproj_path == mmproj_path:
        return "already running"

    if state.is_server_running:
        _log(f"  Config changed. Restarting server...")
        _stop_llama_server()

    resolved_bin = _resolve_llama_server_bin()
    if not os.path.isfile(resolved_bin):
        return (
            f"llama-server binary not found: '{state.config.get('llama_server_bin', '')}'\n"
            f"Resolved to: {resolved_bin}\n\n"
            f"Please install llama-server from the Models tab."
        )

    _log(f"  Starting llama-server...")
    _log(f"     Binary: {os.path.basename(resolved_bin)}")
    _log(f"     Model : {os.path.basename(model_path)}")
    _log(f"     Proj  : {os.path.basename(mmproj_path) if mmproj_path else 'None (Text Only)'}")

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_server.log")

    cmd = [
        resolved_bin,
        "-m", model_path,
        "-ngl", str(state.config["gpu_layers"]),
        "--ctx-size", str(state.config["ctx_size"]),
        "--port", str(state.config["port"]),
        "--host", "127.0.0.1",
        "--flash-attn", "auto",
        "--threads", str(os.cpu_count() or 8),
        "--threads-batch", str(max(8, (os.cpu_count() or 8) // 2)),
    ]

    if mmproj_path and os.path.isfile(mmproj_path):
        cmd.extend(["--mmproj", mmproj_path])

    try:
        with open(log_path, "w") as lf:
            lf.write("Starting llama-server...\n")
        lf = open(log_path, "a", encoding="utf-8")

        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW

        state.server_process = subprocess.Popen(cmd, stdout=lf, stderr=lf, creationflags=creation_flags)
    except FileNotFoundError as e:
        return (
            f"Cannot start llama-server: {e}\n"
            f"Command: {cmd[0]}\n\n"
            f"Ensure llama-server is installed from the Models tab."
        )
    except Exception as e:
        return f"Popen failed: {e}"

    state.active_model_path = model_path
    state.active_mmproj_path = mmproj_path

    for i in range(180):
        time.sleep(1)
        if i % 5 == 0 and i > 0:
            _log(f"  Waiting for server to initialize... ({i}s)")

        if _llama_running():
            try:
                log_content = Path(log_path).read_text(encoding="utf-8")
                if mmproj_path and "error" in log_content.lower() and "mmproj" in log_content.lower():
                     _log(f"  Server started but MMProj load might have failed. Check llama_server.log")
            except:
                pass

            _log(f"  llama-server ready after {i+1}s")
            state.is_server_running = True
            return "started"

        if state.server_process.poll() is not None:
            try:
                tail = Path(log_path).read_text(encoding="utf-8")[-1500:]
            except Exception:
                tail = "(log unreadable)"
            return f"Server crashed after {i+1}s. Check llama_server.log:\n{tail}"

    _log(f"  llama-server did not respond in 180s. Check {log_path}")
    return "timeout"

def _stop_llama_server():
    global state
    if state.server_process and state.server_process.poll() is None:
        state.server_process.terminate()
        try:
            state.server_process.wait(timeout=10)
        except Exception:
            state.server_process.kill()
    state.server_process = None
    state.is_server_running = False
    state.active_model_path = None
    state.active_mmproj_path = None

# ─────────────────────────────────────────────────────────────────────────────
# LLAMA.CPP INSTALLER — Hardware Detection & Binary Download
# ─────────────────────────────────────────────────────────────────────────────
import platform
import struct

LLAMA_CPP_INSTALL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama.cpp")
GITHUB_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

def _detect_hardware():
    """Detect OS, architecture, and GPU vendor for optimal binary selection."""
    info = {
        "os": sys.platform,
        "os_name": "Unknown",
        "arch": platform.machine().lower(),
        "arch_name": "Unknown",
        "gpu_vendor": "None",
        "gpu_name": "Unknown",
        "cuda_version": None,
    }

    if sys.platform == "win32":
        info["os_name"] = "Windows"
    elif sys.platform == "linux":
        info["os_name"] = "Linux"
    elif sys.platform == "darwin":
        info["os_name"] = "macOS"
    else:
        info["os_name"] = sys.platform

    machine = info["arch"]
    if machine in ("x86_64", "amd64", "x64"):
        info["arch_name"] = "x64"
    elif machine in ("aarch64", "arm64"):
        info["arch_name"] = "arm64"
    elif machine in ("i386", "i686", "x86"):
        info["arch_name"] = "x64"
    else:
        info["arch_name"] = machine

    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
            if r.returncode == 0 and r.stdout.strip():
                line = r.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1:
                    info["gpu_name"] = parts[0]
                    info["gpu_vendor"] = "NVIDIA"
                if len(parts) >= 2:
                    r2 = subprocess.run(
                        ["nvidia-smi"], capture_output=True, text=True, timeout=5,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                    )
                    if r2.returncode == 0:
                        import re as _re
                        m = _re.search(r"CUDA Version:\s*([\d.]+)", r2.stdout)
                        if m:
                            info["cuda_version"] = m.group(1)
        except Exception:
            pass
    elif sys.platform == "linux":
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                line = r.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1:
                    info["gpu_name"] = parts[0]
                    info["gpu_vendor"] = "NVIDIA"
                if len(parts) >= 2:
                    r2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                    if r2.returncode == 0:
                        import re as _re
                        m = _re.search(r"CUDA Version:\s*([\d.]+)", r2.stdout)
                        if m:
                            info["cuda_version"] = m.group(1)
        except Exception:
            pass

    if sys.platform == "linux" and info["gpu_vendor"] == "None":
        try:
            r = subprocess.run(
                ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                for line in r.stdout.strip().split("\n"):
                    if "GPU" in line and ":" in line:
                        info["gpu_name"] = line.split(":", 1)[-1].strip()
                        info["gpu_vendor"] = "AMD"
                        break
        except Exception:
            pass

    if sys.platform in ("win32", "linux") and info["gpu_vendor"] == "None":
        try:
            if sys.platform == "win32":
                r = subprocess.run(
                    ["wmic", "path", "win32_videocontroller", "get", "name"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                )
            else:
                r = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=5
                )
            if r.returncode == 0 and r.stdout.strip():
                for line in r.stdout.strip().split("\n"):
                    lower = line.lower()
                    if "intel" in lower and ("arc" in lower or "iris" in lower or "uhd" in lower or "hd graphics" in lower):
                        info["gpu_name"] = line.strip()
                        info["gpu_vendor"] = "Intel"
                        break
        except Exception:
            pass

    return info

def _match_release_asset(assets, hw_info):
    """Match the best release asset for the detected hardware."""
    os_name = hw_info["os_name"]
    arch = hw_info["arch_name"]
    gpu = hw_info["gpu_vendor"]
    cuda_ver = hw_info.get("cuda_version")

    patterns = []

    if os_name == "Windows":
        if gpu == "NVIDIA":
            if cuda_ver:
                cuda_major = cuda_ver.split(".")[0]
                patterns.append(f"win-cuda-{cuda_ver[:4]}-x64")
                patterns.append(f"win-cuda-{cuda_major}-x64")
            patterns.append("win-cuda-13.3-x64")
            patterns.append("win-cuda-12.4-x64")
        elif gpu == "AMD":
            patterns.append("win-hip-radeon-x64")
        elif gpu == "Intel":
            patterns.append("win-sycl-x64")
            patterns.append("win-openvino-2026.2-x64")
        patterns.append(f"win-cpu-{arch}")

    elif os_name == "Linux":
        if gpu == "NVIDIA":
            if cuda_ver:
                cuda_major = cuda_ver.split(".")[0]
                patterns.append(f"ubuntu-cuda-{cuda_ver[:4]}-x64")
                patterns.append(f"ubuntu-cuda-{cuda_major}-x64")
            patterns.append("ubuntu-cuda-12.4-x64")
            patterns.append("ubuntu-cuda-13.3-x64")
        elif gpu == "AMD":
            patterns.append("ubuntu-rocm-7.2-x64")
        patterns.append(f"ubuntu-vulkan-{arch}")
        patterns.append(f"ubuntu-x64")

    elif os_name == "macOS":
        patterns.append(f"macos-{arch}")
        patterns.append(f"macos-x64")

    scored = []
    for asset in assets:
        name = asset["name"].lower()
        if not name.endswith((".zip", ".tar.gz")):
            continue
        if "ui" in name or "xcframework" in name:
            continue

        score = 0
        matched_pattern = None
        for i, pat in enumerate(patterns):
            if pat.lower() in name:
                score = len(patterns) - i
                matched_pattern = pat
                break

        if score > 0:
            scored.append((score, asset, matched_pattern))

    if not scored:
        return None, None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored[0][2]

def _install_prebuilt_binary(asset, hw_info):
    """Download and extract a prebuilt binary from a GitHub release asset."""
    url = asset["browser_download_url"]
    name = asset["name"]
    install_dir = LLAMA_CPP_INSTALL_DIR

    _log(f"  Downloading {name}...")
    _log(f"  URL: {url}")

    os.makedirs(install_dir, exist_ok=True)

    tmp_path = os.path.join(tempfile.gettempdir(), name)

    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = (downloaded / total) * 100
                    if downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                        _log(f"  Download progress: {pct:.0f}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)")

        _log(f"  Download complete: {downloaded // (1024*1024)}MB")

        _log(f"  Extracting to {install_dir}...")
        if name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                zf.extractall(install_dir)
        elif name.endswith(".tar.gz"):
            import tarfile
            with tarfile.open(tmp_path, 'r:gz') as tf:
                tf.extractall(install_dir)

        _log(f"  Extraction complete.")

        bin_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        found_bin = None
        for root, dirs, files in os.walk(install_dir):
            for f in files:
                if f == bin_name or f == "llama-server.exe":
                    found_bin = os.path.join(root, f)
                    break
            if found_bin:
                break

        if found_bin:
            _log(f"  Found binary: {found_bin}")
            state.config["llama_server_bin"] = found_bin
            state.save_config()
            return True, found_bin
        else:
            _log(f"  Warning: {bin_name} not found in extracted files.")
            return False, f"Binary {bin_name} not found in archive"

    except Exception as e:
        _log(f"  Installation failed: {e}")
        return False, str(e)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# HUGGINGFACE MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
PRESET_DOWNLOADS = {
    "qwen3-vl-8b": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "files": ["Qwen3-VL-8B-Instruct-Q8_0.gguf", "mmproj-Qwen3-VL-8B-Instruct-f16.gguf"],
        "label": "Qwen 3 8B VL"
    },
    "gemma-4-12b": {
        "repo": "unsloth/gemma-4-12b-it-GGUF",
        "files": ["gemma-4-12b-it-Q8_0.gguf", "mmproj-gemma-4-12b-it-f16.gguf"],
        "label": "Gemma 4 12B"
    }
}

def _download_hf_files(repo_id, filenames):
    """Download files from HuggingFace using aria2c."""
    state.download_progress = {
        "active": True, "current_file": "", "files_done": 0,
        "files_total": len(filenames), "error": None, "message": ""
    }

    model_dir = state.config.get("model_dir", "./models")
    os.makedirs(model_dir, exist_ok=True)

    for i, filename in enumerate(filenames):
        state.download_progress["current_file"] = filename
        state.download_progress["message"] = f"Downloading {filename} ({i+1}/{len(filenames)})..."
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

        try:
            cmd = [
                "aria2c", "-x", "16", "-s", "16", "-k", "1M",
                "-d", model_dir, "-o", filename, url
            ]
            _log(f"  Downloading {filename} from {repo_id}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                state.download_progress["error"] = f"{filename}: {result.stderr[:200]}"
                _log(f"  Download failed: {filename}: {result.stderr[:200]}")
                break
            state.download_progress["files_done"] = i + 1
            _log(f"  Downloaded: {filename}")
        except subprocess.TimeoutExpired:
            state.download_progress["error"] = f"{filename}: Download timed out"
            _log(f"  Download timed out: {filename}")
            break
        except Exception as e:
            state.download_progress["error"] = f"{filename}: {e}"
            _log(f"  Download error: {filename}: {e}")
            break

    state.download_progress["active"] = False
    if not state.download_progress["error"]:
        state.download_progress["message"] = f"All {len(filenames)} files downloaded successfully!"
        _log(f"  All files from {repo_id} downloaded successfully!")
    else:
        state.download_progress["message"] = f"Download stopped: {state.download_progress['error']}"

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def _save_checkpoint(output_folder, remaining_images, completed, total, config):
    """Save batch checkpoint for resume."""
    checkpoint = {
        "remaining_images": remaining_images,
        "completed": completed,
        "total": total,
        "config": config,
    }
    checkpoint_path = os.path.join(output_folder, "_caption_checkpoint.json")
    try:
        os.makedirs(output_folder, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        _log(f"  Checkpoint saved: {completed}/{total} done, {len(remaining_images)} remaining")
    except Exception as e:
        _log(f"  Failed to save checkpoint: {e}")

def _load_checkpoint(output_folder):
    """Load batch checkpoint."""
    checkpoint_path = os.path.join(output_folder, "_caption_checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _clear_checkpoint(output_folder):
    """Clear batch checkpoint."""
    checkpoint_path = os.path.join(output_folder, "_caption_checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def _frame_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()

def _load_image(file_path):
    img = PILImage.open(file_path).convert("RGB")
    return img

# ─────────────────────────────────────────────────────────────────────────────
# CAPTION CALL
# ─────────────────────────────────────────────────────────────────────────────
def _call_llama(system_prompt, user_text, frames, max_tokens=2048, temperature=0.2):
    content = []

    for f in frames:
        try:
            b64_img = _frame_to_b64(f)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            })
        except Exception as e:
            _log(f"  Failed to encode frame to base64: {e}")

    combined_user_text = f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n\n[USER INPUT]\n{user_text}"
    content.insert(0, {"type": "text", "text": combined_user_text})

    model_name = ""
    if state.config.get("model_path"):
        base_name = os.path.basename(state.config["model_path"])
        model_name = os.path.splitext(base_name)[0]
        if not model_name:
            model_name = ""

    payload = {
        "model": model_name if model_name else "",
        "messages": [
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "clear_history": True
    }

    _log(f"  Calling LLM API... (Model: '{model_name or 'default'}', Frames: {len(frames)}, Max Tokens: {max_tokens}, Context Size: {state.config.get('ctx_size')})")

    url = f"http://127.0.0.1:{state.config['port']}/v1/chat/completions"
    try:
        r = requests.post(url, json=payload, timeout=240)

        if r.status_code != 200:
            _log(f"  API HTTP {r.status_code}: {r.text[:500]}")
            raise RuntimeError(f"API error {r.status_code}")

        data = r.json()

        if "choices" not in data or not data["choices"]:
            _log(f"  API returned empty choices: {data}")
            raise RuntimeError("Empty response from llama-server")

        raw_content = data["choices"][0].get("message", {}).get("content", "")

        if not raw_content or not raw_content.strip():
            _log(f"  LLM returned empty/whitespace content.")
            raise RuntimeError("LLM returned empty response.")

        _log(f"  LLM returned text: {raw_content[:50]}...")

        return raw_content
    except requests.exceptions.ConnectionError:
         _log(f"  Cannot connect to llama-server at {url}. Is it running?")
         raise RuntimeError("Connection refused")
    except Exception as e:
        _log(f"  _call_llama failed: {e}")
        raise

def _clean(raw):
    raw = raw.split("</s>")[-1] if "</s>" in raw else raw
    raw = re.sub(r"```[a-z]*\n?|```", "", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = re.sub(r"^\s*[-*•]\s*", "", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 40]
    if not paras:
        paras = [p.strip() for p in raw.split("\n") if len(p.strip()) > 40]

    return max(paras, key=len) if paras else raw.strip()

def _is_bad(cap):
    if not cap or not cap.strip():
        return "empty"
    if len(cap.split()) < 30:
        return f"too short ({len(cap.split())} words)"
    bad_starts = ("here are","here is","i need to","let me","okay,","the caption","i will","i cannot","i'm sorry","i apologize","i don't","i can't","as an ai")
    first = cap.strip().lower()[:80]
    if any(first.startswith(b) for b in bad_starts):
        return "bad start"
    return ""

def _caption_one(file_path, system_prompt, trigger):
    IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tiff",".tif"}
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in IMG_EXTS:
        return None, f"Unsupported file type: {ext}"

    if not _llama_running():
        return None, "llama-server is not running or ready. Please check Models tab."

    if not state.config.get("mmproj_path") or not os.path.exists(state.config["mmproj_path"]):
        return None, "Cannot caption image: No Vision Projector (MMProj) is loaded. Please select an MMProj file in the Models tab and restart the server."

    try:
        img = _load_image(file_path)
        w, h = img.size
        if max(w, h) > 768:
            s = 768 / max(w, h)
            img = img.resize((int(w * s), int(h * s)), PILImage.LANCZOS)
        frames = [img]
        state.current_image_b64 = _frame_to_b64(img)
    except Exception as e:
        return None, f"image load failed: {e}"

    user_t = "This is a single still image. Write one training caption as a single flowing paragraph. Present tense. 80-160 words. Start immediately."
    max_tokens_img = 1536

    try:
        if not frames:
            raise RuntimeError("No frames available to send to LLM.")

        raw = _call_llama(system_prompt, user_t, frames, max_tokens=max_tokens_img)

        cap = _clean(raw)

        if not cap or len(cap.split()) < 20:
            raise RuntimeError("Caption became too short after cleaning.")

        if _is_bad(cap):
            _log("  Caption flagged as bad start, retrying with generic prompt...")
            try:
                raw = _call_llama(system_prompt, "Look at the image and describe what you see as one flowing paragraph. Shot scale, lighting, subject anatomy, clothing, action. 80-200 words. Start immediately.", frames, max_tokens=max_tokens_img)
                cap = _clean(raw)
            except Exception as retry_err:
                _log(f"  Retry also failed: {retry_err}. Skipping file.")
                return None, f"Caption generation failed after retry: {retry_err}"

        if not cap or len(cap.split()) < 20:
            return None, "Caption too short or empty after retry."

        trig = trigger.strip()
        if trig and trig.lower() not in cap.lower():
            cap = trig + ", " + cap
        return cap, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def _producer(image_paths, q, stats):
    try:
        for img_src in image_paths:
            if state.batch_stop:
                break
            q.put({"type": "image", "path": img_src})
    finally:
        pass

def _consumer(cfg, q, stats, total_count):
    def update_prog():
        with state.stats_lock:
            state.batch_progress["current"] += 1
            if state.batch_progress["start_time"]:
                elapsed = time.time() - state.batch_progress["start_time"]
                rate = state.batch_progress["current"] / elapsed if elapsed > 0 else 0
                remaining = state.batch_progress["total"] - state.batch_progress["current"]
                eta_seconds = remaining / rate if rate > 0 else 0
                state.batch_progress["eta"] = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"

    system_prompt = _safe_get(cfg, "system_prompt", "")
    trigger       = _safe_get(cfg, "trigger", "")

    while True:
        try:
            item = q.get(timeout=1)
        except queue.Empty:
            if state.batch_stop:
                break
            continue

        if item is None:
            q.task_done()
            break

        if state.batch_stop:
            q.task_done()
            continue

        try:
            src = item["path"]
            fname = os.path.basename(src)
            output_folder = _safe_get(cfg, "output_folder", "") or _safe_get(cfg, "input_folder", "")
            txt = os.path.join(output_folder, os.path.splitext(fname)[0] + ".txt")

            if item["type"] == "image":
                temp_dst = os.path.join(tempfile.gettempdir(), f"prepped_{fname}")
                try:
                    pil = _load_image(src)
                    w, h = pil.size
                    if max(w, h) != 768:
                        s = 768 / max(w, h)
                        pil = pil.resize((int(w * s), int(h * s)), PILImage.LANCZOS)

                    state.current_image_b64 = _frame_to_b64(pil)

                    ext_l = os.path.splitext(src)[1].lower()
                    fmt = "JPEG" if ext_l in (".jpg",".jpeg") else "PNG"
                    kw = {"quality": 95} if fmt == "JPEG" else {}
                    pil.save(temp_dst, format=fmt, **kw)
                    pil.close()

                    target_path = temp_dst
                    target_params = (temp_dst, system_prompt, trigger)
                except Exception as e:
                    _log(f"  image prep failed for {fname}: {e}")
                    with state.stats_lock:
                        stats['errors'] += 1
                    q.task_done()
                    continue

            retries = 0
            cap, err = None, None
            while retries < 3:
                try:
                    cap, err = _caption_one(*target_params)
                    if err and ("server" in err.lower() or "connection" in err.lower()):
                        raise RuntimeError(err)
                    break
                except (RuntimeError, requests.exceptions.RequestException) as e:
                    retries += 1
                    _log(f"  Server failure processing {fname} (attempt {retries}/3): {e}")
                    if retries < 3:
                        _log("  Attempting automatic server restart...")
                        status = _start_llama_server()
                        if "started" in status or "already running" in status:
                            _log(f"  Server recovered. Retrying...")
                            time.sleep(2)
                        else:
                            _log(f"  Restart failed: {status}")
                    else:
                        err = f"Server failed after 3 attempts: {e}"

            if err:
                _log(f"  caption failed for {fname}: {err}")
                with state.stats_lock:
                    stats['errors'] += 1
            else:
                Path(txt).write_text(cap, encoding="utf-8")
                _log(f"  {fname}: {cap[:100]}...")
                with state.stats_lock:
                    stats['done'] += 1
            update_prog()

        except Exception as e:
            _log(f"  unexpected error processing {item.get('path')}: {e}")
            with state.stats_lock:
                stats['errors'] += 1
            update_prog()
        finally:
            q.task_done()

def _run_batch(cfg, resume=False):
    global state
    state.batch_stop = False
    state.batch_config = cfg
    _log(f"  Batch started: input={_safe_get(cfg,'input_folder','')}")

    output_folder = _safe_get(cfg, "output_folder", "") or _safe_get(cfg, "input_folder", "")
    os.makedirs(output_folder, exist_ok=True)

    if resume:
        checkpoint = _load_checkpoint(output_folder)
        if not checkpoint:
            _log("  No checkpoint found. Starting fresh.")
            resume = False
        else:
            remaining = checkpoint["remaining_images"]
            completed = checkpoint["completed"]
            total = checkpoint["total"]
            _log(f"  Resuming: {completed}/{total} done, {len(remaining)} remaining")
    else:
        _clear_checkpoint(output_folder)
        input_folder = _safe_get(cfg, "input_folder", "")
        if not os.path.isdir(input_folder):
            _log(f"  input folder not found: {input_folder}")
            return

        skip_ex = _safe_get(cfg, "skip_existing", False)
        IMAGE_EXTS = (".jpg",".jpeg",".png",".webp",".bmp",".tiff",".tif")
        all_files = sorted(os.listdir(input_folder))
        all_images = [os.path.join(input_folder, f) for f in all_files if f.lower().endswith(IMAGE_EXTS)]

        if skip_ex:
            remaining = []
            skipped = 0
            for img_path in all_images:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                txt = os.path.join(output_folder, stem + ".txt")
                if os.path.exists(txt):
                    skipped += 1
                    continue
                remaining.append(img_path)
            if skipped > 0:
                _log(f"  Skipped {skipped} images with existing captions")
        else:
            remaining = all_images
        completed = 0
        total = len(all_images)

    _log("  checking llama-server status...")
    if not _llama_running():
         _log("  llama-server is not running. Attempting start...")
         status = _start_llama_server()
         if "not found" in status or "error" in status or "crashed" in status or "timeout" in status:
             _log(f"  llama-server failed: {status}")
             return
    else:
         _log("  llama-server is already running")

    _log("  Warming up LLM...")
    try:
        dummy_img = PILImage.new('RGB', (100, 100), color='black')
        _call_llama("You are an image captioner.", "Describe this image.", [dummy_img], max_tokens=50)
        _log("  LLM warm-up complete.")
    except Exception as e:
        _log(f"  Warm-up failed (ignoring): {e}")

    q = queue.Queue()
    state.batch_queue = q
    stats = {'done': 0, 'skipped': 0, 'errors': 0}

    state.batch_progress["total"] = total
    state.batch_progress["start_time"] = time.time()
    state.batch_progress["current"] = completed

    prod_thread = threading.Thread(target=_producer, args=(remaining, q, stats), daemon=True)
    prod_thread.start()

    num_workers = int(_safe_get(cfg, "num_workers", 1))
    _log(f"  Starting {num_workers} consumer worker(s)...")
    consumers = []
    for i in range(num_workers):
        t = threading.Thread(target=_consumer, args=(cfg, q, stats, total), daemon=True)
        t.start()
        consumers.append(t)

    prod_thread.join()

    for _ in range(num_workers):
        q.put(None)

    for t in consumers:
        t.join()

    if state.batch_stop:
        remaining_items = []
        while not q.empty():
            try:
                item = q.get_nowait()
                if item and item.get("type") == "image":
                    remaining_items.append(item["path"])
                q.task_done()
            except queue.Empty:
                break
        _save_checkpoint(output_folder, remaining_items, state.batch_progress["current"], total, cfg)
    else:
        _clear_checkpoint(output_folder)

    _log(f"\n{'─'*54}")
    if state.batch_stop:
        _log(f"  STOPPED — {stats['done']} captioned, {stats['skipped']} skipped, {stats['errors']} errors")
        _log(f"  Press Resume to continue from checkpoint.")
    else:
        _log(f"  done: {stats['done']} captioned, {stats['skipped']} skipped, {stats['errors']} errors")
    _log(f"{'─'*54}")
    state.batch_progress["eta"] = "Done" if not state.batch_stop else "Stopped"

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP & UI
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Captioner</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#fafafa;--bg-soft:#f4f4f4;--bg-card:#ffffff;
  --border:#ddd;--border-soft:#eee;
  --ink:#1a1a1a;--ink-2:#2a2a2a;--ink-3:#444;
  --muted:#555;--muted-2:#777;--muted-3:#888;--muted-4:#999;--muted-5:#aaa;
  --accent:#0f3787;--accent-deep:#0a275f;--accent-soft:#dde6f5;--accent-hover:#c8d6ee;
  --success:#16a34a;--danger:#dc2626;
  --radius:4px;
  --font-sans:'Inter','Helvetica Neue',sans-serif;
  --font-mono:'JetBrains Mono',monospace;
  --shadow-sm:0 1px 2px rgba(0,0,0,0.04);
  --shadow-md:0 2px 8px rgba(0,0,0,0.06);
}
[data-theme="amoled"]{
  --bg:#000000;--bg-soft:#0a0a0a;--bg-card:#111111;
  --border:#222;--border-soft:#1a1a1a;
  --ink:#f0f0f0;--ink-2:#e0e0e0;--ink-3:#ccc;
  --muted:#aaa;--muted-2:#888;--muted-3:#777;--muted-4:#555;--muted-5:#444;
  --accent:#4d8bf5;--accent-deep:#3a7ae0;--accent-soft:rgba(77,139,245,0.12);--accent-hover:rgba(77,139,245,0.18);
  --shadow-sm:0 1px 2px rgba(0,0,0,0.3);
  --shadow-md:0 2px 8px rgba(0,0,0,0.4);
}
body{
  font-family:var(--font-sans);font-size:12px;font-weight:300;line-height:1.6;
  color:var(--ink);background:var(--bg);
  padding:24px 32px 64px;overflow-x:hidden;
  transition:background 0.3s,color 0.3s;
}
@media(max-width:640px){body{padding:16px 14px 48px;}h1{font-size:18px;}}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--muted-3);}
*{scrollbar-width:thin;scrollbar-color:var(--border) transparent;}
.header-row{
  display:flex;justify-content:space-between;align-items:flex-start;
  gap:24px;flex-wrap:wrap;margin-bottom:16px;padding-bottom:12px;
  border-bottom:1px solid var(--border);
}
.title-block{flex:1 1 auto;min-width:0;}
.title-row{display:flex;align-items:center;gap:16px;flex-wrap:wrap;}
h1{
  font-family:var(--font-mono);font-size:22px;font-weight:500;
  letter-spacing:0.2px;color:var(--ink);line-height:1.25;
}
.subtitle{
  font-size:12px;font-weight:300;color:var(--muted);
  margin-top:8px;max-width:700px;line-height:1.55;
}
.toolbar{display:flex;align-items:center;gap:8px;flex-shrink:0;}
.btn{
  font-family:var(--font-mono);font-size:10px;font-weight:400;
  letter-spacing:0.5px;padding:5px 11px;border:1px solid var(--border);
  border-radius:var(--radius);background:var(--bg-card);color:var(--muted);
  cursor:pointer;transition:all 0.15s;text-decoration:none;
  display:inline-flex;align-items:center;line-height:1.4;min-height:26px;
}
.btn:hover{border-color:var(--muted-3);color:var(--ink);}
.btn.active{background:var(--ink);color:#fff;border-color:var(--ink);}
[data-theme="amoled"] .btn.active{background:var(--accent);border-color:var(--accent);}
.btn-primary{
  font-family:var(--font-mono);font-size:11px;font-weight:500;
  letter-spacing:0.8px;text-transform:uppercase;
  padding:7px 14px;border:1px solid var(--accent);
  background:var(--accent);color:#fff;cursor:pointer;
  border-radius:var(--radius);transition:all 0.15s;
  display:inline-flex;align-items:center;text-decoration:none;
}
.btn-primary:hover{background:var(--accent-deep);border-color:var(--accent-deep);}
.btn-primary:disabled{opacity:0.4;cursor:not-allowed;}
.btn-danger{
  font-family:var(--font-mono);font-size:11px;font-weight:500;
  letter-spacing:0.8px;text-transform:uppercase;
  padding:7px 14px;border:1px solid var(--danger);
  background:var(--danger);color:#fff;cursor:pointer;
  border-radius:var(--radius);transition:all 0.15s;
  display:inline-flex;align-items:center;text-decoration:none;
  width:100%;
}
.btn-danger:hover{background:#b91c1c;border-color:#b91c1c;}
.btn-success{
  font-family:var(--font-mono);font-size:11px;font-weight:500;
  letter-spacing:0.8px;text-transform:uppercase;
  padding:7px 14px;border:1px solid var(--success);
  background:var(--success);color:#fff;cursor:pointer;
  border-radius:var(--radius);transition:all 0.15s;
  display:inline-flex;align-items:center;text-decoration:none;
  width:100%;
}
.btn-success:hover{background:#15803d;border-color:#15803d;}
.theme-toggle{
  display:flex;align-items:center;gap:6px;
  font-family:var(--font-mono);font-size:10px;font-weight:400;
  color:var(--muted-3);letter-spacing:0.5px;cursor:pointer;
  user-select:none;padding:4px 8px;border:1px solid var(--border);
  border-radius:var(--radius);background:var(--bg-card);transition:all 0.15s;
}
.theme-toggle:hover{border-color:var(--muted-3);color:var(--ink);}
.theme-toggle .icon{font-size:13px;}
.tab-bar{
  width:100%;display:flex;gap:0;margin-bottom:20px;
  border-bottom:1px solid var(--border);
}
.tab-btn{
  font-family:var(--font-mono);font-size:11px;font-weight:500;
  padding:8px 20px;border:none;background:transparent;
  color:var(--muted-3);cursor:pointer;letter-spacing:0.8px;
  text-transform:uppercase;transition:all 0.15s;position:relative;
}
.tab-btn::after{
  content:'';position:absolute;bottom:-1px;left:0;
  width:100%;height:2px;background:var(--accent);
  transform:scaleX(0);transition:transform 0.2s;
}
.tab-btn.active{color:var(--accent);}
.tab-btn.active::after{transform:scaleX(1);}
.tab-btn:hover{color:var(--ink);}
.tab-panel{width:100%;display:none;}
.tab-panel.active{display:block;}
.layout{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
@media(max-width:700px){.layout{grid-template-columns:1fr;}}
.section-title{
  font-family:var(--font-mono);font-size:10px;font-weight:400;
  text-transform:uppercase;letter-spacing:1.5px;color:var(--ink-3);
  margin-top:20px;margin-bottom:10px;border-bottom:1px solid var(--border);
  padding-bottom:6px;display:flex;align-items:center;gap:10px;
}
.section-title:first-child{margin-top:0;}
.section-title .hint{
  margin-left:auto;color:var(--muted-4);font-size:9px;
  letter-spacing:0.5px;text-transform:none;font-weight:300;
}
.card{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;margin-bottom:12px;
  box-shadow:var(--shadow-sm);transition:background 0.3s,border-color 0.3s;
}
.section{padding:14px 16px;border-bottom:1px solid var(--border-soft);}
.section:last-child{border-bottom:none;}
.slabel{
  font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:1px;text-transform:uppercase;color:var(--muted-2);
  margin-bottom:8px;
}
.path-input{
  width:100%;background:var(--bg-soft);border:1px solid var(--border);
  border-radius:var(--radius);color:var(--ink);
  font-family:var(--font-sans);font-size:12px;font-weight:300;
  padding:7px 10px;outline:none;transition:border-color 0.15s;
  margin-bottom:6px;
}
.path-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(15,55,135,0.08);}
[data-theme="amoled"] .path-input:focus{box-shadow:0 0 0 3px rgba(77,139,245,0.15);}
textarea.path-input{resize:vertical;min-height:60px;line-height:1.55;}
select.path-input{
  appearance:none;-webkit-appearance:none;
  background-image:url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23777%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20128a17.5%2017.5%200%200%200%2025.3%200l128-128c3.6-3.6%205.4-7.9%205.4-12.9%200-5-1.8-9.3-5.4-12.9z%2F%3E%3C%2Fsvg%3E");
  background-repeat:no-repeat;background-position:right 8px top 50%;
  background-size:0.6rem auto;padding-right:24px;
}
.hint{font-size:10px;color:var(--muted-3);margin-top:4px;line-height:1.5;}
.toggle-row{
  display:flex;align-items:center;gap:10px;cursor:pointer;
  user-select:none;margin-bottom:8px;
}
.toggle-track{
  width:34px;height:18px;background:var(--border);
  border-radius:9px;position:relative;flex-shrink:0;
  transition:background 0.2s;
}
.toggle-track::after{
  content:'';position:absolute;top:2px;left:2px;
  width:14px;height:14px;background:var(--muted-4);
  border-radius:50%;transition:left 0.2s,background 0.2s;
}
.toggle-track.on{background:var(--accent-soft);}
.toggle-track.on::after{left:18px;background:var(--accent);}
.toggle-label{font-size:11px;color:var(--ink-2);font-weight:300;}
.log-box{
  background:var(--bg-soft);border:1px solid var(--border);
  border-radius:var(--radius);padding:10px;
  font-family:var(--font-mono);font-size:10px;line-height:1.8;
  color:var(--ink-3);height:380px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-word;
}
.status-dot{
  display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--border);margin-right:6px;flex-shrink:0;
  transition:background 0.3s;
}
.status-dot.green{background:var(--success);}
.status-dot.red{background:var(--danger);}
.status-row{
  display:flex;align-items:center;
  font-family:var(--font-mono);font-size:10px;color:var(--muted-3);
  margin-top:8px;
}
.val-dot{
  width:8px;height:8px;border-radius:50%;display:inline-block;
  margin-right:8px;background:var(--border);transition:background 0.3s;
}
.val-dot.valid{background:var(--success);}
.val-dot.invalid{background:var(--danger);}
.val-row{display:flex;align-items:center;margin-bottom:6px;font-size:11px;}
.progress-container{display:flex;flex-direction:column;gap:6px;margin-top:12px;}
.progress-bar-bg{width:100%;height:3px;background:var(--border);border-radius:2px;overflow:hidden;}
.progress-bar-fill{width:0%;height:100%;background:var(--accent);transition:width 0.3s;}
.progress-text{
  font-family:var(--font-mono);font-size:10px;color:var(--muted-3);
  text-align:center;letter-spacing:0.3px;
}
.preview-container{margin-top:12px;text-align:center;}
.preview-img{max-width:200px;max-height:200px;border:1px solid var(--border);border-radius:var(--radius);}
.preview-label{
  font-family:var(--font-mono);font-size:9px;letter-spacing:1px;
  text-transform:uppercase;color:var(--muted-4);margin-top:6px;
}
.preset-card{
  display:flex;justify-content:space-between;align-items:center;
  padding:10px 12px;border:1px solid var(--border);
  border-radius:var(--radius);margin-bottom:8px;
  transition:border-color 0.15s;
}
.preset-card:hover{border-color:var(--accent);}
.preset-info{font-size:11px;color:var(--ink-2);line-height:1.5;}
.preset-info strong{font-size:12px;color:var(--ink);display:block;}
.preset-info span{font-size:10px;color:var(--muted-3);}
.hf-repo{
  padding:8px 12px;border:1px solid var(--border);
  border-radius:var(--radius);margin-bottom:4px;
  cursor:pointer;transition:all 0.15s;
}
.hf-repo:hover{border-color:var(--accent);background:var(--accent-soft);}
.hf-repo.selected{border-color:var(--accent);background:var(--accent-soft);}
.hf-repo strong{font-size:11px;color:var(--ink);}
.hf-repo span{font-size:10px;color:var(--muted-3);margin-left:8px;}
.hf-file{
  display:flex;align-items:center;gap:8px;
  padding:6px 10px;border:1px solid var(--border);
  border-radius:var(--radius);margin-bottom:4px;
  cursor:pointer;font-size:11px;color:var(--ink-2);
  transition:border-color 0.15s;
}
.hf-file:hover{border-color:var(--accent);}
.hf-file input[type="checkbox"]{accent-color:var(--accent);}
.download-status{
  padding:10px 12px;border:1px solid var(--border);
  border-radius:var(--radius);margin-top:8px;
  font-size:11px;color:var(--ink-2);background:var(--bg-soft);
}
</style>
</head>
<body>

<div class="header-row">
  <div class="title-block">
    <div class="title-row">
      <h1>Dataset Captioner</h1>
    </div>
    <div class="subtitle">Image captioning studio powered by local LLM inference via llama-server.</div>
  </div>
  <div class="toolbar">
    <div class="theme-toggle" onclick="toggleTheme()" title="Toggle AMOLED dark mode">
      <span class="icon" id="theme-icon">&#9790;</span>
      <span id="theme-label">Dark</span>
    </div>
  </div>
</div>

<div class="tab-bar">
  <div class="tab-btn active" id="tb-images" onclick="switchTab('images')">Images</div>
  <div class="tab-btn" id="tb-models" onclick="switchTab('models')">Models</div>
</div>

<!-- IMAGES TAB -->
<div class="tab-panel active" id="panel-images">
  <div class="section-title">Configuration</div>
  <div class="layout">
    <div>
      <div class="card"><div class="section">
        <div class="slabel">Folders</div>
        <input class="path-input" id="i_input_folder" placeholder="Input folder">
        <input class="path-input" id="i_output_folder" placeholder="Output folder (blank = same as input)">
      </div></div>

      <div class="card"><div class="section">
        <div class="slabel">Training Type</div>
        <select class="path-input" id="i_training_type" onchange="onTrainingTypeChange(this)"></select>
        <div class="slabel" style="margin-top:12px;">System Prompt</div>
        <textarea class="path-input" id="i_system_prompt" rows="10" placeholder="System prompt for creating captions..."></textarea>
      </div></div>

      <div class="card"><div class="section">
        <div class="slabel">Trigger Token</div>
        <input class="path-input" id="i_trigger" placeholder="e.g. tami (blank = none)">
        <div class="hint">Prefix every caption with this token</div>
      </div></div>

      <div class="card"><div class="section">
        <div class="slabel">Options</div>
        <label class="toggle-row" onclick="itog('skip_existing')">
          <div class="toggle-track on" id="itr-skip_existing"></div>
          <span class="toggle-label">Skip images with existing .txt sidecar</span>
        </label>
      </div></div>
    </div>

    <div>
      <div class="card"><div class="section">
        <div class="slabel">Quick Test</div>
        <button class="btn" style="width:100%;" onclick="testSingle()">Test First File</button>
      </div></div>

      <div class="card"><div class="section">
        <div class="slabel">Run</div>
        <button class="btn-primary" id="i_go-btn" onclick="startImages()" style="width:100%;">Start Captioning</button>
        <div class="progress-container" id="i_progress-container" style="display:none;">
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="i_progress-fill"></div></div>
          <div class="progress-text" id="i_progress-text">Processed 0/0 &middot; ETA: Calculating...</div>
          <div id="i_live-preview" class="preview-container" style="display:none;">
            <img id="i_preview-img" class="preview-img" src="">
            <div class="preview-label">Live Processing</div>
          </div>
        </div>
        <div class="status-row">
          <span class="status-dot" id="i_status-dot"></span>
          <span id="i_status-text">llama-server: unknown</span>
        </div>
      </div></div>

      <div class="card"><div class="section">
        <div class="slabel">Live Log</div>
        <div class="log-box" id="i_log-box">Ready.</div>
      </div></div>
    </div>
  </div>
</div>

<!-- MODELS TAB -->
<div class="tab-panel" id="panel-models">
  <div class="section-title">llama.cpp Installer</div>
  <div class="layout">
    <div class="card"><div class="section">
      <div class="slabel">System Info</div>
      <div id="hw-info" style="font-size:11px;color:var(--muted-2);margin-bottom:12px;">Detecting hardware...</div>
      <div id="install-status" style="font-size:11px;color:var(--muted-2);margin-bottom:12px;"></div>
      <button class="btn-primary" id="btn-auto-install" onclick="autoInstallBinary()" style="width:100%;">Auto Download &amp; Install Best Binary</button>
      <div class="hint" style="margin-top:6px;">Automatically detects your hardware and installs the optimal llama-server binary.</div>
    </div></div>
  </div>

  <div class="section-title" style="margin-top:24px;">Model Configuration</div>
  <div class="layout">
    <div class="card"><div class="section">
      <div class="slabel">Status</div>
      <div style="font-size:11px;color:var(--muted-2);margin-bottom:12px;" id="model-status-display">
        Loading status...
      </div>

      <div class="slabel">Select Model (.gguf)</div>
      <select id="model-select" class="path-input" style="margin-bottom:4px;">
        <option value="">-- Scanning --</option>
      </select>
      <div class="val-row">
        <span id="model-val-dot" class="val-dot"></span>
        <span id="model-val-text" style="color:var(--muted-3);">Waiting for validation</span>
      </div>

      <div class="slabel" style="margin-top:12px;">Select MMProj</div>
      <select id="mmproj-select" class="path-input" style="margin-bottom:4px;">
        <option value="">-- Scanning --</option>
      </select>
      <div class="val-row">
        <span id="mmproj-val-dot" class="val-dot"></span>
        <span id="mmproj-val-text" style="color:var(--muted-3);">Waiting for validation</span>
      </div>

      <button class="btn" style="width:100%;margin-top:12px;margin-bottom:12px;" onclick="validateFiles()">Validate Selected Files</button>

      <button class="btn-primary" id="apply-model-btn" onclick="applyModel()" disabled style="width:100%;">Apply &amp; Restart Server</button>
      <div class="hint" style="margin-top:6px;">Validation ensures files exist before attempting to restart the server.</div>
    </div></div>
  </div>

  <div class="section-title" style="margin-top:24px;">Model Downloader</div>
  <div class="layout">
    <div>
      <div class="card"><div class="section">
        <div class="slabel">Quick Downloads</div>
        <div class="preset-card">
          <div class="preset-info">
            <strong>Qwen 3 8B VL</strong>
            <span>Q8_0 main model + F16 mmproj</span>
            <span>Qwen/Qwen3-VL-8B-Instruct-GGUF</span>
          </div>
          <button class="btn-primary" onclick="downloadPreset('qwen3-vl-8b')" style="white-space:nowrap;">Download</button>
        </div>
        <div class="preset-card">
          <div class="preset-info">
            <strong>Gemma 4 12B</strong>
            <span>Q8_0 main model + F16 mmproj</span>
            <span>unsloth/gemma-4-12b-it-GGUF</span>
          </div>
          <button class="btn-primary" onclick="downloadPreset('gemma-4-12b')" style="white-space:nowrap;">Download</button>
        </div>
        <div id="download-status-box" class="download-status" style="display:none;">
          <div id="download-status-text"></div>
          <div class="progress-container" style="margin-top:6px;">
            <div class="progress-bar-bg"><div class="progress-bar-fill" id="download-progress-fill"></div></div>
          </div>
        </div>
      </div></div>
    </div>

    <div>
      <div class="card"><div class="section">
        <div class="slabel">Search HuggingFace</div>
        <div style="display:flex;gap:8px;margin-bottom:8px;">
          <input class="path-input" id="hf-search-input" placeholder="Search for VLM GGUFs..." style="flex:1;margin-bottom:0;">
          <button class="btn-primary" onclick="searchHF()" style="white-space:nowrap;">Search</button>
        </div>
        <div id="hf-results" style="max-height:200px;overflow-y:auto;"></div>
        <div id="hf-files" style="display:none;margin-top:10px;">
          <div class="slabel" id="hf-files-label">Files</div>
          <div id="hf-file-list" style="max-height:150px;overflow-y:auto;"></div>
          <button class="btn-primary" onclick="downloadSelected()" style="width:100%;margin-top:8px;">Download Selected</button>
        </div>
      </div></div>
    </div>
  </div>
</div>

<script>
function toggleTheme(){
  const html=document.documentElement;
  const isAmoled=html.getAttribute('data-theme')==='amoled';
  if(isAmoled){html.removeAttribute('data-theme');localStorage.setItem('theme','light');}
  else{html.setAttribute('data-theme','amoled');localStorage.setItem('theme','amoled');}
  updateThemeIcon();
}
function updateThemeIcon(){
  const isAmoled=document.documentElement.getAttribute('data-theme')==='amoled';
  document.getElementById('theme-icon').innerHTML=isAmoled?'&#9788;':'&#9790;';
  document.getElementById('theme-label').textContent=isAmoled?'Light':'Dark';
}
(function(){const t=localStorage.getItem('theme');if(t==='amoled')document.documentElement.setAttribute('data-theme','amoled');updateThemeIcon();})();

function switchTab(name){
  ['images','models'].forEach(t=>{
    const btn=document.getElementById('tb-'+t);
    const panel=document.getElementById('panel-'+t);
    if(btn&&panel){btn.classList.toggle('active',t===name);panel.classList.toggle('active',t===name);}
  });
  if(name==='models'){loadModels();detectHardware();}
}

const I={system_prompt:'',trigger:'',skip_existing:true};
function itog(key){
  I[key]=!I[key];
  const t=document.getElementById('itr-'+key);
  if(t)I[key]?t.classList.add('on'):t.classList.remove('on');
}

let _logInterval=null,_lastLen=0;
let _batchState='idle';

function updateBatchButton(){
  const btn=document.getElementById('i_go-btn');
  if(_batchState==='idle'){
    btn.textContent='Start Captioning';
    btn.className='btn-primary';
    btn.style.width='100%';
    btn.onclick=startImages;
    btn.disabled=false;
  }else if(_batchState==='running'){
    btn.textContent='Stop';
    btn.className='btn-danger';
    btn.style.width='100%';
    btn.onclick=stopBatch;
    btn.disabled=false;
  }else if(_batchState==='stopped'){
    btn.textContent='Resume';
    btn.className='btn-success';
    btn.style.width='100%';
    btn.onclick=resumeBatch;
    btn.disabled=false;
  }
}

function _startBatch(cfg){
  _batchState='running';
  updateBatchButton();
  document.getElementById('i_log-box').textContent='Starting...';
  _lastLen=0;
  fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})
  .then(r=>r.json()).then(d=>{
    if(d.ok){_logInterval=setInterval(()=>pollLog(),800);}
    else{document.getElementById('i_log-box').textContent=d.error;_batchState='idle';updateBatchButton();}
  });
}

function startImages(){
  const cfg={
    ...I,
    input_folder:document.getElementById('i_input_folder').value.trim(),
    output_folder:document.getElementById('i_output_folder').value.trim(),
    trigger:document.getElementById('i_trigger').value.trim(),
    system_prompt:document.getElementById('i_system_prompt').value.trim(),
    mode:'image',num_workers:1
  };
  if(!cfg.input_folder){alert('Set input folder first');return;}
  _startBatch(cfg);
}

function stopBatch(){
  fetch('/stop',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.ok){
      if(_logInterval)clearInterval(_logInterval);
      _batchState='stopped';
      updateBatchButton();
      pollLog();
    }
  });
}

function resumeBatch(){
  const cfg={
    ...I,
    input_folder:document.getElementById('i_input_folder').value.trim(),
    output_folder:document.getElementById('i_output_folder').value.trim(),
    trigger:document.getElementById('i_trigger').value.trim(),
    system_prompt:document.getElementById('i_system_prompt').value.trim(),
    mode:'image',num_workers:1
  };
  _batchState='running';
  updateBatchButton();
  document.getElementById('i_log-box').textContent='Resuming...';
  _lastLen=0;
  fetch('/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})
  .then(r=>r.json()).then(d=>{
    if(d.ok){_logInterval=setInterval(()=>pollLog(),800);}
    else{document.getElementById('i_log-box').textContent=d.error;_batchState='idle';updateBatchButton();}
  });
}

async function testSingle(){
  const logBox=document.getElementById('i_log-box');
  const folder=document.getElementById('i_input_folder').value.trim();
  if(!folder){alert('Please provide an input folder first');return;}
  logBox.textContent='Testing first file... please wait...';
  try{
    const res=await fetch('/test_single',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({folder,cfg:{...I,input_folder:folder,output_folder:folder,
        system_prompt:document.getElementById('i_system_prompt').value.trim(),
        trigger:document.getElementById('i_trigger').value.trim()}})});
    const data=await res.json();
    if(data.ok){
      logBox.textContent='TEST RESULT:\n\n'+data.caption;
      if(data.preview){
        const c=document.getElementById('i_progress-container');if(c)c.style.display='block';
        const p=document.getElementById('i_live-preview');const img=document.getElementById('i_preview-img');
        if(p&&img){p.style.display='block';img.src='data:image/jpeg;base64,'+data.preview;}
      }
    }else{logBox.textContent='TEST FAILED:\n'+data.error;}
  }catch(e){logBox.textContent='NETWORK ERROR: '+e;}
}

function pollLog(){
  fetch('/log?from='+_lastLen).then(r=>r.json()).then(d=>{
    const box=document.getElementById('i_log-box');
    if(d.lines.length){if(_lastLen===0)box.textContent='';box.textContent+=d.lines.join('\n')+'\n';box.scrollTop=box.scrollHeight;_lastLen+=d.lines.length;}
    if(d.done){
      clearInterval(_logInterval);
      _batchState=d.has_checkpoint?'stopped':'idle';
      updateBatchButton();
      document.getElementById('i_progress-container').style.display='none';
    }
    _updateDot(d.server_running);
  });
  fetch('/progress').then(r=>r.json()).then(p=>{
    const c=document.getElementById('i_progress-container');
    if(c){
      c.style.display=(p.total>0&&p.current<p.total)?'block':'none';
      const f=document.getElementById('i_progress-fill');const t=document.getElementById('i_progress-text');
      if(p.total>0){f.style.width=(p.current/p.total*100)+'%';t.textContent='Processed '+p.current+'/'+p.total+' \u00b7 ETA: '+p.eta;}
      const pv=document.getElementById('i_live-preview');const img=document.getElementById('i_preview-img');
      if(pv&&p.current_image){pv.style.display='block';img.src='data:image/jpeg;base64,'+p.current_image;}
      else if(pv){pv.style.display='none';}
    }
  }).catch(e=>console.log("Progress poll error:",e));
}

function _updateDot(running){
  const d=document.getElementById('i_status-dot');const t=document.getElementById('i_status-text');
  if(d)d.className='status-dot'+(running?' green':'');
  if(t)t.textContent='llama-server: '+(running?'running':'stopped');
}
setInterval(()=>{fetch('/status').then(r=>r.json()).then(d=>_updateDot(d.running));},3000);

// ─────────────────────────────────────────────────────────────────────────────
// LLAMA.CPP INSTALLER JS
// ─────────────────────────────────────────────────────────────────────────────
async function detectHardware(){
  try{
    const res = await fetch('/installer/detect_hardware');
    const data = await res.json();
    if(data.ok){
      const hw = data.hardware;
      let html = `<strong>${hw.os_name}</strong> &middot; <strong>${hw.arch_name}</strong>`;
      if(hw.gpu_vendor !== 'None'){
        html += ` &middot; <strong>${hw.gpu_vendor}</strong>: ${hw.gpu_name}`;
        if(hw.cuda_version) html += ` (CUDA ${hw.cuda_version})`;
      } else {
        html += ` &middot; CPU only`;
      }
      document.getElementById('hw-info').innerHTML = html;

      if(data.installed){
        document.getElementById('install-status').innerHTML = '<span style="color:var(--success);">&#10003; llama-server installed</span>';
        document.getElementById('btn-auto-install').textContent = 'Reinstall Binary';
      } else {
        document.getElementById('install-status').innerHTML = '<span style="color:var(--danger);">&#10007; llama-server not found</span>';
      }
    }
  } catch(e){
    document.getElementById('hw-info').textContent = 'Error detecting hardware: ' + e;
  }
}

async function autoInstallBinary(){
  const btn = document.getElementById('btn-auto-install');
  btn.disabled = true; btn.textContent = 'Installing...';
  document.getElementById('install-status').innerHTML = '<span style="color:var(--accent);">Detecting hardware and downloading best binary...</span>';
  try{
    const res = await fetch('/installer/auto_install', {method: 'POST'});
    const data = await res.json();
    if(data.ok){
      document.getElementById('install-status').innerHTML = '<span style="color:var(--success);">&#10003; ' + data.message + '</span>';
      btn.textContent = 'Reinstall Binary';
      detectHardware();
    } else {
      document.getElementById('install-status').innerHTML = '<span style="color:var(--danger);">&#10007; ' + data.error + '</span>';
    }
  } catch(e){
    document.getElementById('install-status').innerHTML = '<span style="color:var(--danger);">&#10007; Error: ' + e + '</span>';
  } finally {
    btn.disabled = false; btn.textContent = btn.textContent === 'Installing...' ? 'Auto Download & Install Best Binary' : btn.textContent;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// MODEL MANAGEMENT JS
// ─────────────────────────────────────────────────────────────────────────────
async function loadModels(){
  const ms=document.getElementById('model-select');const ps=document.getElementById('mmproj-select');
  const sd=document.getElementById('model-status-display');
  try{
    const res=await fetch('/models');const data=await res.json();
    ms.innerHTML='<option value="">-- Select GGUF --</option>';
    data.available.gguf.forEach(m=>{const o=document.createElement('option');o.value=m.path;o.textContent=m.name;if(m.path===data.current.model)o.selected=true;ms.appendChild(o);});
    ps.innerHTML='<option value="">-- Select MMProj --</option>';
    data.available.mmproj.forEach(m=>{const o=document.createElement('option');o.value=m.path;o.textContent=m.name;if(m.path===data.current.mmproj)o.selected=true;ps.appendChild(o);});
    sd.innerHTML='Server: '+(data.current.is_running?'Running':'Stopped')+'<br>Model: '+(data.current.model?data.current.model.split('/').pop():'None')+'<br>MMProj: '+(data.current.mmproj?data.current.mmproj.split('/').pop():'None');
    document.getElementById('model-val-dot').className='val-dot';document.getElementById('mmproj-val-dot').className='val-dot';
    document.getElementById('model-val-text').textContent='Waiting for validation';
    document.getElementById('mmproj-val-text').textContent='Waiting for validation';
    document.getElementById('apply-model-btn').disabled=true;
  }catch(e){sd.textContent="Error loading model list.";console.error(e);}
}

async function validateFiles(){
  const mp=document.getElementById('model-select').value;const pp=document.getElementById('mmproj-select').value;
  if(!mp){alert("Please select a model first.");return;}
  const md=document.getElementById('model-val-dot');const mt=document.getElementById('model-val-text');
  const pd=document.getElementById('mmproj-val-dot');const pt=document.getElementById('mmproj-val-text');
  const btn=document.getElementById('apply-model-btn');
  mt.textContent="Checking...";pt.textContent="Checking...";md.className="val-dot";pd.className="val-dot";
  try{
    const res=await fetch('/validate_model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:mp,mmproj:pp})});
    const data=await res.json();
    md.className="val-dot "+(data.model_valid?"valid":"invalid");mt.textContent=data.model_valid?"Valid":"Invalid or Missing";
    if(data.mmproj_valid){pd.className="val-dot valid";pt.textContent="Valid";}
    else if(!pp){pd.className="val-dot valid";pt.textContent="Empty (Text Only)";}
    else{pd.className="val-dot invalid";pt.textContent="Invalid or Missing";}
    btn.disabled=!(data.model_valid&&(data.mmproj_valid||!pp));
    btn.textContent=btn.disabled?"Files Invalid":"Apply & Restart Server";
  }catch(e){alert("Validation error: "+e);md.className="val-dot invalid";pd.className="val-dot invalid";btn.disabled=true;}
}

async function applyModel(){
  const mp=document.getElementById('model-select').value;const pp=document.getElementById('mmproj-select').value;
  const btn=document.getElementById('apply-model-btn');btn.disabled=true;btn.textContent="Restarting...";
  try{
    const res=await fetch('/set_model',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:mp,mmproj:pp})});
    const data=await res.json();
    if(data.ok){alert("Server restarted successfully!");loadModels();}
    else{alert("Failed to restart: "+data.error);}
  }catch(e){alert("Network error: "+e);}
  finally{btn.disabled=false;btn.textContent="Apply & Restart Server";}
}

// ─────────────────────────────────────────────────────────────────────────────
// MODEL DOWNLOADER JS
// ─────────────────────────────────────────────────────────────────────────────
let _selectedRepo=null;
let _downloadPollInterval=null;

async function searchHF(){
  const query=document.getElementById('hf-search-input').value.trim();
  if(!query)return;
  const resultsDiv=document.getElementById('hf-results');
  resultsDiv.innerHTML='<div style="color:var(--muted-3);font-size:11px;padding:8px;">Searching...</div>';
  try{
    const res=await fetch('/hf/search?q='+encodeURIComponent(query));
    const data=await res.json();
    if(!data.ok||!data.results.length){
      resultsDiv.innerHTML='<div style="color:var(--muted-3);font-size:11px;padding:8px;">No results found.</div>';
      return;
    }
    let html='';
    data.results.forEach(repo=>{
      html+=`<div class="hf-repo" onclick="selectRepo('${repo.id}',this)">`;
      html+=`<strong>${repo.id}</strong>`;
      html+=`<span>${repo.downloads} downloads</span>`;
      html+=`</div>`;
    });
    resultsDiv.innerHTML=html;
    document.getElementById('hf-files').style.display='none';
  }catch(e){
    resultsDiv.innerHTML='<div style="color:var(--danger);font-size:11px;padding:8px;">Search failed: '+e+'</div>';
  }
}

async function selectRepo(repoId,el){
  _selectedRepo=repoId;
  document.querySelectorAll('.hf-repo').forEach(r=>r.classList.remove('selected'));
  if(el)el.classList.add('selected');
  const filesDiv=document.getElementById('hf-files');
  const fileList=document.getElementById('hf-file-list');
  const label=document.getElementById('hf-files-label');
  label.textContent='Files in '+repoId;
  fileList.innerHTML='<div style="color:var(--muted-3);font-size:11px;padding:8px;">Loading files...</div>';
  filesDiv.style.display='block';
  try{
    const res=await fetch('/hf/files?repo='+encodeURIComponent(repoId));
    const data=await res.json();
    if(!data.ok||!data.files.length){
      fileList.innerHTML='<div style="color:var(--muted-3);font-size:11px;padding:8px;">No GGUF files found.</div>';
      return;
    }
    let html='';
    data.files.forEach(file=>{
      html+=`<label class="hf-file">`;
      html+=`<input type="checkbox" value="${file.name}">`;
      html+=`<span>${file.name}</span>`;
      html+=`</label>`;
    });
    fileList.innerHTML=html;
  }catch(e){
    fileList.innerHTML='<div style="color:var(--danger);font-size:11px;padding:8px;">Failed to load files: '+e+'</div>';
  }
}

async function downloadSelected(){
  if(!_selectedRepo){alert('Please select a repository first.');return;}
  const checkboxes=document.querySelectorAll('#hf-file-list input[type="checkbox"]:checked');
  const filenames=Array.from(checkboxes).map(cb=>cb.value);
  if(!filenames.length){alert('Please select at least one file to download.');return;}
  const ariaRes=await fetch('/check_aria2c');const ariaData=await ariaRes.json();
  if(!ariaData.ok){alert('aria2c is not installed. Please install it: sudo apt install aria2');return;}
  try{
    const res=await fetch('/hf/download',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({repo_id:_selectedRepo,filenames:filenames})});
    const data=await res.json();
    if(data.ok){startDownloadProgressPolling();}
    else{alert('Error: '+data.error);}
  }catch(e){alert('Network error: '+e);}
}

async function downloadPreset(preset){
  const ariaRes=await fetch('/check_aria2c');const ariaData=await ariaRes.json();
  if(!ariaData.ok){alert('aria2c is not installed. Please install it: sudo apt install aria2');return;}
  try{
    const res=await fetch('/hf/download_preset',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({preset:preset})});
    const data=await res.json();
    if(data.ok){startDownloadProgressPolling();}
    else{alert('Error: '+data.error);}
  }catch(e){alert('Network error: '+e);}
}

function startDownloadProgressPolling(){
  const statusBox=document.getElementById('download-status-box');
  statusBox.style.display='block';
  if(_downloadPollInterval)clearInterval(_downloadPollInterval);
  _downloadPollInterval=setInterval(async()=>{
    try{
      const res=await fetch('/hf/download_progress');
      const data=await res.json();
      const textEl=document.getElementById('download-status-text');
      const fillEl=document.getElementById('download-progress-fill');
      if(data.active){
        textEl.textContent=data.message||('Downloading '+data.current_file+'...');
        fillEl.style.width=(data.files_total>0?(data.files_done/data.files_total*100):0)+'%';
      }else{
        clearInterval(_downloadPollInterval);
        fillEl.style.width=data.error?'0%':'100%';
        textEl.textContent=data.error?('Error: '+data.error):(data.message||'Download complete!');
        if(!data.error){
          setTimeout(()=>{statusBox.style.display='none';loadModels();},3000);
        }
      }
    }catch(e){}
  },1000);
}

async function initPrompts(){
  try{const res=await fetch('/prompts');const data=await res.json();
    if(data.ok){const s=document.getElementById('i_training_type');data.prompts.forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p;s.appendChild(o});}
  }catch(e){console.error("Error loading prompts:",e);}
}

async function onTrainingTypeChange(el){
  const name=el.value;if(!name)return;
  try{const res=await fetch('/prompt_content?name='+encodeURIComponent(name));const data=await res.json();
    if(data.ok)document.getElementById('i_system_prompt').value=data.content;
  }catch(e){console.error("Error loading prompt content:",e);}
}

window.onload=()=>{initPrompts();};
</script>
</body>
</html>"""

_batch_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_errors.log")

@app.route("/prompts", methods=["GET"])
def get_prompts():
    prompts_dir = "prompts"
    if not os.path.exists(prompts_dir):
        return jsonify({"ok": False, "error": "Prompts directory not found"}), 404
    try:
        files = [os.path.splitext(f)[0] for f in os.listdir(prompts_dir) if f.endswith(".txt")]
        return jsonify({"ok": True, "prompts": sorted(files)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/prompt_content", methods=["GET"])
def get_prompt_content():
    prompt_name = request.args.get("name")
    if not prompt_name:
        return jsonify({"ok": False, "error": "Missing prompt name"}), 400
    prompts_dir = "prompts"
    file_path = os.path.join(prompts_dir, f"{prompt_name}.txt")
    if not os.path.exists(file_path):
        return jsonify({"ok": False, "error": "Prompt file not found"}), 404
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        return jsonify({"ok": True, "content": content})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/start", methods=["POST"])
def start():
    global state
    if state.batch_running:
        return jsonify({"ok": False, "error": "batch already running"})
    with state.log_lock:
        state.log_lines = []
    state.batch_stop = False
    state.batch_running = True
    cfg = request.get_json()
    if not cfg:
        return jsonify({"ok": False, "error": "invalid JSON"})

    out = cfg.get("output_folder") or cfg["input_folder"]

    def run():
        global state
        try:
            _run_batch(cfg, resume=False)
        except Exception as e:
            err = f"FATAL batch error:\n{traceback.format_exc()}"
            print(err, flush=True)
            with open(_batch_log_file, "a", encoding="utf-8") as f:
                f.write(err + "\n")
            with state.log_lock:
                state.log_lines.append(err)
        finally:
            state.batch_running = False
            with state.log_lock:
                state.log_lines.append("\n[Batch process finished]")

    state.batch_thread = threading.Thread(target=run, daemon=True)
    state.batch_thread.start()
    return jsonify({"ok": True})

@app.route("/resume", methods=["POST"])
def resume():
    global state
    if state.batch_running:
        return jsonify({"ok": False, "error": "batch already running"})
    with state.log_lock:
        state.log_lines = []
    state.batch_stop = False
    state.batch_running = True
    cfg = request.get_json()
    if not cfg:
        return jsonify({"ok": False, "error": "invalid JSON"})

    def run():
        global state
        try:
            _run_batch(cfg, resume=True)
        except Exception as e:
            err = f"FATAL batch error:\n{traceback.format_exc()}"
            print(err, flush=True)
            with open(_batch_log_file, "a", encoding="utf-8") as f:
                f.write(err + "\n")
            with state.log_lock:
                state.log_lines.append(err)
        finally:
            state.batch_running = False
            with state.log_lock:
                state.log_lines.append("\n[Batch process finished]")

    state.batch_thread = threading.Thread(target=run, daemon=True)
    state.batch_thread.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    global state
    state.batch_stop = True
    _log("Stop requested. Finishing current item and saving checkpoint...")
    return jsonify({"ok": True})

@app.route("/log")
def log():
    from_idx = int(request.args.get("from", 0))
    with state.log_lock:
        lines = state.log_lines[from_idx:]
    output_folder = state.batch_config.get("output_folder", "") or state.batch_config.get("input_folder", "")
    has_checkpoint = False
    if output_folder:
        cp = _load_checkpoint(output_folder)
        has_checkpoint = cp is not None and len(cp.get("remaining_images", [])) > 0
    return jsonify({"lines": lines, "done": not state.batch_running, "server_running": _llama_running(), "has_checkpoint": has_checkpoint})

@app.route("/status")
def status():
    return jsonify({"running": _llama_running(), "batch": state.batch_running})

@app.route("/progress")
def progress():
    return jsonify({**state.batch_progress, "current_image": state.current_image_b64})

@app.route("/test_single", methods=["POST"])
def test_single():
    data = request.get_json()
    folder = data.get("folder", "")
    cfg = data.get("cfg", {})

    if not folder or not os.path.isdir(folder):
        return jsonify({"ok": False, "error": f"Input folder not found: {folder}"})

    if not _llama_running():
        return jsonify({"ok": False, "error": "llama-server is not running. Please configure and start it from the Models tab first."})

    IMAGE_EXTS = (".jpg",".jpeg",".png",".webp",".bmp",".tiff",".tif")
    all_files = sorted(os.listdir(folder))
    matches = [f for f in all_files if f.lower().endswith(IMAGE_EXTS)]

    if not matches:
        return jsonify({"ok": False, "error": "No compatible image files found in folder."})

    file_path = os.path.join(folder, matches[0])

    system_prompt = _safe_get(cfg, "system_prompt", "")
    trigger       = _safe_get(cfg, "trigger", "")

    try:
        cap, err = _caption_one(file_path, system_prompt, trigger)
        if err:
            return jsonify({"ok": False, "error": err})
        return jsonify({"ok": True, "caption": cap, "preview": state.current_image_b64})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/models", methods=["GET"])
def get_models():
    model_dir = state.config.get("model_dir", "./models")
    available = _scan_available_models(model_dir=model_dir)
    return jsonify({
        "available": available,
        "current": {
            "model": state.config["model_path"],
            "mmproj": state.config["mmproj_path"],
            "model_dir": model_dir,
            "is_running": state.is_server_running
        }
    })

@app.route("/validate_model", methods=["POST"])
def validate_model():
    data = request.get_json()
    model_path = data.get("model", "")
    mmproj_path = data.get("mmproj", "")

    model_valid = False
    mmproj_valid = False

    if model_path and os.path.exists(model_path) and os.path.isfile(model_path):
        model_valid = True

    if not mmproj_path:
        mmproj_valid = True
    elif mmproj_path and os.path.exists(mmproj_path) and os.path.isfile(mmproj_path):
        mmproj_valid = True

    return jsonify({
        "model_valid": model_valid,
        "mmproj_valid": mmproj_valid
    })

@app.route("/set_model", methods=["POST"])
def set_model():
    data = request.get_json()
    new_model = data.get("model")
    new_mmproj = data.get("mmproj", "")

    if not new_model:
        return jsonify({"ok": False, "error": "Missing model path"}), 400

    if not os.path.exists(new_model):
        return jsonify({"ok": False, "error": f"Model file not found: {new_model}"}), 400

    if new_mmproj and not os.path.exists(new_mmproj):
        return jsonify({"ok": False, "error": f"MMProj file not found: {new_mmproj}"}), 400

    state.config["model_path"] = new_model
    state.config["mmproj_path"] = new_mmproj
    state.save_config()

    if state.batch_running:
        return jsonify({"ok": False, "error": "Cannot switch model while batch is running"}), 409

    status_msg = _start_llama_server()

    if "failed" in status_msg or "error" in status_msg.lower():
        return jsonify({"ok": False, "error": status_msg}), 500

    return jsonify({"ok": True, "message": f"Server restarted with {os.path.basename(new_model)}"})

# ─────────────────────────────────────────────────────────────────────────────
# LLAMA.CPP INSTALLER API
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/installer/detect_hardware", methods=["GET"])
def detect_hardware():
    hw = _detect_hardware()
    bin_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    installed = False
    installed_path = None

    resolved = _resolve_llama_server_bin()
    if os.path.isfile(resolved):
        installed = True
        installed_path = resolved
    else:
        for root, dirs, files in os.walk(LLAMA_CPP_INSTALL_DIR):
            for f in files:
                if f == bin_name or f == "llama-server.exe":
                    installed_path = os.path.join(root, f)
                    installed = True
                    break
            if installed:
                break

    return jsonify({
        "ok": True,
        "hardware": hw,
        "installed": installed,
        "installed_path": installed_path,
        "install_dir": LLAMA_CPP_INSTALL_DIR
    })

@app.route("/installer/auto_install", methods=["POST"])
def auto_install():
    """Auto-detect hardware and install best binary."""
    hw = _detect_hardware()

    try:
        r = requests.get(GITHUB_API_URL, timeout=15)
        r.raise_for_status()
        release = r.json()
        assets = release.get("assets", [])
        best_asset, pattern = _match_release_asset(assets, hw)

        if not best_asset:
            return jsonify({"ok": False, "error": "No matching binary found for your hardware"})

        _log(f"  Auto-installing: {best_asset['name']} (matched: {pattern})")
        success, result = _install_prebuilt_binary(best_asset, hw)

        if success:
            return jsonify({"ok": True, "path": result, "message": f"Installed {best_asset['name']}"})
        else:
            return jsonify({"ok": False, "error": result})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# HUGGINGFACE MODEL DOWNLOAD API
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/check_aria2c", methods=["GET"])
def check_aria2c():
    return jsonify({"ok": shutil.which("aria2c") is not None})

@app.route("/hf/search", methods=["GET"])
def hf_search():
    query = request.args.get("q", "")
    if not query:
        return jsonify({"ok": False, "error": "Missing query"}), 400

    try:
        r = requests.get(
            "https://huggingface.co/api/models",
            params={"search": query + " gguf", "sort": "downloads", "direction": "-1", "limit": 20},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()

        results = []
        for model in data:
            model_id = model.get("id", "")
            if not model_id:
                continue
            results.append({
                "id": model_id,
                "downloads": model.get("downloads", 0),
            })

        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/hf/files", methods=["GET"])
def hf_files():
    repo_id = request.args.get("repo", "")
    if not repo_id:
        return jsonify({"ok": False, "error": "Missing repo"}), 400

    try:
        r = requests.get(f"https://huggingface.co/api/models/{repo_id}", timeout=15)
        r.raise_for_status()
        data = r.json()

        siblings = data.get("siblings", [])
        files = []
        for s in siblings:
            filename = s.get("rfilename", "")
            if filename.endswith(".gguf"):
                files.append({"name": filename})

        return jsonify({"ok": True, "files": files})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/hf/download_preset", methods=["POST"])
def download_preset():
    data = request.get_json()
    preset = data.get("preset")

    if preset not in PRESET_DOWNLOADS:
        return jsonify({"ok": False, "error": f"Unknown preset: {preset}"}), 400

    if state.download_progress.get("active"):
        return jsonify({"ok": False, "error": "Download already in progress"}), 409

    preset_data = PRESET_DOWNLOADS[preset]
    repo_id = preset_data["repo"]
    filenames = preset_data["files"]

    _log(f"  Starting download: {preset_data['label']} from {repo_id}")

    thread = threading.Thread(target=_download_hf_files, args=(repo_id, filenames), daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": f"Downloading {preset_data['label']}"})

@app.route("/hf/download", methods=["POST"])
def download_hf_files():
    data = request.get_json()
    repo_id = data.get("repo_id")
    filenames = data.get("filenames", [])

    if not repo_id or not filenames:
        return jsonify({"ok": False, "error": "Missing repo_id or filenames"}), 400

    if state.download_progress.get("active"):
        return jsonify({"ok": False, "error": "Download already in progress"}), 409

    _log(f"  Starting download: {len(filenames)} files from {repo_id}")

    thread = threading.Thread(target=_download_hf_files, args=(repo_id, filenames), daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": f"Downloading {len(filenames)} files from {repo_id}"})

@app.route("/hf/download_progress", methods=["GET"])
def download_progress():
    return jsonify(state.download_progress)

if __name__ == "__main__":
    print("\n" + "─"*52)
    print("  Dataset Captioner — Image Captioning Edition")
    print(f"  model  : {state.config['model_path']}")
    print("─"*52)
    print("  opening http://127.0.0.1:7860\n")
    try:
        threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    except Exception:
        print("  [Note] Headless environment detected. Open http://127.0.0.1:7860 manually.")
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
