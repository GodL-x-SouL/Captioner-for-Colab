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
import zipfile
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template_string, request
from PIL import Image as PILImage

# ─────────────────────────────────────────────────────────────────────────────
# EARLY ENVIRONMENT & DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def _diag(msg):
    print(f"[DIAG] {msg}", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLAB DETECTION & DEFAULT PATHS
# ─────────────────────────────────────────────────────────────────────────────
def _is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IS_COLAB = _is_colab()

if IS_COLAB:
    DEFAULT_INPUT_FOLDER = "/content/Captioner-for-Colab/input"
    DEFAULT_OUTPUT_FOLDER = "/content/Captioner-for-Colab/output"
else:
    DEFAULT_INPUT_FOLDER = os.path.join(SCRIPT_DIR, "input")
    DEFAULT_OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "output")

os.makedirs(DEFAULT_INPUT_FOLDER, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)

def _check_llama_binary():
    """Check if llama-server is accessible."""
    raw = "llama-server"
    # Try direct PATH lookup
    if shutil.which(raw):
        return True
    # On Windows, try with .exe
    if sys.platform == "win32" and shutil.which(raw + ".exe"):
        return True
    return False

if _check_llama_binary():
    _diag("llama-server found in PATH")
else:
    _diag("llama-server not in PATH. Set LLAMA_SERVER_BIN env var, add llama-server to PATH, or set the full path in Models tab.")

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
            "model_dir": MODEL_DIR,
            "mmproj_dir": MODEL_DIR,
            "port": int(os.environ.get("LLAMA_PORT", 18080)),
            "ctx_size": 16384,
            "gpu_layers": 99,
            "input_folder": DEFAULT_INPUT_FOLDER,
            "output_folder": DEFAULT_OUTPUT_FOLDER
        }
        self.stats_lock = threading.Lock()
        self._load_config()
        self.config["model_dir"] = MODEL_DIR
        self.config["mmproj_dir"] = MODEL_DIR
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
        self.batch_progress = {"current": 0, "total": 0, "start_time": None, "eta": "Calculating..."}
        self.current_image_b64 = None
        
        self.completed_files = set()
        self.downloader_running = False
        self.download_progress = {
            "active": False,
            "filename": "",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "speed_bps": 0,
            "eta_seconds": 0,
            "percent": 0,
            "status": "idle"
        }

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
def _scan_available_models(model_dir=None, mmproj_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    if mmproj_dir is None:
        mmproj_dir = MODEL_DIR
    ggufs = []
    mmprojs = []

    if os.path.exists(model_dir):
        try:
            for f in sorted(os.listdir(model_dir)):
                fp = os.path.join(model_dir, f)
                if os.path.isfile(fp) and f.lower().endswith(".gguf") and not ('mmproj' in f.lower() or 'projector' in f.lower()):
                    ggufs.append({"name": f, "path": fp})
        except Exception as e:
            _log(f"Error scanning model_dir {model_dir}: {e}")

    if os.path.exists(mmproj_dir):
        try:
            for f in sorted(os.listdir(mmproj_dir)):
                fp = os.path.join(mmproj_dir, f)
                if not os.path.isfile(fp):
                    continue
                lower_f = f.lower()
                if lower_f.endswith((".gguf", ".safetensors", ".pt")) and ('mmproj' in lower_f or 'projector' in lower_f):
                    mmprojs.append({"name": f, "path": fp})
        except Exception as e:
            _log(f"Error scanning mmproj_dir {mmproj_dir}: {e}")

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

    # If it's already an absolute path to a file that exists, use it
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw

    # Check shutil.which first (searches PATH)
    found = shutil.which(raw)
    if found:
        return found

    # On Windows, try appending .exe if not already there
    if sys.platform == "win32":
        if not raw.lower().endswith(".exe"):
            found = shutil.which(raw + ".exe")
            if found:
                return found

    # Search LLAMA_CPP_INSTALL_DIR recursively for llama-server
    bin_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    if os.path.isdir(LLAMA_CPP_INSTALL_DIR):
        for root, dirs, files in os.walk(LLAMA_CPP_INSTALL_DIR):
            for f in files:
                # Match exact name or starts with llama-server (handles renamed binaries)
                if f == bin_name or (f.startswith("llama-server") and os.access(os.path.join(root, f), os.X_OK)):
                    found_bin = os.path.join(root, f)
                    # Update config so we don't need to search again
                    state.config["llama_server_bin"] = found_bin
                    state.save_config()
                    _log(f"Resolved llama-server from install dir: {found_bin}")
                    return found_bin

    # Try common locations relative to the script
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

    return raw  # Return as-is, Popen will raise the error


def _auto_setup_llama_binaries():
    """Automatically downloads and configures the optimized llama-server binary for Colab/T4."""
    bin_name = "llama-server"
    if sys.platform == "win32":
        bin_name += ".exe"

    # All binaries go into LLAMA_CPP_INSTALL_DIR so they live alongside llama.cpp source
    bin_dir = LLAMA_CPP_INSTALL_DIR
    bin_path = os.path.join(bin_dir, bin_name)

    # Scan install dir recursively for any existing llama-server binary
    if os.path.isdir(bin_dir):
        for root, dirs, files in os.walk(bin_dir):
            for f in files:
                if f == bin_name or (f.startswith("llama-server") and os.access(os.path.join(root, f), os.X_OK)):
                    found_bin = os.path.join(root, f)
                    state.config["llama_server_bin"] = found_bin
                    state.save_config()
                    _log(f"Auto-setup: Found existing llama-server binary at {found_bin}")
                    return True

    if os.path.exists(bin_path):
        state.config["llama_server_bin"] = bin_path
        state.save_config()
        _log(f"Auto-setup: Found existing llama-server binary at {bin_path}")
        return True

    _log("Auto-setup: Optimized T4 llama-server binary not found. Initiating automatic setup...")
    url = "https://github.com/GodL-x-SouL/Captioner-for-Colab/releases/download/v1.0/llama-b9763-bin-linux-cuda-x64.tar.gz"
    os.makedirs(bin_dir, exist_ok=True)
    tmp_tar = os.path.join(tempfile.gettempdir(), "llama_binaries.tar.gz")

    try:
        _log("Auto-setup: downloading llama.cpp binaries...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(tmp_tar, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (1024 * 1024) == 0:
                    pct = int(downloaded * 100 / total)
                    _log(f"Auto-setup: downloaded {pct}%...")

        import tarfile
        # Tarball contains a llama.cpp/ folder — extract to temp, then flatten into bin_dir
        tmp_extract = os.path.join(tempfile.gettempdir(), "llama_extract_tmp")
        if os.path.exists(tmp_extract):
            shutil.rmtree(tmp_extract)
        os.makedirs(tmp_extract, exist_ok=True)
        with tarfile.open(tmp_tar, 'r:gz') as tf:
            tf.extractall(tmp_extract, filter='data')
        # Find the inner llama.cpp/ folder and move its contents to bin_dir
        inner_cpp = os.path.join(tmp_extract, "llama.cpp")
        if os.path.isdir(inner_cpp):
            for item in os.listdir(inner_cpp):
                src = os.path.join(inner_cpp, item)
                dst = os.path.join(bin_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            shutil.rmtree(tmp_extract, ignore_errors=True)
        else:
            # Fallback: contents extracted directly into tmp_extract, move them
            for item in os.listdir(tmp_extract):
                src = os.path.join(tmp_extract, item)
                dst = os.path.join(bin_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            shutil.rmtree(tmp_extract, ignore_errors=True)

        if os.path.exists(bin_path):
            if sys.platform != "win32":
                os.chmod(bin_path, 0o755)
            state.config["llama_server_bin"] = bin_path
            state.save_config()
            _log(f"Auto-setup: Successfully installed and configured llama-server binary at {bin_path}")
            return True
        else:
            _log(f"Auto-setup error: llama-server binary not found at {bin_path} after extraction.")
            return False
    except Exception as e:
        _log(f"Auto-setup failed: {e}")
        return False
    finally:
        if os.path.exists(tmp_tar):
            try:
                os.remove(tmp_tar)
            except:
                pass


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

    # Resolve the binary path — run auto-setup first if needed
    resolved_bin = _resolve_llama_server_bin()
    if not os.path.isfile(resolved_bin):
        # Try auto-setup as a last resort
        _auto_setup_llama_binaries()
        resolved_bin = _resolve_llama_server_bin()
    if not os.path.isfile(resolved_bin):
        return (
            f"llama-server binary not found: '{state.config.get('llama_server_bin', '')}'\n"
            f"Resolved to: {resolved_bin}\n\n"
            f"Please set the full path to llama-server in the Models tab or add it to your PATH.\n"
            f"You can also set the LLAMA_SERVER_BIN environment variable."
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

        # Set LD_LIBRARY_PATH to include the binary's directory so shared libs are found
        env = os.environ.copy()
        bin_dir = os.path.dirname(os.path.abspath(resolved_bin))
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = bin_dir + (":" + existing_ld if existing_ld else "")

        state.server_process = subprocess.Popen(cmd, stdout=lf, stderr=lf, creationflags=creation_flags, env=env)
    except FileNotFoundError as e:
        return (
            f"Cannot start llama-server: {e}\n"
            f"Command: {cmd[0]}\n\n"
            f"Ensure llama-server is installed and accessible.\n"
            f"You can set the full path in config.json under 'llama_server_bin'."
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
LLAMA_CPP_INSTALL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama.cpp")


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
# ZIP FOLDER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _zip_folder_contents(folder_path, output_zip_path=None):
    """Zip all contents of a folder into a zip file.
    
    Args:
        folder_path: Path to the folder to zip
        output_zip_path: Optional path for the zip file. If None, creates zip in parent dir.
    
    Returns:
        (success: bool, result: str) - result is zip_path on success, error message on failure
    """
    if not os.path.isdir(folder_path):
        return False, f"Folder not found: {folder_path}"
    
    if output_zip_path is None:
        parent_dir = os.path.dirname(folder_path)
        folder_name = os.path.basename(folder_path)
        output_zip_path = os.path.join(parent_dir, f"{folder_name}.zip")
    
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)
        
        zip_size = os.path.getsize(output_zip_path)
        return True, output_zip_path
    except Exception as e:
        return False, str(e)

def _zip_input_folder(output_zip_path=None):
    """Zip the input folder contents."""
    return _zip_folder_contents(DEFAULT_INPUT_FOLDER, output_zip_path)

def _zip_output_folder(output_zip_path=None):
    """Zip the output folder contents."""
    return _zip_folder_contents(DEFAULT_OUTPUT_FOLDER, output_zip_path)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

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
def _producer(cfg, q, stats):
    try:
        input_folder  = _safe_get(cfg, "input_folder", "")
        output_folder = _safe_get(cfg, "output_folder", "") or input_folder
        skip_ex       = _safe_get(cfg, "skip_existing", False)

        IMAGE_EXTS = (".jpg",".jpeg",".png",".webp",".bmp",".tiff",".tif")
        all_files  = sorted(os.listdir(input_folder))
        images = [os.path.join(input_folder,f) for f in all_files if f.lower().endswith(IMAGE_EXTS)]

        for img_src in images:
            if state.batch_stop:
                break
            with state.stats_lock:
                if img_src in state.completed_files:
                    continue
            stem = os.path.splitext(os.path.basename(img_src))[0]
            txt = os.path.join(output_folder, stem + ".txt")
            if skip_ex and os.path.exists(txt):
                stats['skipped'] += 1
                continue
            q.put({"type": "image", "path": img_src, "txt": txt})
    finally:
        pass

def _consumer(cfg, q, stats):
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
            txt = item["txt"]
            fname = os.path.basename(src)

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
                    state.completed_files.add(src)
            update_prog()

        except Exception as e:
            _log(f"  unexpected error processing {item.get('path')}: {e}")
            with state.stats_lock:
                stats['errors'] += 1
            update_prog()
        finally:
            q.task_done()

def _run_batch(cfg):
    global state
    state.batch_stop = False
    _log(f"  Batch started: input={_safe_get(cfg,'input_folder','')}")

    input_folder = _safe_get(cfg, "input_folder", "")
    if not os.path.isdir(input_folder):
        _log(f"  input folder not found: {input_folder}")
        return

    output_folder = _safe_get(cfg, "output_folder", "") or input_folder
    os.makedirs(output_folder, exist_ok=True)

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

    all_files = sorted(os.listdir(input_folder))
    IMAGE_EXTS = (".jpg",".jpeg",".png",".webp",".bmp",".tiff",".tif")
    images = [f for f in all_files if f.lower().endswith(IMAGE_EXTS)]

    state.batch_progress["total"] = len(images)
    state.batch_progress["start_time"] = time.time()
    state.batch_progress["current"] = 0

    prod_thread = threading.Thread(target=_producer, args=(cfg, q, stats), daemon=True)
    prod_thread.start()

    num_workers = int(_safe_get(cfg, "num_workers", 1))
    _log(f"  Starting {num_workers} consumer worker(s)...")
    consumers = []
    for i in range(num_workers):
        t = threading.Thread(target=_consumer, args=(cfg, q, stats), daemon=True)
        t.start()
        consumers.append(t)

    prod_thread.join()

    for _ in range(num_workers):
        q.put(None)

    for t in consumers:
        t.join()

    _log(f"\n{'─'*54}")
    _log(f"  done: {stats['done']} captioned, {stats['skipped']} skipped, {stats['errors']} errors")
    _log(f"{'─'*54}")
    state.batch_progress["eta"] = "Done"

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
/* Custom scrollbar matching the overall UI */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--muted-4);
}
* {
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}
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
  font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:0.8px;text-transform:uppercase;
  padding:6px 14px;border:1px solid var(--danger);
  background:transparent;color:var(--danger);cursor:pointer;
  border-radius:var(--radius);transition:all 0.15s;width:100%;
}
.btn-danger:hover{background:var(--danger);color:#fff;}
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
.pill-group{display:flex;gap:4px;flex-wrap:wrap;}
.pill{
  font-family:var(--font-mono);font-size:10px;font-weight:400;
  letter-spacing:0.3px;padding:5px 10px;border:1px solid var(--border);
  border-radius:var(--radius);background:var(--bg-card);color:var(--muted);
  cursor:pointer;transition:all 0.15s;user-select:none;
}
.pill:hover{border-color:var(--accent);color:var(--accent);}
.pill.active{background:var(--accent);border-color:var(--accent);color:#fff;}
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
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
}
.log-box::-webkit-scrollbar{width:6px;}
.log-box::-webkit-scrollbar-track{background:transparent;}
.log-box::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
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
.toast-container{
  position:fixed;top:20px;right:20px;z-index:9999;
  display:flex;flex-direction:column;gap:8px;
}
.toast{
  font-family:var(--font-mono);font-size:11px;font-weight:400;
  padding:10px 16px;border-radius:var(--radius);
  border:1px solid var(--border);background:var(--bg-card);
  color:var(--ink);box-shadow:var(--shadow-md);
  animation:toastIn 0.3s ease;max-width:350px;
  transition:opacity 0.3s,transform 0.3s;
}
.toast.success{border-color:var(--success);color:var(--success);}
.toast.error{border-color:var(--danger);color:var(--danger);}
.toast.info{border-color:var(--accent);color:var(--accent);}
.toast.fade-out{opacity:0;transform:translateX(20px);}
@keyframes toastIn{from{opacity:0;transform:translateX(20px);}to{opacity:1;transform:translateX(0);}}
</style>
</head>
<body>

<div class="toast-container" id="toast-container"></div>

<div class="header-row">
  <div class="title-block">
    <div class="title-row">
      <h1>Dataset Captioner</h1>
    </div>
    <div class="subtitle">Image captioning studio powered by local LLM inference via llama-server.</div>
  </div>
  <div class="toolbar">
    <div class="theme-toggle" onclick="refreshUI()" title="Refresh UI">
      <span class="icon">&#8635;</span>
      <span>Refresh</span>
    </div>
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
        <button class="btn-danger" id="i_stop-btn" onclick="stopBatch()" style="margin-top:6px;width:100%;">Stop</button>
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

      <div class="card"><div class="section">
        <div class="slabel">Zip &amp; Clean Folders</div>
        <button class="btn-primary" style="width:100%;margin-bottom:6px;" onclick="zipFolder('input')">Zip Input Folder</button>
        <button class="btn-primary" style="width:100%;margin-bottom:6px;" onclick="zipFolder('output')">Zip Output Folder</button>
        <button class="btn-danger" style="width:100%;margin-bottom:6px;" onclick="deleteFolder('input')">Delete Input Contents</button>
        <button class="btn-danger" style="width:100%;margin-bottom:6px;" onclick="deleteFolder('output')">Delete Output Contents</button>
        <div class="hint">Zip for download or clear folder contents for Google Colab</div>
      </div></div>
    </div>
  </div>
</div>

<!-- MODELS TAB -->
<div class="tab-panel" id="panel-models">
  <div class="section-title">Model Configuration</div>
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

      <div class="slabel" style="margin-top:12px;">Select MMProj (GGUF/Safetensors)</div>
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
    <div class="card"><div class="section">
      <div class="slabel">Presets</div>
      <div style="display:flex;gap:8px;margin-bottom:12px;">
        <button class="btn-primary" onclick="downloadPreset('qwen')" style="flex:1;">Download Qwen 3 8B VL (Q8_0 + F16 MMProj)</button>
        <button class="btn-primary" onclick="downloadPreset('gemma')" style="flex:1;">Download Gemma 4 12B (Q8_0 + F16 MMProj)</button>
      </div>

      <div class="slabel">Search Hugging Face</div>
      <div style="display:flex;gap:8px;margin-bottom:12px;">
        <input class="path-input" id="hf-search-input" placeholder="Search models, e.g. smolvlm" style="flex:1;" onkeydown="if(event.key==='Enter')searchHF()">
        <button class="btn" onclick="searchHF()">Search</button>
      </div>

      <div id="hf-search-results" style="max-height:200px;overflow-y:auto;margin-bottom:12px;display:none;border:1px solid var(--border);padding:6px;border-radius:var(--radius);background:var(--bg-soft);"></div>
      <div id="hf-files-section" style="display:none;margin-top:12px;">
        <div class="slabel" id="hf-files-title">Files in Repository</div>
        <div id="hf-files-list" style="max-height:200px;overflow-y:auto;margin-bottom:12px;border:1px solid var(--border);padding:6px;border-radius:var(--radius);background:var(--bg-soft);"></div>
      </div>
      
      <div id="download-progress-display" style="font-size:11px;color:var(--muted-2);margin-top:8px;"></div>
      
      <div id="dl-progress-container" style="display:none;margin-top:12px;padding:12px;border:1px solid var(--border);border-radius:var(--radius);background:var(--bg-soft);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="font-family:var(--font-mono);font-size:10px;font-weight:500;color:var(--ink-3);text-transform:uppercase;letter-spacing:0.5px;" id="dl-filename">Downloading...</span>
          <span style="font-family:var(--font-mono);font-size:10px;color:var(--muted-3);" id="dl-percent">0%</span>
        </div>
        <div class="progress-bar-bg" style="margin-bottom:8px;">
          <div class="progress-bar-fill" id="dl-progress-fill" style="width:0%;background:var(--accent);transition:width 0.3s;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:10px;color:var(--muted-3);">
          <span id="dl-speed">Speed: --</span>
          <span id="dl-downloaded">0 / 0</span>
          <span id="dl-eta">ETA: --</span>
        </div>
      </div>
    </div></div>
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
  if(name==='models'){loadModels();}
}

const I={system_prompt:'',trigger:'',skip_existing:true};
function itog(key){
  I[key]=!I[key];
  const t=document.getElementById('itr-'+key);
  if(t)I[key]?t.classList.add('on'):t.classList.remove('on');
}

let _logInterval=null,_lastLen=0;

function _startBatch(cfg){
  document.getElementById('i_go-btn').disabled=true;
  document.getElementById('i_log-box').textContent='Starting...';
  _lastLen=0;
  fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})
  .then(r=>r.json()).then(d=>{
    if(d.ok){_logInterval=setInterval(()=>pollLog(),800);}
    else{document.getElementById('i_log-box').textContent=d.error;document.getElementById('i_go-btn').disabled=false;}
  });
}

function startImages(){
  const stopBtn = document.getElementById('i_stop-btn');
  if (stopBtn) {
    stopBtn.textContent = 'Stop';
    stopBtn.className = 'btn-danger';
    stopBtn.style.background = '';
    stopBtn.style.borderColor = '';
  }

  const cfg={
    ...I,
    input_folder:document.getElementById('i_input_folder').value.trim(),
    output_folder:document.getElementById('i_output_folder').value.trim(),
    trigger:document.getElementById('i_trigger').value.trim(),
    system_prompt:document.getElementById('i_system_prompt').value.trim(),
    mode:'image',num_workers:1,
    resume:false
  };
  if(!cfg.input_folder){alert('Set input folder first');return;}
  _startBatch(cfg);
}

function resumeImages() {
  const cfg={
    ...I,
    input_folder:document.getElementById('i_input_folder').value.trim(),
    output_folder:document.getElementById('i_output_folder').value.trim(),
    trigger:document.getElementById('i_trigger').value.trim(),
    system_prompt:document.getElementById('i_system_prompt').value.trim(),
    mode:'image',num_workers:1,
    resume:true
  };
  if(!cfg.input_folder){alert('Set input folder first');return;}
  _startBatch(cfg);
}

function stopBatch(){
  const btn = document.getElementById('i_stop-btn');
  if (btn.textContent === 'Stop') {
    fetch('/stop',{method:'POST'}).then(r=>r.json()).then(d=>{
      btn.textContent = 'Resume';
      btn.className = 'btn-primary';
      btn.style.background = 'var(--success)';
      btn.style.borderColor = 'var(--success)';
    });
  } else {
    btn.textContent = 'Stop';
    btn.className = 'btn-danger';
    btn.style.background = '';
    btn.style.borderColor = '';
    resumeImages();
  }
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
      document.getElementById('i_go-btn').disabled=false;
      document.getElementById('i_progress-container').style.display='none';
      const stopBtn = document.getElementById('i_stop-btn');
      if (stopBtn && stopBtn.textContent !== 'Resume') {
        stopBtn.textContent = 'Stop';
        stopBtn.className = 'btn-danger';
        stopBtn.style.background = '';
        stopBtn.style.borderColor = '';
      }
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

async function loadModels(){
  const ms=document.getElementById('model-select');const ps=document.getElementById('mmproj-select');
  const sd=document.getElementById('model-status-display');
  try{
    const res=await fetch('/models');const data=await res.json();
    ms.innerHTML='<option value="">-- Select GGUF --</option>';
    data.available.gguf.forEach(m=>{const o=document.createElement('option');o.value=m.path;o.textContent=m.name;if(m.path===data.current.model)o.selected=true;ms.appendChild(o);});
    ps.innerHTML='<option value="">-- Select File --</option>';
    data.available.mmproj.forEach(m=>{const o=document.createElement('option');o.value=m.path;o.textContent=m.name;if(m.path===data.current.mmproj)o.selected=true;ps.appendChild(o);});
    sd.innerHTML='Server: '+(data.current.is_running?'Running':'Stopped')+'<br>Model: '+(data.current.model?data.current.model.split('/').pop():'None')+'<br>MMProj: '+(data.current.mmproj?data.current.mmproj.split('/').pop():'None');
    document.getElementById('model-val-dot').className='val-dot';document.getElementById('mmproj-val-dot').className='val-dot';
    document.getElementById('model-val-text').textContent='Waiting for validation';
    document.getElementById('mmproj-val-text').textContent='Waiting for validation';
    document.getElementById('apply-model-btn').disabled=true;
  }catch(e){sd.textContent="Error loading model list.";console.error(e);}
}

async function refreshModels(){
  const ms=document.getElementById('model-select');const ps=document.getElementById('mmproj-select');
  try{
    const res=await fetch('/models');
    const data=await res.json();
    ms.innerHTML='<option value="">-- Select GGUF --</option>';
    data.available.gguf.forEach(m=>{const o=document.createElement('option');o.value=m.path;o.textContent=m.name;if(m.path===data.current.model)o.selected=true;ms.appendChild(o);});
    ps.innerHTML='<option value="">-- Select File --</option>';
    data.available.mmproj.forEach(m=>{const o=document.createElement('option');o.value=m.path;o.textContent=m.name;if(m.path===data.current.mmproj)o.selected=true;ps.appendChild(o);});
  }catch(e){console.error("Error refreshing models:",e);}
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

// Model Downloader functions
async function downloadPreset(presetName) {
  const display = document.getElementById('download-progress-display');
  display.innerHTML = '<span style="color:var(--accent);">Starting download... Check Live Log at bottom of Images tab.</span>';
  try {
    const res = await fetch('/downloader/download_preset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({preset: presetName})
    });
    const data = await res.json();
    if(data.ok) {
      display.innerHTML = '<span style="color:var(--success);">Download started! Check Live Log at bottom of Images tab.</span>';
    } else {
      display.innerHTML = '<span style="color:var(--danger);">Failed to start: ' + data.error + '</span>';
    }
  } catch(e) {
    display.innerHTML = '<span style="color:var(--danger);">Error: ' + e + '</span>';
  }
}

async function searchHF() {
  const query = document.getElementById('hf-search-input').value.trim();
  if(!query) { alert('Please enter a search query'); return; }
  const resultsDiv = document.getElementById('hf-search-results');
  resultsDiv.style.display = 'block';
  resultsDiv.innerHTML = '<span style="color:var(--accent);">Searching...</span>';
  document.getElementById('hf-files-section').style.display = 'none';
  try {
    const res = await fetch('/downloader/search?query=' + encodeURIComponent(query));
    const data = await res.json();
    if(data.ok && data.results.length > 0) {
      let html = '<table style="width:100%;font-size:10px;border-collapse:collapse;">';
      data.results.forEach(r => {
        html += `<tr style="border-bottom:1px solid var(--border-soft);cursor:pointer;" onclick="selectHFRepo('${r.id}')">`;
        html += `<td style="padding:6px;font-weight:bold;color:var(--accent);">${r.id}</td>`;
        html += `<td style="padding:6px;text-align:right;color:var(--muted-3);">${r.downloads} downloads</td>`;
        html += `</tr>`;
      });
      html += '</table>';
      resultsDiv.innerHTML = html;
    } else {
      resultsDiv.innerHTML = '<span style="color:var(--muted-3);">No repositories found.</span>';
    }
  } catch(e) {
    resultsDiv.innerHTML = '<span style="color:var(--danger);">Search error: ' + e + '</span>';
  }
}

async function selectHFRepo(repoId) {
  const filesSection = document.getElementById('hf-files-section');
  const filesList = document.getElementById('hf-files-list');
  const filesTitle = document.getElementById('hf-files-title');
  filesSection.style.display = 'block';
  filesTitle.textContent = 'Files in ' + repoId;
  filesList.innerHTML = '<span style="color:var(--accent);">Loading files...</span>';
  try {
    const res = await fetch('/downloader/files?repo_id=' + encodeURIComponent(repoId));
    const data = await res.json();
    if(data.ok && data.files.length > 0) {
      let html = '<table style="width:100%;font-size:10px;border-collapse:collapse;">';
      data.files.forEach(f => {
        html += `<tr style="border-bottom:1px solid var(--border-soft);">`;
        html += `<td style="padding:6px;word-break:break-all;">${f}</td>`;
        html += `<td style="padding:6px;text-align:right;"><button class="btn" onclick="downloadHFFile('${repoId}','${f}')" style="padding:2px 8px;">Download</button></td>`;
        html += `</tr>`;
      });
      html += '</table>';
      filesList.innerHTML = html;
    } else {
      filesList.innerHTML = '<span style="color:var(--muted-3);">No matching GGUF or MMProj files found.</span>';
    }
  } catch(e) {
    filesList.innerHTML = '<span style="color:var(--danger);">Error loading files: ' + e + '</span>';
  }
}

async function downloadHFFile(repoId, filename) {
  const display = document.getElementById('download-progress-display');
  display.innerHTML = `<span style="color:var(--accent);">Starting download of ${filename}... Check Live Log at bottom of Images tab.</span>`;
  try {
    const res = await fetch('/downloader/download_file', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({repo_id: repoId, filename: filename})
    });
    const data = await res.json();
    if(data.ok) {
      display.innerHTML = `<span style="color:var(--success);">Started downloading ${filename}! Check Live Log at bottom of Images tab.</span>`;
    } else {
      display.innerHTML = '<span style="color:var(--danger);">Failed to start: ' + data.error + '</span>';
    }
  } catch(e) {
    display.innerHTML = '<span style="color:var(--danger);">Error: ' + e + '</span>';
  }
}

if(document.getElementById('tb-models').classList.contains('active')){loadModels();}

async function initPrompts(){
  try{const res=await fetch('/prompts');const data=await res.json();
    if(data.ok){const s=document.getElementById('i_training_type');data.prompts.forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p;s.appendChild(o);});}
  }catch(e){console.error("Error loading prompts:",e);}
}

async function onTrainingTypeChange(el){
  const name=el.value;if(!name)return;
  try{const res=await fetch('/prompt_content?name='+encodeURIComponent(name));const data=await res.json();
    if(data.ok)document.getElementById('i_system_prompt').value=data.content;
  }catch(e){console.error("Error loading prompt content:",e);}
}

window.onload=()=>{initPrompts();};

// ─────────────────────────────────────────────────────────────────────────────
// REFRESH UI
// ─────────────────────────────────────────────────────────────────────────────
function refreshUI(){
  loadModels();
  initPrompts();
  showToast('UI refreshed', 'info');
}

// ─────────────────────────────────────────────────────────────────────────────
// TOAST NOTIFICATIONS
// ─────────────────────────────────────────────────────────────────────────────
function showToast(message, type='info', duration=4000){
  const container=document.getElementById('toast-container');
  if(!container)return;
  const toast=document.createElement('div');
  toast.className='toast '+type;
  toast.textContent=message;
  container.appendChild(toast);
  setTimeout(()=>{
    toast.classList.add('fade-out');
    setTimeout(()=>toast.remove(),300);
  },duration);
}

// ─────────────────────────────────────────────────────────────────────────────
// DOWNLOAD PROGRESS POLLING
// ─────────────────────────────────────────────────────────────────────────────
let _dlProgressInterval=null;
let _dlWasActive=false;

function formatBytes(bytes){
  if(bytes===0) return '0 B';
  const k=1024;
  const sizes=['B','KB','MB','GB'];
  const i=Math.floor(Math.log(bytes)/Math.log(k));
  return parseFloat((bytes/Math.pow(k,i)).toFixed(1))+' '+sizes[i];
}

function formatEta(seconds){
  if(seconds<=0||seconds>86400) return '--';
  if(seconds<60) return seconds+'s';
  const m=Math.floor(seconds/60);
  const s=seconds%60;
  return m+'m '+s+'s';
}

function pollDownloadProgress(){
  fetch('/downloader/progress').then(r=>r.json()).then(p=>{
    const container=document.getElementById('dl-progress-container');
    if(!container) return;
    
    if(p.active || p.status==='downloading'){
      container.style.display='block';
      _dlWasActive=true;
      
      document.getElementById('dl-filename').textContent='Downloading: '+p.filename;
      document.getElementById('dl-percent').textContent=p.percent+'%';
      document.getElementById('dl-progress-fill').style.width=p.percent+'%';
      document.getElementById('dl-speed').textContent='Speed: '+formatBytes(p.speed_bps)+'/s';
      document.getElementById('dl-downloaded').textContent=formatBytes(p.downloaded_bytes)+' / '+formatBytes(p.total_bytes);
      document.getElementById('dl-eta').textContent='ETA: '+formatEta(p.eta_seconds);
    } else {
      if(_dlWasActive){
        _dlWasActive=false;
        if(p.status==='completed'){
          showToast('Download completed: '+p.filename, 'success', 5000);
          document.getElementById('dl-percent').textContent='100%';
          document.getElementById('dl-progress-fill').style.width='100%';
          document.getElementById('dl-eta').textContent='Done';
          // Refresh model list
          loadModels();
          setTimeout(()=>{container.style.display='none';},3000);
        } else if(p.status==='failed'){
          showToast('Download failed: '+p.filename, 'error', 5000);
          container.style.display='none';
        } else {
          container.style.display='none';
        }
      }
    }
  }).catch(e=>{});
}

function startDlProgressPolling(){
  if(_dlProgressInterval) return;
  _dlProgressInterval=setInterval(pollDownloadProgress,1000);
}

// Start polling immediately
startDlProgressPolling();

// Also trigger notification when download starts via preset/HF buttons
const origDownloadPreset=downloadPreset;
downloadPreset=function(name){
  showToast('Starting download: '+name+' model...', 'info');
  startDlProgressPolling();
  return origDownloadPreset(name);
};

const origDownloadHFFile=downloadHFFile;
downloadHFFile=function(repoId,filename){
  showToast('Starting download: '+filename, 'info');
  startDlProgressPolling();
  return origDownloadHFFile(repoId,filename);
};

// ─────────────────────────────────────────────────────────────────────────────
// ZIP FOLDER FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────
async function zipFolder(type) {
  showToast('Starting zip operation...', 'info');
  try {
    const res = await fetch('/zip/' + type, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
    const data = await res.json();
    if (data.ok) {
      const sizeMB = (data.size / (1024 * 1024)).toFixed(2);
      showToast(type.charAt(0).toUpperCase() + type.slice(1) + ' folder zipped successfully! Size: ' + sizeMB + ' MB', 'success', 5000);
    } else {
      showToast('Failed to zip ' + type + ' folder: ' + data.error, 'error');
    }
  } catch(e) {
    showToast('Network error: ' + e, 'error');
  }
}

async function deleteFolder(type) {
  if (!confirm('Delete all contents of the ' + type + ' folder?')) return;
  showToast('Deleting ' + type + ' folder contents...', 'info');
  try {
    const res = await fetch('/delete/' + type, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'}
    });
    const data = await res.json();
    if (data.ok) {
      showToast(type.charAt(0).toUpperCase() + type.slice(1) + ' folder cleared! Removed ' + data.deleted + ' items.', 'success', 5000);
    } else {
      showToast('Failed to delete ' + type + ' folder: ' + data.error, 'error');
    }
  } catch(e) {
    showToast('Network error: ' + e, 'error');
  }
}
</script>
</body>
</html>"""

_batch_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_errors.log")

@app.route("/prompts", methods=["GET"])
def get_prompts():
    if not os.path.exists(PROMPTS_DIR):
        return jsonify({"ok": False, "error": "Prompts directory not found"}), 404
    try:
        files = [os.path.splitext(f)[0] for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
        return jsonify({"ok": True, "prompts": sorted(files)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/prompt_content", methods=["GET"])
def get_prompt_content():
    prompt_name = request.args.get("name")
    if not prompt_name:
        return jsonify({"ok": False, "error": "Missing prompt name"}), 400
    file_path = os.path.join(PROMPTS_DIR, f"{prompt_name}.txt")
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

    resume = cfg.get("resume", False)
    if not resume:
        with state.stats_lock:
            state.completed_files.clear()

    out = cfg.get("output_folder") or cfg["input_folder"]
    if cfg.get("skip_existing", False) and os.path.isdir(out):
        txt_count = len([f for f in os.listdir(out) if f.endswith(".txt")])
        if txt_count > 0:
            _log(f"  skip_existing is ON: {txt_count} .txt files already exist in output folder")

    def run():
        global state
        try:
            _run_batch(cfg)
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
    _log("Stop requested. Clearing queue and finishing current items...")

    if state.batch_queue:
        count = 0
        while not state.batch_queue.empty():
            try:
                state.batch_queue.get_nowait()
                state.batch_queue.task_done()
                count += 1
            except queue.Empty:
                break
        if count > 0:
            _log(f"  Cleared {count} pending items from queue.")

    return jsonify({"ok": True})

@app.route("/log")
def log():
    from_idx = int(request.args.get("from", 0))
    with state.log_lock:
        lines = state.log_lines[from_idx:]
    return jsonify({"lines": lines, "done": not state.batch_running, "server_running": _llama_running()})

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
    model_dir = request.args.get("model_dir", state.config.get("model_dir", MODEL_DIR))
    mmproj_dir = request.args.get("mmproj_dir", state.config.get("mmproj_dir", MODEL_DIR))

    available = _scan_available_models(model_dir=model_dir, mmproj_dir=mmproj_dir)
    return jsonify({
        "available": available,
        "current": {
            "model": state.config["model_path"],
            "mmproj": state.config["mmproj_path"],
            "model_dir": state.config.get("model_dir", MODEL_DIR),
            "mmproj_dir": state.config.get("mmproj_dir", MODEL_DIR),
            "llama_server_bin": state.config.get("llama_server_bin", "llama-server"),
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
    new_model_dir = data.get("model_dir", "")
    new_mmproj_dir = data.get("mmproj_dir", "")
    new_llama_bin = data.get("llama_server_bin", "")

    if not new_model:
        return jsonify({"ok": False, "error": "Missing model path"}), 400

    if not os.path.exists(new_model):
        return jsonify({"ok": False, "error": f"Model file not found: {new_model}"}), 400

    if new_mmproj and not os.path.exists(new_mmproj):
        return jsonify({"ok": False, "error": f"MMProj file not found: {new_mmproj}"}), 400

    state.config["model_path"] = new_model
    state.config["mmproj_path"] = new_mmproj
    if new_model_dir:
        state.config["model_dir"] = new_model_dir
    if new_mmproj_dir:
        state.config["mmproj_dir"] = new_mmproj_dir
    if new_llama_bin:
        state.config["llama_server_bin"] = new_llama_bin
    state.save_config()

    if state.batch_running:
        return jsonify({"ok": False, "error": "Cannot switch model while batch is running"}), 409

    status_msg = _start_llama_server()

    if "failed" in status_msg or "error" in status_msg.lower():
        return jsonify({"ok": False, "error": status_msg}), 500

    return jsonify({"ok": True, "message": f"Server restarted with {os.path.basename(new_model)}"})


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOADER UTILS & ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
def _download_hf_file(repo_id, filename):
    dest_dir = os.path.abspath(state.config.get("model_dir", MODEL_DIR))
    os.makedirs(dest_dir, exist_ok=True)
    
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    _log(f"Starting download of {filename} from {repo_id}...")
    
    # Initialize progress tracking
    with state.stats_lock:
        state.download_progress.update({
            "active": True,
            "filename": filename,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "speed_bps": 0,
            "eta_seconds": 0,
            "percent": 0,
            "status": "downloading"
        })
    
    final_path = os.path.join(dest_dir, filename)
    tmp_path = final_path + ".downloading"
    
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()
        last_log_time = start_time
        
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    eta = (total - downloaded) / speed if speed > 0 and total > 0 else 0
                    pct = int(downloaded * 100 / total) if total > 0 else 0
                    
                    with state.stats_lock:
                        state.download_progress.update({
                            "downloaded_bytes": downloaded,
                            "total_bytes": total,
                            "speed_bps": int(speed),
                            "eta_seconds": int(eta),
                            "percent": pct
                        })
                    
                    # Log every 5 seconds or at milestones
                    now = time.time()
                    if now - last_log_time >= 5 or pct in (25, 50, 75):
                        dl_mb = downloaded / (1024 * 1024)
                        tot_mb = total / (1024 * 1024) if total > 0 else 0
                        speed_mb = speed / (1024 * 1024)
                        _log(f"[Download] {filename}: {pct}% ({dl_mb:.1f}/{tot_mb:.1f} MB) - {speed_mb:.1f} MB/s")
                        last_log_time = now
        
        # Move from temp to final path
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(tmp_path, final_path)
        
        _log(f"Successfully downloaded {filename} to {dest_dir}")
        with state.stats_lock:
            state.download_progress.update({
                "active": False,
                "status": "completed",
                "percent": 100
            })
        return True
    except Exception as e:
        _log(f"Download failed: {e}")
        with state.stats_lock:
            state.download_progress.update({
                "active": False,
                "status": "failed"
            })
        # Cleanup partial download
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        return False

def _downloader_worker(repo_id, filename):
    global state
    state.downloader_running = True
    try:
        _download_hf_file(repo_id, filename)
    finally:
        state.downloader_running = False

def _preset_downloader_worker(preset):
    global state
    state.downloader_running = True
    try:
        dest_dir = os.path.abspath(state.config.get("model_dir", MODEL_DIR))
        if preset == "qwen":
            repo_id = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
            model_file = "Qwen3VL-8B-Instruct-Q8_0.gguf"
            mmproj_file = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
            
            _log("Preset Download: Qwen 3 8B VL selected. Downloading model and projector...")
            ok1 = _download_hf_file(repo_id, model_file)
            if ok1:
                _log("Model downloaded. Waiting 5s before projector download to avoid throttling...")
                time.sleep(5)
                _download_hf_file(repo_id, mmproj_file)
        elif preset == "gemma":
            repo_id = "unsloth/gemma-4-12b-it-GGUF"
            model_file = "gemma-4-12b-it-Q8_0.gguf"
            mmproj_file = "mmproj-F16.gguf"
            
            _log("Preset Download: Gemma 4 12B selected. Downloading model and projector...")
            ok1 = _download_hf_file(repo_id, model_file)
            if ok1:
                _log("Model downloaded. Waiting 5s before projector download to avoid throttling...")
                time.sleep(5)
                _download_hf_file(repo_id, mmproj_file)
    finally:
        state.downloader_running = False

@app.route("/downloader/search", methods=["GET"])
def downloader_search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"ok": False, "error": "Query is required"}), 400
    try:
        url = f"https://huggingface.co/api/models?search={query}&limit=20"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        results = []
        for item in data:
            results.append({
                "id": item.get("id"),
                "downloads": item.get("downloads", 0)
            })
            
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/downloader/files", methods=["GET"])
def downloader_files():
    repo_id = request.args.get("repo_id", "")
    if not repo_id:
        return jsonify({"ok": False, "error": "Repo ID is required"}), 400
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        files = [s["rfilename"] for s in data.get("siblings", [])]
        allowed_exts = {".gguf", ".safetensors", ".bin", ".pt"}
        filtered_files = [f for f in files if any(f.endswith(ext) for ext in allowed_exts) or "mmproj" in f.lower() or "projector" in f.lower()]
        
        return jsonify({"ok": True, "files": sorted(filtered_files)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/downloader/download_file", methods=["POST"])
def downloader_download_file():
    global state
    if state.downloader_running:
        return jsonify({"ok": False, "error": "Another download is already in progress."}), 409
        
    data = request.get_json()
    repo_id = data.get("repo_id")
    filename = data.get("filename")
    
    if not repo_id or not filename:
        return jsonify({"ok": False, "error": "repo_id and filename are required"}), 400
        
    t = threading.Thread(target=_downloader_worker, args=(repo_id, filename), daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/downloader/download_preset", methods=["POST"])
def downloader_download_preset():
    global state
    if state.downloader_running:
        return jsonify({"ok": False, "error": "Another download is already in progress."}), 409
        
    data = request.get_json()
    preset = data.get("preset")
    if preset not in ("qwen", "gemma"):
        return jsonify({"ok": False, "error": "Invalid preset"}), 400
        
    t = threading.Thread(target=_preset_downloader_worker, args=(preset,), daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/downloader/status", methods=["GET"])
def downloader_status():
    return jsonify({"running": state.downloader_running})

@app.route("/downloader/progress", methods=["GET"])
def downloader_progress():
    with state.stats_lock:
        return jsonify(state.download_progress.copy())


# ─────────────────────────────────────────────────────────────────────────────
# ZIP FOLDER ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/zip/input", methods=["POST"])
def zip_input():
    data = request.get_json() or {}
    output_path = data.get("output_path")
    success, result = _zip_input_folder(output_path)
    if success:
        return jsonify({"ok": True, "zip_path": result, "size": os.path.getsize(result)})
    else:
        return jsonify({"ok": False, "error": result}), 500

@app.route("/zip/output", methods=["POST"])
def zip_output():
    data = request.get_json() or {}
    output_path = data.get("output_path")
    success, result = _zip_output_folder(output_path)
    if success:
        return jsonify({"ok": True, "zip_path": result, "size": os.path.getsize(result)})
    else:
        return jsonify({"ok": False, "error": result}), 500

@app.route("/zip/custom", methods=["POST"])
def zip_custom():
    data = request.get_json()
    folder = data.get("folder", "")
    output_path = data.get("output_path")
    if not folder:
        return jsonify({"ok": False, "error": "Missing folder parameter"}), 400
    success, result = _zip_folder_contents(folder, output_path)
    if success:
        return jsonify({"ok": True, "zip_path": result, "size": os.path.getsize(result)})
    else:
        return jsonify({"ok": False, "error": result}), 500

@app.route("/delete/input", methods=["POST"])
def delete_input():
    return _delete_folder_contents(DEFAULT_INPUT_FOLDER)

@app.route("/delete/output", methods=["POST"])
def delete_output():
    return _delete_folder_contents(DEFAULT_OUTPUT_FOLDER)

def _delete_folder_contents(folder_path):
    if not os.path.isdir(folder_path):
        return jsonify({"ok": False, "error": f"Folder not found: {folder_path}"}), 404
    try:
        deleted = 0
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                deleted += 1
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                deleted += 1
        return jsonify({"ok": True, "deleted": deleted})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    _auto_setup_llama_binaries()
    print("\n" + "─"*52)
    print("  Dataset Captioner — Image Captioning Edition")
    print(f"  model  : {state.config['model_path']}")
    print(f"  server : {state.config['llama_server_bin']}")
    print("─"*52)
    print("  opening http://127.0.0.1:7860\n")
    try:
        threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    except Exception:
        print("  [Note] Headless environment detected. Open http://127.0.0.1:7860 manually.")
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
