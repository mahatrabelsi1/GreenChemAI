from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]
PIPER_MODEL_DIR = ROOT / "models" / "piper"
DEFAULT_MODEL = PIPER_MODEL_DIR / "en_US-amy-medium.onnx"
DEFAULT_CONFIG = PIPER_MODEL_DIR / "en_US-amy-medium.onnx.json"
GENERATED_AUDIO_DIR = ROOT / "assets" / "generated_audio"


def piper_status() -> dict:
    executable = os.environ.get("PIPER_EXECUTABLE") or shutil.which("piper")
    if not executable:
        executable_name = "piper.exe" if os.name == "nt" else "piper"
        candidates = [
            Path(sysconfig.get_path("scripts")) / executable_name,
        ]
        if os.name == "nt":
            appdata = os.environ.get("APPDATA")
            if appdata:
                candidates.append(
                    Path(appdata)
                    / "Python"
                    / f"Python{sys.version_info.major}{sys.version_info.minor}"
                    / "Scripts"
                    / executable_name
                )
        for candidate in candidates:
            if candidate.exists():
                executable = str(candidate)
                break
    model = Path(os.environ.get("PIPER_MODEL_PATH", DEFAULT_MODEL))
    config = Path(os.environ.get("PIPER_CONFIG_PATH", DEFAULT_CONFIG))
    return {
        "available": bool(executable and model.exists() and config.exists()),
        "executable": executable or "",
        "model": str(model),
        "config": str(config),
        "missing": [
            label
            for label, present in [
                ("piper executable", bool(executable)),
                ("voice model .onnx", model.exists()),
                ("voice config .json", config.exists()),
            ]
            if not present
        ],
    }


def synthesize_with_piper(text: str, voice_key: str = "amy-medium") -> Path | None:
    status = piper_status()
    if not status["available"]:
        return None

    GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{voice_key}:{text}".encode("utf-8")).hexdigest()[:16]
    output_path = GENERATED_AUDIO_DIR / f"tts_{digest}.wav"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    command = [
        status["executable"],
        "--model",
        status["model"],
        "--config",
        status["config"],
        "--output_file",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except Exception:
        return None

    if not output_path.exists() or output_path.stat().st_size == 0:
        return None
    return output_path


def audio_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"
