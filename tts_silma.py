"""
SILMA TTS API client.
Generates Arabic speech via https://api.silma.ai/tts
All parameters from the official API docs are supported.
"""
import os
import base64
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv

# Use bundled ffmpeg (no system install needed)
try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    _FFMPEG = "ffmpeg"

from pydub import AudioSegment
AudioSegment.converter = str(_FFMPEG)
AudioSegment.ffmpeg = str(_FFMPEG)

load_dotenv()

SILMA_API_KEY = os.getenv("SILMA_API_KEY", "").strip('"').strip("'")
SILMA_USER_ID = os.getenv("SILMA_USER_ID", "").strip('"').strip("'")

SILMA_GENERATE_URL = "https://api.silma.ai/tts/generate"
OUTPUT_DIR = Path("Output_Voice")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model options
MODELS = {
    "silma-tts-pro-ksa-large": "Saudi dialect, 330M parameters",
    "silma-tts-pro-ksa-small": "Saudi dialect, 150M parameters (faster, lower quality)",
    "silma-tts-pro-msa-large": "Modern Standard Arabic, 330M parameters",
    "silma-tts-pro-msa-small": "Modern Standard Arabic, 150M parameters (faster, lower quality)",
}

# Default voice options
VOICES = ["Sulaiman", "Salma", "Salman", "Sarah", "Sam", "Samantha"]


def generate_speech(
    text: str,
    output_filename: str = None,
    *,
    model_id: str = "silma-tts-pro-ksa-large",
    reference_audio_id: str = "Sulaiman",
    nfe_steps: int = 16,
    seed: int = 42,
    remove_silence: bool = False,
    speaking_speed: float = 1.1,
    use_ema: bool = None,
    normalize_numbers: bool = True,
    pronunciation_overrides: dict = None,
    custom_ref_audio: str = None,
    enable_server_pronunciation_overrides: bool = False,
    user_id: str = None,
) -> str:
    """
    Generate speech from text using SILMA TTS API.
    https://dev.silma.ai/apis/silma-tts-api-1/versions/.../paths/generate/post

    Args:
        text: Arabic text to synthesize (with or without tashkeel).
        output_filename: Optional output filename (without extension).
        model_id: silma-tts-pro-ksa-large, silma-tts-pro-ksa-small,
                  silma-tts-pro-msa-large, silma-tts-pro-msa-small.
        reference_audio_id: Voice style. Default: Sulaiman, Salma, Salman, Sarah, Sam, Samantha.
                           Or custom Voice ID from app.silma.ai/voices, or "Custom" with custom_ref_audio.
        nfe_steps: Function evaluation steps (speed/quality). Recommended 16.
        seed: Random seed for reproducibility.
        remove_silence: Strip silence from output.
        speaking_speed: Speech speed (increments of 0.1).
        use_ema: EMA weights. False for KSA, True for MSA. Auto-set if None.
        normalize_numbers: Convert numbers to words.
        pronunciation_overrides: Dict of words -> tashkeel pronunciations, e.g. {"اكل":"اُكِل"}.
        custom_ref_audio: Base64-encoded custom reference audio (optional).
        enable_server_pronunciation_overrides: Use overrides from your account.
        user_id: For custom voices / pronunciation overrides. Uses SILMA_USER_ID from env if not set.

    Returns:
        Path to saved MP3 file.
    """
    if not SILMA_API_KEY:
        raise ValueError("SILMA_API_KEY not found in .env")

    # Auto-set use_ema: false for KSA, true for MSA
    if use_ema is None:
        use_ema = "msa" in model_id.lower()

    payload = {
        "model_id": model_id,
        "text": text,
        "reference_audio_id": reference_audio_id,
        "nfe_steps": nfe_steps,
        "seed": seed,
        "remove_silence": remove_silence,
        "speaking_speed": speaking_speed,
        "use_ema": use_ema,
        "normalize_numbers": normalize_numbers,
        "enable_server_pronunciation_overrides": enable_server_pronunciation_overrides,
    }

    if pronunciation_overrides:
        payload["pronunciation_overrides"] = pronunciation_overrides
    if custom_ref_audio:
        payload["custom_ref_audio"] = custom_ref_audio
    if user_id or SILMA_USER_ID:
        payload["user_id"] = user_id or SILMA_USER_ID

    headers = {
        "Content-Type": "application/json",
        "apiKey": SILMA_API_KEY,
    }

    resp = requests.post(SILMA_GENERATE_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    audio_b64 = data.get("audio_base64_encoded")
    if not audio_b64:
        raise ValueError("No audio in SILMA response")

    audio_bytes = base64.b64decode(audio_b64)

    if output_filename is None:
        safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_text = safe_text.replace(" ", "_")
        output_filename = f"silma_{safe_text}" if safe_text else "silma_output"

    if not output_filename.endswith(".mp3"):
        output_filename = output_filename.replace(".wav", "").replace(".mp3", "") + ".mp3"

    # Detect format: WAV starts with RIFF, MP3 with ID3 or 0xFF 0xFB/0xFA
    if audio_bytes[:4] == b"RIFF":
        fmt = "wav"
    elif audio_bytes[:3] == b"ID3" or (len(audio_bytes) >= 2 and audio_bytes[:2] in (b"\xff\xfb", b"\xff\xfa")):
        fmt = "mp3"
    else:
        fmt = "wav"

    segment = AudioSegment.from_file(BytesIO(audio_bytes), format=fmt)
    output_path = OUTPUT_DIR / output_filename
    segment.export(str(output_path), format="mp3")

    print(f"SILMA audio saved to: {output_path}")
    return str(output_path)
