"""
SILMA TTS V2 Streaming API client.
Uses the streaming endpoint to generate Arabic speech.
Based on SILMA_TTS_V2_Streaming.ipynb
"""
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

SILMA_V2_URL = os.getenv(
    "SILMA_V2_URL",
    "https://2orycinwjnbwqz-7860.proxy.runpod.net/stream"
)
OUTPUT_DIR = Path("Output_Voice")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 250

VOICES = ["sarah", "salman"]


def generate_speech(
    text: str,
    output_filename: str = None,
    *,
    voice_id: str = "sarah",
    speed: float = 0.5,
    creativity: float = 0.5,
) -> str:
    """
    Generate speech from text using SILMA TTS V2 Streaming API.

    Args:
        text: Arabic text (max 250 chars).
        output_filename: Optional output filename (without extension).
        voice_id: Reference voice — "sarah" or "salman".
        speed: Speech speed (0–1), maps to cfg.
        creativity: Creativity & style (0–1).

    Returns:
        Path to saved WAV file.
    """
    text = text.strip()[:MAX_TEXT_LENGTH]
    if not text:
        raise ValueError("Text cannot be empty")

    voice_id = voice_id.lower() if voice_id else "sarah"
    if voice_id not in VOICES:
        voice_id = "sarah"

    payload = {
        "text": text,
        "voice_id": voice_id,
        "cfg": float(speed),
        "creativity": float(creativity),
    }

    chunks = []
    carry_over = b""

    with requests.post(SILMA_V2_URL, json=payload, stream=True, timeout=120) as r:
        if r.status_code != 200:
            try:
                err = r.json()
                msg = err.get("detail", str(err))
            except Exception:
                msg = r.text or f"HTTP {r.status_code}"
            raise ValueError(f"SILMA V2 API error: {msg}")

        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                current_data = carry_over + chunk
                num_floats = len(current_data) // 4
                cut_off = num_floats * 4
                valid_bytes = current_data[:cut_off]
                carry_over = current_data[cut_off:]

                if valid_bytes:
                    waveform = np.frombuffer(valid_bytes, dtype=np.float32)
                    chunks.append(waveform)

    if not chunks:
        raise ValueError("No audio received from SILMA V2 API")

    full_waveform = np.concatenate(chunks)

    if output_filename is None:
        safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_text = safe_text.replace(" ", "_")
        output_filename = f"silma_v2_{safe_text}" if safe_text else "silma_v2_output"

    if not output_filename.endswith(".wav"):
        output_filename = output_filename.replace(".mp3", "") + ".wav"

    output_path = OUTPUT_DIR / output_filename

    import torch
    import torchaudio as ta
    tensor = torch.from_numpy(full_waveform.astype(np.float32)).unsqueeze(0)
    ta.save(str(output_path), tensor, SAMPLE_RATE)

    print(f"SILMA V2 audio saved to: {output_path}")
    return str(output_path)
