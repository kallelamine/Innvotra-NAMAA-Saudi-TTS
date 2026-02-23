# Arabic TTS — NAMAA & SILMA

**صوت طبيعي — تحويل النص من العربية إلى اللهجة السعودية أو العربية الفصحى**  
*Natural voice — Convert text from Arabic to Saudi dialect or Modern Standard Arabic*

A Text-to-Speech application that generates natural Arabic speech. Choose between two engines:

- **NAMAA Saudi TTS** — Local model (Hugging Face), Saudi dialect, runs on your machine
- **SILMA TTS API** — Cloud API, Saudi dialect or MSA, multiple voices and models

NAMAA outputs **WAV** (no external tools), SILMA outputs **MP3**. Flask web interface included.

---

## Features

- **Model choice** — Switch between NAMAA (local) and SILMA (API) in the Web UI
- **Saudi dialect & MSA** — NAMAA for Saudi; SILMA for Saudi or Modern Standard Arabic
- **WAV (NAMAA) / MP3 (SILMA)** — NAMAA saves WAV directly; SILMA uses bundled ffmpeg for MP3
- **Web UI** — Text input, parameter controls, audio playback
- **CLI mode** — Run NAMAA TTS from the command line via `tts_saudi.py`
- **Voice cloning (NAMAA)** — Optional reference audio for custom voice/style transfer
- **Multiple voices (SILMA)** — Sulaiman, Salma, Salman, Sarah, Sam, Samantha; plus custom voices

---

## Project Structure

```
Arabic-TTS/
├── app.py              # Flask web application
├── tts_saudi.py        # NAMAA TTS engine (local)
├── tts_silma.py        # SILMA TTS API client (cloud)
├── requirements.txt    # Python dependencies
├── .env                # Tokens and keys (create this, do not commit)
├── .env.example        # Example env file
├── templates/
│   └── index.html      # Web UI
└── Output_Voice/       # Generated audio (WAV from NAMAA, MP3 from SILMA)
```

---

## Prerequisites

- **Python 3.9+**
- **CUDA** (recommended for NAMAA) or CPU
- **Hugging Face account** — for NAMAA model access
- **SILMA account** — for SILMA API (get API key and User ID from [app.silma.ai](https://app.silma.ai/api-keys))

> **Note:** NAMAA outputs WAV directly (no extra tools). SILMA uses `imageio-ffmpeg` for MP3 (no system ffmpeg needed).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kallelamine/Innvotra-NAMAA-Saudi-TTS.git
cd Innvotra-NAMAA-Saudi-TTS
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the project root:

```env
# Required for NAMAA (local model)
HF_token=your_huggingface_token_here

# Required for SILMA (API)
SILMA_API_KEY=your_silma_api_key_here
SILMA_USER_ID=your_silma_user_id_here
```

- **HF_token** — Get from [Hugging Face → Access Tokens](https://huggingface.co/settings/tokens). Need read access to NAMAA-Saudi-TTS.
- **SILMA_API_KEY** & **SILMA_USER_ID** — Get from [SILMA API Keys](https://app.silma.ai/api-keys).

---

## How to Use the Platform

### Web UI

1. Start the server:

   ```bash
   python app.py
   ```

2. Open **http://127.0.0.1:5000** in your browser.

3. Enter or paste Arabic text in the text area (up to 500 characters for SILMA, 300 for NAMAA).

4. Choose a model:
   - **NAMAA Saudi (محلي)** — Local inference, Saudi dialect, uses exaggeration, temperature, CFG, seed
   - **SILMA TTS (API)** — Cloud API, choose model (KSA/MSA), voice, speed, and other options

5. Adjust parameters for the selected model (see options below).

6. Click **توليد الصوت** (Generate voice).

7. Play the generated audio in the player below.

### Command Line (NAMAA only)

```bash
python tts_saudi.py
```

This runs example generations and saves WAV files to `Output_Voice/`.

---

## Options & Parameters

### NAMAA Saudi TTS (local)

| Parameter       | Default | Range    | Description                                      |
|----------------|---------|----------|--------------------------------------------------|
| **exaggeration** | 0.5   | 0.25–2.0 | Emotion/expressiveness                           |
| **temperature**  | 0.8   | 0.05–5.0 | Randomness (lower = more consistent)             |
| **cfg_weight**   | 0.5   | 0.0–1.0  | Classifier-free guidance / pace control          |
| **seed**         | 0     | 0 = random | Reproducibility (use > 0 for same output)     |

- **reference_audio_path** — Optional 3–10 second WAV/FLAC in Saudi dialect for voice cloning

### SILMA TTS (API)

| Parameter           | Default   | Description                                    |
|--------------------|-----------|------------------------------------------------|
| **model_id**       | silma-tts-pro-ksa-large | KSA large/small, MSA large/small       |
| **reference_audio_id** | Sulaiman | Voice: Sulaiman, Salma, Salman, Sarah, Sam, Samantha |
| **nfe_steps**      | 16        | Speed/quality trade-off                        |
| **seed**           | 42        | Reproducibility                                |
| **speaking_speed** | 1.1       | Speech speed (increments of 0.1)               |
| **remove_silence** | false     | Strip silence from output                      |
| **use_ema**        | auto      | false for KSA, true for MSA                    |
| **normalize_numbers** | true  | Convert numbers to words                       |

- **pronunciation_overrides** — JSON object for custom pronunciations (e.g. `{"اكل":"اُكِل"}`)

---

## Use Cases

### When to use NAMAA Saudi TTS (local)

| Use case | Why NAMAA |
|----------|-----------|
| **Offline / on-premise** | Runs locally, no external API calls |
| **Privacy & data control** | Text and audio stay on your machine |
| **High-volume / cost control** | No per-request API fees |
| **Saudi dialect focus** | Tuned for Saudi Arabic |
| **Voice cloning** | Use reference audio for custom voices |
| **Research / experimentation** | Full control over model and parameters |

### When to use SILMA TTS (API)

| Use case | Why SILMA |
|----------|-----------|
| **No GPU or heavy setup** | Runs in the cloud, lightweight client |
| **Modern Standard Arabic (MSA)** | Dedicated MSA models (msa-large, msa-small) |
| **Multiple voices** | Built-in Sulaiman, Salma, Salman, Sarah, Sam, Samantha |
| **Custom voices** | Use Voice IDs from [app.silma.ai/voices](https://app.silma.ai/voices) |
| **Faster prototyping** | No model download, works immediately |
| **Pronunciation control** | Pronunciation overrides and server-side customization |

---

## Python API Examples

### NAMAA

```python
from tts_saudi import initialize_model, generate_speech

model = initialize_model(device="cuda")  # or "cpu"

output_path = generate_speech(
    model,
    text="آبي أروح البقالة أشتري كم غرض وأرجع بسرعة.",
    exaggeration=0.5,
    temperature=0.8,
    cfg_weight=0.5,
    seed=0,
    reference_audio_path=None,
    device="cuda"
)
# WAV saved to Output_Voice/
```

### SILMA

```python
from tts_silma import generate_speech

output_path = generate_speech(
    text="بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ",
    model_id="silma-tts-pro-ksa-large",
    reference_audio_id="Sulaiman",
    nfe_steps=16,
    seed=42,
    speaking_speed=1.1,
)
# MP3 saved to Output_Voice/
```

---

## Models

### NAMAA

- **Base model**: [NAMAA-Space/NAMAA-Saudi-TTS](https://huggingface.co/NAMAA-Space/NAMAA-Saudi-TTS)
- **Architecture**: Chatterbox Multilingual TTS (ResembleAI)
- **Output**: Saudi dialect Arabic, WAV
- **License**: MIT

### SILMA

- **API**: [SILMA TTS API](https://dev.silma.ai/apis/silma-tts-api-1)
- **Models**: KSA (Saudi) and MSA (Modern Standard Arabic), large and small variants
- **Output**: MP3
- **Voices**: Default (Sulaiman, Salma, etc.) or custom from your account

---

## Limitations

- NAMAA: Lack of tashkeel may affect pronunciation; GPU recommended
- SILMA: Requires internet and API quota
- Text limits: 300 chars (NAMAA), 500 chars (SILMA) in the Web UI

---

## License

MIT License — see [NAMAA-Saudi-TTS](https://huggingface.co/NAMAA-Space/NAMAA-Saudi-TTS) for model license.

---

## Acknowledgments

- [NAMAA Community](https://huggingface.co/NAMAA-Space) — Saudi dialect TTS model
- [SILMA](https://silma.ai) — Arabic TTS API
- [Resemble AI](https://github.com/resemble-ai/chatterbox) — Chatterbox TTS architecture
