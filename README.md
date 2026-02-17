# Innvotra NAMAA Saudi TTS

**صوت طبيعي — تحويل النص من العربية إلى اللهجة السعودية**  
*Natural voice — Convert text from Arabic to Saudi dialect*

A Text-to-Speech application that generates natural **Saudi dialect Arabic** speech from text. Built on the [NAMAA-Saudi-TTS](https://huggingface.co/NAMAA-Space/NAMAA-Saudi-TTS) model from Hugging Face, with a Flask web interface for easy use.

---

## Features

- **Saudi dialect output** — Generates natural Saudi Arabic speech (not Modern Standard Arabic)
- **Web UI** — Flask interface with text input, parameter controls, and audio playback
- **CLI mode** — Run TTS from the command line via `tts_saudi.py`
- **Voice cloning** — Optional reference audio for custom voice/style transfer
- **Tunable parameters** — Exaggeration, temperature, CFG weight, seed (matching [HF Space demo](https://huggingface.co/spaces/omarelshehy/NAMAA-Saudi-Voice))

---

## Project Structure

```
Innvotra-NAMAA-Saudi-TTS/
├── app.py              # Flask web application
├── tts_saudi.py        # TTS engine (model loading + speech generation)
├── requirements.txt    # Python dependencies
├── .env                # Hugging Face token (create this, do not commit)
├── .env.example        # Example env file (optional)
├── templates/
│   └── index.html      # Web UI
└── Output_Voice/       # Generated audio files
```

---

## Prerequisites

- **Python 3.9+**
- **CUDA** (recommended) or CPU
- **Hugging Face account** with access to [NAMAA-Space/NAMAA-Saudi-TTS](https://huggingface.co/NAMAA-Space/NAMAA-Saudi-TTS)

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

```
HF_token=your_huggingface_token_here
```

Get your token from [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens). You need read access to the NAMAA-Saudi-TTS model.

---

## How to Run

### Web UI (Flask)

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### Command Line (CLI)

```bash
python tts_saudi.py
```

This runs the main script with example texts and saves audio to `Output_Voice/`.

---

## Options & Parameters

### Text Input

- **Max 300 characters** — Text is truncated to 300 characters (matching the Hugging Face demo)
- **Saudi dialect recommended** — Best results with Saudi dialect text; MSA is also supported

### TTS Parameters

| Parameter      | Default | Range   | Description                                              |
|----------------|---------|---------|----------------------------------------------------------|
| **exaggeration** | 0.5   | 0.25–2.0 | Emotion/expressiveness. Lower = calmer, higher = more expressive |
| **temperature**  | 0.8   | 0.05–5.0 | Randomness. Lower = more consistent, higher = more varied |
| **cfg_weight**   | 0.5   | 0.0–1.0  | Classifier-free guidance / pace control                 |
| **seed**         | 0     | 0 = random | Reproducibility. Use any integer > 0 for same output   |

### Voice Cloning (Optional)

- **reference_audio_path** — Path to a 3–10 second audio file (WAV/FLAC) in Saudi dialect
- Used for voice/style transfer
- If not provided, the model uses its default Saudi voice

### Python API Example

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
    reference_audio_path=None,  # or "path/to/voice.wav"
    device="cuda"
)
# Audio saved to Output_Voice/
```

---

## Pushing to GitHub

### 1. Create a `.gitignore` file

Create `.gitignore` in the project root:

```
# Environment
.env
venv/
.venv/
env/

# Output
Output_Voice/
*.wav

# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/
build/

# IDE
.idea/
.vscode/
*.swp
*.swo
```

### 2. Initialize Git (if not already)

```bash
git init
```

### 3. Add remote and push

```bash
git remote add origin https://github.com/kallelamine/Innvotra-NAMAA-Saudi-TTS.git
git add .
git commit -m "Initial commit: NAMAA Saudi TTS with Flask UI"
git branch -M main
git push -u origin main
```

### 4. Optional: Create `.env.example`

For others to know what to configure:

```
HF_token=your_huggingface_token_here
```

Do **not** commit `.env` — it contains your secret token.

---

## Model

- **Base model**: [NAMAA-Space/NAMAA-Saudi-TTS](https://huggingface.co/NAMAA-Space/NAMAA-Saudi-TTS)
- **Architecture**: Chatterbox Multilingual TTS (ResembleAI)
- **Output**: Saudi dialect Arabic speech (WAV)
- **License**: MIT

---

## Limitations

- Lack of tashkeel (diacritics) may affect pronunciation
- Numeric normalization may be improved in future versions
- GPU recommended for faster inference

---

## License

MIT License — see [NAMAA-Saudi-TTS](https://huggingface.co/NAMAA-Space/NAMAA-Saudi-TTS) for model license.

---

## Acknowledgments

- [NAMAA Community](https://huggingface.co/NAMAA-Space) — Saudi dialect TTS model
- [Resemble AI](https://github.com/resemble-ai/chatterbox) — Chatterbox TTS architecture
- [Hugging Face Space demo](https://huggingface.co/spaces/omarelshehy/NAMAA-Saudi-Voice) — Parameter reference
