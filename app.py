"""
Flask web app for NAMAA Saudi TTS.
Users can enter text, set parameters, generate speech, and listen to the output.
"""
import os
import uuid
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify, send_from_directory
from tts_saudi import initialize_model, generate_speech

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Global model (loaded on first use)
MODEL = None
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
OUTPUT_DIR = Path("Output_Voice")
OUTPUT_DIR.mkdir(exist_ok=True)

# Example Saudi dialect texts (from HF Space demo)
SAUDI_EXAMPLES = [
    "آبي أروح البقالة أشتري كم غرض وأرجع بسرعة.",
    "آبي أطلع مشوار خفيف وبأرجع قبل المغرب.",
    "ترى الموضوع بسيط، خلّصه اليوم وارتاح.",
    "أنا بالطريق الحين، باقي عشر دقايق وأوصل.",
    "خلّنا نبدأ اليوم، والباقي يجي مع الوقت.",
    "وش رايك نخلص الشغل اليوم ونرتاح بكرة؟",
    "إذا احتجت شي، كلّمني وأنا أجيك.",
    "الوضع تمام، الأمور ماشية مثل ما نبي.",
    "ياهلا ومرحبا فيكم، نرحّب فيكم أجمل ترحيب.",
]


def get_model():
    """Load model on first use."""
    global MODEL
    if MODEL is None:
        MODEL = initialize_model(device=DEVICE)
    return MODEL


@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html", examples=SAUDI_EXAMPLES)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate TTS audio from text."""
    try:
        data = request.get_json() or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Text is required"}), 400

        exaggeration = float(data.get("exaggeration", 0.5))
        temperature = float(data.get("temperature", 0.8))
        cfg_weight = float(data.get("cfg_weight", 0.5))
        seed = int(data.get("seed", 0))
        reference_path = data.get("reference_audio_path") or None

        model = get_model()
        filename = f"tts_{uuid.uuid4().hex[:12]}.wav"
        output_path = generate_speech(
            model,
            text,
            output_filename=filename,
            reference_audio_path=reference_path,
            exaggeration=exaggeration,
            temperature=temperature,
            seed=seed,
            cfg_weight=cfg_weight,
            device=DEVICE,
        )

        # Return relative URL for the audio file
        rel_path = Path(output_path).name
        return jsonify({
            "success": True,
            "audio_url": f"/audio/{rel_path}",
            "filename": rel_path,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    return send_from_directory(OUTPUT_DIR, filename, mimetype="audio/wav")


@app.route("/api/examples")
def api_examples():
    """Return list of example texts."""
    return jsonify({"examples": SAUDI_EXAMPLES})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
