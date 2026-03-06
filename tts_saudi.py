import os
import random
import numpy as np
import torch
import torch.serialization
import torchaudio as ta
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from chatterbox import mtl_tts
from pathlib import Path

load_dotenv()

if os.name == 'nt':
    os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

HF_TOKEN = os.getenv("HF_token", "").strip('"').strip("'")
if not HF_TOKEN:
    raise ValueError("HF_token not found in .env file")

OUTPUT_DIR = Path("Output_Voice")
OUTPUT_DIR.mkdir(exist_ok=True)

def initialize_model(device="cuda"):
    default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    cache_exists = os.path.exists(
        os.path.join(default_cache, "models--NAMAA-Space--NAMAA-Saudi-TTS", "snapshots")
    )

    if cache_exists:
        print("Loading model from cache...")
        try:
            ckpt_dir = snapshot_download(
                repo_id="NAMAA-Space/NAMAA-Saudi-TTS",
                repo_type="model",
                revision="main",
                token=HF_TOKEN if HF_TOKEN else None,
                local_files_only=True,
            )
        except Exception:
            print("Cache incomplete, downloading missing files...")
            ckpt_dir = snapshot_download(
                repo_id="NAMAA-Space/NAMAA-Saudi-TTS",
                repo_type="model",
                revision="main",
                token=HF_TOKEN if HF_TOKEN else None,
            )
    else:
        print("Downloading model from Hugging Face (one-time)...")
        ckpt_dir = snapshot_download(
            repo_id="NAMAA-Space/NAMAA-Saudi-TTS",
            repo_type="model",
            revision="main",
            token=HF_TOKEN if HF_TOKEN else None,
        )

    print("Loading model...")
    if device == "cpu":
        original_torch_load = torch.load
        original_serialization_load = torch.serialization.load
        
        def patched_load(f, *args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = torch.device('cpu')
            elif isinstance(kwargs.get('map_location'), str) and 'cuda' in kwargs['map_location'].lower():
                kwargs['map_location'] = torch.device('cpu')
            return original_torch_load(f, *args, **kwargs)
        
        torch.load = patched_load
        torch.serialization.load = patched_load
    
    try:
        model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)
    finally:
        if device == "cpu":
            torch.load = original_torch_load
            torch.serialization.load = original_serialization_load
    
    t3_state = load_safetensors(
        f"{ckpt_dir}/t3_mtl23ls_v2.safetensors",
        device=device
    )
    model.t3.load_state_dict(t3_state)
    model.t3.to(device).eval()
    
    print("Model loaded successfully!")
    return model

def set_seed(seed: int, device: str):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Seed value (0 = no seed set, use random)
        device: Device string ("cuda" or "cpu")
    """
    if seed and int(seed) != 0:
        torch.manual_seed(int(seed))
        if device == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))
        print(f"Random seed set to: {seed}")


def generate_speech(
    model, 
    text, 
    output_filename=None, 
    reference_audio_path=None,
    exaggeration=0.5,
    temperature=0.8,
    seed=0,
    cfg_weight=0.5,
    device="cuda"
):
    """
    Generate speech from text using NAMAA-Saudi-TTS model.
    This function matches the exact parameters from the Hugging Face Space demo:
    https://huggingface.co/spaces/omarelshehy/NAMAA-Saudi-Voice
    
    Args:
        model: Loaded ChatterboxMultilingualTTS model
        text: Arabic text to synthesize (max 300 characters, will be truncated)
        output_filename: Optional output filename (without extension)
        reference_audio_path: Optional path to reference audio for voice cloning/style transfer
                            If None, uses default reference audio from the model
        exaggeration: Emotion/expressiveness control (default: 0.5, range: 0.25-2.0)
                     - Lower values = more neutral/calm
                     - Higher values = more expressive/emotional
        temperature: Controls randomness/creativity (default: 0.8, range: 0.05-5.0)
                    - Lower = more deterministic, consistent
                    - Higher = more varied, creative
        seed: Random seed for reproducibility (default: 0 = random)
             - Set to any integer > 0 for reproducible results
        cfg_weight: Classifier-free guidance weight / Pace control (default: 0.5, range: 0.0-1.0)
                  - Controls how much the model follows the conditioning
                  - Higher values may affect pacing
    
    Returns:
        Path to saved audio file
    """
    # Set seed if specified
    set_seed(seed, device)
    
    # Truncate text to 300 characters (matching the demo)
    text = text[:300].strip()
    if not text:
        raise ValueError("Text cannot be empty")
    
    print(f"Generating speech for text: {text[:50]}...")
    print(f"Parameters: exaggeration={exaggeration}, temperature={temperature}, cfg_weight={cfg_weight}, seed={seed}")
    
    # Prepare generation kwargs (matching the demo exactly)
    generate_kwargs = {
        "exaggeration": float(exaggeration),
        "temperature": float(temperature),
        "cfg_weight": float(cfg_weight),
    }
    
    # Use reference audio if provided, otherwise model uses default
    if reference_audio_path and os.path.exists(reference_audio_path):
        print(f"Using reference audio for voice/style transfer: {reference_audio_path}")
        generate_kwargs["audio_prompt_path"] = reference_audio_path
    
    # Generate speech (matching the demo exactly)
    wav = model.generate(
        text,
        language_id="ar",
        **generate_kwargs,
    )
    
    # Generate output filename
    if output_filename is None:
        safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_text = safe_text.replace(' ', '_')
        output_filename = f"saudi_tts_{safe_text}" if safe_text else "saudi_tts_output"
    
    if not output_filename.endswith('.wav'):
        output_filename = output_filename.replace('.mp3', '') + '.wav'

    output_path = OUTPUT_DIR / output_filename
    ta.save(str(output_path), wav, model.sr)
    print(f"Audio saved to: {output_path}")
    return str(output_path)

def main():
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device")
    else:
        device = "cpu"
        print("Using CPU device (CUDA not available)")
    
    try:
        model = initialize_model(device=device)
        
        # Example text from the Hugging Face Space demo
        text = "آبي أروح البقالة أشتري كم غرض وأرجع بسرعة."
        print(f"Generating speech for text:\n{text}\n")
        
        # Example 1: Default parameters (matching the Hugging Face Space demo)
        print("\n=== Example 1: Default parameters (matching HF Space demo) ===")
        output_path = generate_speech(
            model, 
            text,
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.5,
            seed=0,
            device=device
        )
        print(f"✅ Default audio saved to: {output_path}")
        
        # Example 2: More expressive voice
        print("\n=== Example 2: More expressive voice (exaggeration=1.0) ===")
        output_path_expressive = generate_speech(
            model, 
            text,
            exaggeration=1.0,
            temperature=0.8,
            cfg_weight=0.5,
            seed=0,
            device=device,
            output_filename="saudi_tts_expressive"
        )
        print(f"✅ Expressive audio saved to: {output_path_expressive}")
        
        # Example 3: Lower temperature (more deterministic)
        print("\n=== Example 3: Lower temperature (more deterministic) ===")
        output_path_deterministic = generate_speech(
            model, 
            text,
            exaggeration=0.5,
            temperature=0.3,
            cfg_weight=0.5,
            seed=42,  # Fixed seed for reproducibility
            device=device,
            output_filename="saudi_tts_deterministic"
        )
        print(f"✅ Deterministic audio saved to: {output_path_deterministic}")
        
        # Example 4: Higher CFG weight
        print("\n=== Example 4: Higher CFG weight (different pacing) ===")
        output_path_cfg = generate_speech(
            model, 
            text,
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.8,
            seed=0,
            device=device,
            output_filename="saudi_tts_high_cfg"
        )
        print(f"✅ High CFG audio saved to: {output_path_cfg}")
        
        # Example 5: Using reference audio for voice cloning
        # Uncomment and provide a path to a reference audio file:
        # reference_audio = "path/to/your/reference_audio.wav"  # or .flac
        # if os.path.exists(reference_audio):
        #     print("\n=== Example 5: Voice cloning with reference audio ===")
        #     output_path_cloned = generate_speech(
        #         model, 
        #         text,
        #         reference_audio_path=reference_audio,
        #         exaggeration=0.5,
        #         temperature=0.8,
        #         cfg_weight=0.5,
        #         seed=0,
        #         device=device,
        #         output_filename="saudi_tts_cloned"
        #     )
        #     print(f"✅ Cloned voice audio saved to: {output_path_cloned}")
        
        print("\n" + "="*70)
        print("PARAMETERS (Matching Hugging Face Space Demo)")
        print("="*70)
        print("""
Available Parameters (same as https://huggingface.co/spaces/omarelshehy/NAMAA-Saudi-Voice):

1. exaggeration (default: 0.5, range: 0.25-2.0)
   - Controls emotion/expressiveness
   - Lower = more neutral/calm
   - Higher = more expressive/emotional
   Example: generate_speech(model, text, exaggeration=1.0)

2. temperature (default: 0.8, range: 0.05-5.0)
   - Controls randomness/creativity
   - Lower = more deterministic, consistent
   - Higher = more varied, creative
   Example: generate_speech(model, text, temperature=0.3)

3. cfg_weight (default: 0.5, range: 0.0-1.0)
   - Classifier-free guidance weight / Pace control
   - Controls how much the model follows conditioning
   Example: generate_speech(model, text, cfg_weight=0.8)

4. seed (default: 0 = random)
   - Random seed for reproducibility
   - Set to any integer > 0 for reproducible results
   Example: generate_speech(model, text, seed=42)

5. reference_audio_path (optional)
   - Path to reference audio file for voice cloning/style transfer
   - If not provided, uses default reference audio
   - Recommended: 3-10 second audio sample in Saudi dialect
   Example: generate_speech(model, text, reference_audio_path="voice_sample.wav")

Note: Text is automatically truncated to 300 characters (matching the demo).
        """)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
