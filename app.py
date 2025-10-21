import os
import torch
import logging
from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator as Translator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Flask App
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Hugging Face Token
# ------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not found. Please set it in RunPod environment variables.")

# ------------------------------
# Model Setup
# ------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"
pipe = None  # Global pipeline variable

def load_model():
    """Load the DeepSeek model into memory with 4-bit quantization."""
    global pipe
    if pipe is None:
        logger.info("🚀 Loading DeepSeek-V3.1 with 4-bit quantization...")

        # 4-bit quantization configuration
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            use_auth_token=HF_TOKEN,
            quantization_config=quant_config,
            device_map="auto",
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
        )

        logger.info("✅ Model loaded successfully in 4-bit mode.")
    return pipe

# ------------------------------
# Tone Instructions & Fluff Words
# ------------------------------
TONE_INSTRUCTIONS = {
    "default": "Write in a balanced, natural restaurant style that is clear and appetizing.",
    "premium": "Write in a premium, high-end tone, highlighting exclusivity and top quality.",
}
FLUFF_WORDS = ["enjoy", "try", "savor", "delight", "experience"]

# ------------------------------
# Enhance Description
# ------------------------------
def enhance_description(original: str, tone: str = "premium", language: str = "en") -> str:
    """Enhance a restaurant menu description in the desired tone and language."""
    prompt = f"""
You are a restaurant branding expert. Rewrite the following text into a polished, appetizing menu description.

Original: {original}

Rules:
1. Write in {TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS['default'])}
2. Preserve all brand names and quantities exactly as given.
3. Keep quantities in digit form (e.g., 12 wings, 2 strips).
4. Limit the description to a maximum of 3 concise sentences.
5. Do not use these words: {", ".join(FLUFF_WORDS)}.
6. Output only the final menu description with no extra commentary.

Enhanced description:
"""
    try:
        pipe = load_model()
        outputs = pipe(prompt)
        result = outputs[0]["generated_text"].replace(prompt, "").strip()
        if language.lower() != "en":
            result = Translator(source="en", target=language).translate(result)
        return result
    except Exception as e:
        logger.error(f"❌ Error enhancing description: {e}", exc_info=True)
        return f"[Error: {e}]"

# ------------------------------
# Flask Endpoints
# ------------------------------
@app.route("/enhance", methods=["POST"])
def enhance_route():
    """API endpoint to enhance restaurant menu descriptions."""
    data = request.get_json()
    text = data.get("text")
    tone = data.get("tone", "premium")
    language = data.get("language", "en")

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    description = enhance_description(text, tone=tone, language=language)
    return jsonify({
        "input_text": text,
        "tone": tone,
        "language": language,
        "enhanced_description": description
    })

@app.route("/health", methods=["GET"])
def health_route():
    """Simple health check to confirm model is loaded."""
    if pipe is None:
        return jsonify({"status": "loading", "message": "Model is not loaded yet"}), 503
    return jsonify({"status": "ready", "message": "Model is loaded and ready"}), 200

# ------------------------------
# Preload Model at Startup
# ------------------------------
logger.info("💡 Preloading model at startup...")
load_model()
