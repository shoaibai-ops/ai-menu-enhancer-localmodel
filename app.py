import os
import torch
import logging
from deep_translator import GoogleTranslator as Translator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#
# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Hugging Face Token (from environment)
# ------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not found. Please set it in RunPod environment variables.")

# ------------------------------
# Model Setup (Lazy Load)
# ------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"
pipe = None  # Lazy initialization

def get_pipeline():
    """Load the DeepSeek model only once per container (lazy load)."""
    global pipe
    if pipe is None:
        logger.info("🚀 Loading DeepSeek-V3.1 model for the first time...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        logger.info("✅ Model loaded successfully.")
    return pipe

# ------------------------------
# Tone Instructions
# ------------------------------
TONE_INSTRUCTIONS = {
    "default": "Write in a balanced, natural restaurant style that is clear and appetizing.",
    "premium": "Write in a premium, high-end tone, highlighting exclusivity and top quality.",
}

# ------------------------------
# Words to Avoid
# ------------------------------
FLUFF_WORDS = ["enjoy", "try", "savor", "delight", "experience"]

# ------------------------------
# Main Function
# ------------------------------
def generate_description(original: str, tone: str = "premium", language: str = "en") -> str:
    """Generate a restaurant-style menu description."""
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
        pipe = get_pipeline()
        outputs = pipe(prompt, max_new_tokens=200, temperature=0.3, top_p=0.9)
        result = outputs[0]["generated_text"].replace(prompt, "").strip()

        # Optional translation if not English
        if language.lower() != "en":
            result = Translator(source="en", target=language).translate(result)

        return result

    except Exception as e:
        logger.error(f"❌ Error generating description: {e}", exc_info=True)
        return f"[Error: {e}]"
