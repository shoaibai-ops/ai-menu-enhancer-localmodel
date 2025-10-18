import runpod
import logging
from datetime import datetime
from flask import Flask, jsonify
import threading

# ==========================================================
# 🧾 Logging Configuration
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger("runpod_handler")

# ==========================================================
# 💓 Flask Healthcheck Server
# ==========================================================
health_app = Flask(__name__)

@health_app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Nous-Hermes-2-Mistral-7B-DPO model ready!"
    })

def start_health_server():
    """Start the healthcheck Flask server in a background thread."""
    def run_server():
        health_app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

    threading.Thread(target=run_server, daemon=True).start()
    logger.info("✅ Healthcheck server started on port 8080.")

# ==========================================================
# 🚀 RunPod Handler Function
# ==========================================================
def handler(event):
    try:
        from app import generate_description  # Lazy import — avoids blocking cold start
        logger.info("📩 New request received")

        # Extract input safely
        data = event.get("input", {}) or {}
        original = data.get("input") or data.get("text", "")
        tone = data.get("tone", "premium")
        language = data.get("language", "en")

        if not original:
            logger.warning("⚠️ Missing 'input' or 'text' field in request.")
            return {"error": "Missing 'input' or 'text' field in request."}

        logger.info(f"🧠 Processing input: {original[:80]}...")
        logger.info(f"🎨 Tone: {tone} | 🌐 Language: {language}")

        # Generate enhanced description
        result = generate_description(original, tone, language)

        logger.info("✅ Successfully generated enhanced description.")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "input_text": original,
            "enhanced_description": result,
        }

    except Exception as e:
        logger.error(f"❌ Error in handler: {e}", exc_info=True)
        return {"error": str(e)}

# ==========================================================
# 🏁 Entry Point — RunPod Serverless
# ==========================================================
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod handler + healthcheck server...")
    start_health_server()
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logger.error(f"❌ Failed to start RunPod serverless handler: {e}", exc_info=True)
