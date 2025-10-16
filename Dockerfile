# =====================================
# 🧠 Base Image — lightweight Python + CUDA
# =====================================
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# =====================================
# 🧩 System Setup
# =====================================
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1  # ensures logs are streamed instantly

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =====================================
# 🧱 Copy Project Files
# =====================================
COPY . /app

# =====================================
# 🧰 Install Dependencies
# =====================================
RUN pip install --no-cache-dir -r requirements.txt

# =====================================
# ⚡ Optional: Pre-Download Model
# =====================================
# Skip pre-download during build; model will load dynamically at runtime
# This avoids long image build times and RunPod timeouts.
# To prefetch, you could uncomment and pass HF_TOKEN as build arg if needed.
# ARG HF_TOKEN
# RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3.1', token='$HF_TOKEN'); \
#     AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-V3.1', token='$HF_TOKEN')"

# =====================================
# 🌐 Expose Flask Healthcheck Port
# =====================================
EXPOSE 8080

# =====================================
# 🚀 Start RunPod Serverless Handler
# =====================================
CMD ["python3", "runpod_handler.py"]
