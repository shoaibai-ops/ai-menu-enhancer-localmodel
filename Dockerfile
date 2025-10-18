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
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl \
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
# ⚡ (Optional) Pre-download Mistral model
# =====================================
# You can uncomment this block to speed up cold starts
# ARG HF_TOKEN
# RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('NousResearch/Nous-Hermes-2-Mistral-7B-DPO', token='$HF_TOKEN'); \
#     AutoModelForCausalLM.from_pretrained('NousResearch/Nous-Hermes-2-Mistral-7B-DPO', token='$HF_TOKEN')"

# =====================================
# 🌐 Expose Flask / RunPod port
# =====================================
EXPOSE 8080

# =====================================
# 🚀 Start RunPod Handler
# =====================================
CMD ["python3", "runpod_handler.py"]
