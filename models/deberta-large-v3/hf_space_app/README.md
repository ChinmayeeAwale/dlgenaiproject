---
title: MCQ Solver DeBERTa LoRA
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# MCQ Solver Space

This Space serves a DeBERTa-v3-large multiple-choice model with a LoRA adapter.

## Files required

- app.py
- requirements.txt
- mcq_lora_adapter/ (your saved adapter folder with adapter_config.json and adapter_model.safetensors)

## Optional environment variables

- BASE_MODEL_NAME: defaults to microsoft/deberta-v3-large
- LOCAL_ADAPTER_DIR: defaults to ./mcq_lora_adapter
- ADAPTER_REPO_ID: optional fallback Hugging Face repo id for adapter weights
- MAX_LENGTH: defaults to 256

## Quick deploy

1. Create a new Hugging Face Space (Gradio SDK).
2. Upload this folder contents.
3. Ensure mcq_lora_adapter is present (or set ADAPTER_REPO_ID).
4. Space starts automatically.
