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

## Streamlit deployment

1. Push this folder to a GitHub repository.
2. Open Streamlit Community Cloud and create a new app from that repo.
3. Set main file path to streamlit_app.py.
4. Ensure requirements.txt from this folder is used.
5. Deploy.

### Local run command

streamlit run streamlit_app.py

### Optional environment variables

- BASE_MODEL_NAME (default: microsoft/deberta-v3-large)
- LOCAL_ADAPTER_DIR (default: ./mcq_lora_adapter)
- ADAPTER_REPO_ID (optional model adapter repo fallback)
- MAX_LENGTH (default: 256)

### Stability note

DeBERTa-v3-large is heavy for free CPU instances. If startup fails due to memory limits,
switch BASE_MODEL_NAME to a smaller backbone such as microsoft/deberta-v3-base and retrain
or provide a smaller adapter/backbone pair.
