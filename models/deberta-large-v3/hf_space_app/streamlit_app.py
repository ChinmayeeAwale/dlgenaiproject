import os
from functools import lru_cache

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForMultipleChoice, AutoTokenizer

BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "microsoft/deberta-v3-large")
LOCAL_ADAPTER_DIR = os.getenv("LOCAL_ADAPTER_DIR", "./mcq_lora_adapter")
ADAPTER_REPO_ID = os.getenv("ADAPTER_REPO_ID", "")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_artifacts():
    device = pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForMultipleChoice.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if os.path.isdir(LOCAL_ADAPTER_DIR):
        adapter_source = LOCAL_ADAPTER_DIR
    elif ADAPTER_REPO_ID.strip():
        adapter_source = ADAPTER_REPO_ID.strip()
    else:
        raise FileNotFoundError(
            "Could not find LoRA adapter. Put adapter files in ./mcq_lora_adapter "
            "or set ADAPTER_REPO_ID to a valid Hugging Face repo id."
        )

    model = PeftModel.from_pretrained(base_model, adapter_source)
    model.eval()
    model.to(device)
    return tokenizer, model, device, adapter_source


def predict_top3(prompt: str, options: list[str]):
    tokenizer, model, device, _ = load_artifacts()

    prompt_list = [prompt] * 5
    encoded = tokenizer(
        prompt_list,
        options,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.unsqueeze(0).to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]

    probs = torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(probs, k=3)

    top3_prediction = " ".join(OPTION_LETTERS[idx] for idx in top_indices.tolist())
    ranked_rows = []
    for idx, score in zip(top_indices.tolist(), top_values.tolist()):
        ranked_rows.append((OPTION_LETTERS[idx], options[idx], float(score)))

    return top3_prediction, ranked_rows


def main():
    st.set_page_config(page_title="MCQ Solver DeBERTa LoRA", page_icon="📘", layout="wide")

    st.title("MCQ Solver: DeBERTa-v3-Large + LoRA")
    st.caption("Predict Top-3 options in competition format")

    with st.expander("Model and deployment info", expanded=False):
        st.write("Backbone:", BASE_MODEL_NAME)
        st.write("Local adapter path:", LOCAL_ADAPTER_DIR)
        st.write("Adapter fallback repo:", ADAPTER_REPO_ID if ADAPTER_REPO_ID else "Not set")
        st.write("Device:", pick_device())

    prompt = st.text_area("Prompt", height=140, placeholder="Enter the question stem")

    col1, col2 = st.columns(2)
    with col1:
        option_a = st.text_input("Option A")
        option_b = st.text_input("Option B")
        option_c = st.text_input("Option C")
    with col2:
        option_d = st.text_input("Option D")
        option_e = st.text_input("Option E")

    options = [option_a, option_b, option_c, option_d, option_e]

    if st.button("Predict Top-3", type="primary"):
        if not prompt.strip():
            st.error("Please provide a prompt.")
            return
        if any(not x.strip() for x in options):
            st.error("Please fill all five options A-E.")
            return

        try:
            with st.spinner("Running inference..."):
                top3, ranked = predict_top3(prompt, options)
        except Exception as exc:
            st.exception(exc)
            return

        st.success("Prediction complete")
        st.subheader("Top-3 Prediction")
        st.code(top3)

        st.subheader("Ranked choices")
        for label, text, score in ranked:
            st.write(f"{label}: {text}  |  confidence={score:.4f}")


if __name__ == "__main__":
    main()
