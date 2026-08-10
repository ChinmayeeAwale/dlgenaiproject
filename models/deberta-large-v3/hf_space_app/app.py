import os
from functools import lru_cache

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForMultipleChoice, AutoTokenizer

# Base backbone used during LoRA training.
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "microsoft/deberta-v3-large")
# Local adapter directory path inside the Space repo.
LOCAL_ADAPTER_DIR = os.getenv("LOCAL_ADAPTER_DIR", "./mcq_lora_adapter")
# Optional fallback adapter on the Hub, e.g. "username/repo".
ADAPTER_REPO_ID = os.getenv("ADAPTER_REPO_ID", "")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def load_artifacts():
    device = _pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForMultipleChoice.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    adapter_source = None
    if os.path.isdir(LOCAL_ADAPTER_DIR):
        adapter_source = LOCAL_ADAPTER_DIR
    elif ADAPTER_REPO_ID.strip():
        adapter_source = ADAPTER_REPO_ID.strip()
    else:
        raise FileNotFoundError(
            "Could not find LoRA adapter. Place adapter files in ./mcq_lora_adapter "
            "or set ADAPTER_REPO_ID to a valid Hugging Face repo id."
        )

    model = PeftModel.from_pretrained(base_model, adapter_source)
    model.eval()
    model.to(device)

    return tokenizer, model, device, adapter_source


def rank_choices(prompt: str, option_a: str, option_b: str, option_c: str, option_d: str, option_e: str):
    try:
        tokenizer, model, device, adapter_source = load_artifacts()
    except Exception as exc:
        return "Model load failed", f"Startup error: {exc}"

    options = [option_a, option_b, option_c, option_d, option_e]

    if not prompt.strip():
        return "Please provide a prompt.", ""

    if any(not choice.strip() for choice in options):
        return "Please fill all 5 options (A-E).", ""

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

    try:
        with torch.no_grad():
            logits = model(**encoded).logits[0]
    except Exception as exc:
        return "Inference failed", f"Inference error: {exc}"

    probs = torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(probs, k=3)

    ranked = []
    for idx, score in zip(top_indices.tolist(), top_values.tolist()):
        letter = OPTION_LETTERS[idx]
        text = options[idx]
        ranked.append(f"{letter}: {text} (confidence={score:.4f})")

    top3_prediction = " ".join(OPTION_LETTERS[idx] for idx in top_indices.tolist())
    details = (
        f"Adapter source: {adapter_source}\n"
        f"Backbone: {BASE_MODEL_NAME}\n"
        f"Top-3 label string (submission format): {top3_prediction}\n\n"
        "Ranked choices:\n" + "\n".join(ranked)
    )

    return top3_prediction, details


with gr.Blocks(title="MCQ Solver - DeBERTa v3 + LoRA") as demo:
    gr.Markdown(
        """
# MCQ Solver (DeBERTa-v3-Large + LoRA)
Enter one question prompt and five answer options. The app returns the top-3 prediction string in competition format.
"""
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Enter the question stem...")
            option_a = gr.Textbox(label="Option A")
            option_b = gr.Textbox(label="Option B")
            option_c = gr.Textbox(label="Option C")
            option_d = gr.Textbox(label="Option D")
            option_e = gr.Textbox(label="Option E")
            run_btn = gr.Button("Predict Top-3", variant="primary")

        with gr.Column():
            top3_output = gr.Textbox(label="Top-3 Prediction (A B C format)")
            details_output = gr.Textbox(label="Details", lines=12)

    run_btn.click(
        fn=rank_choices,
        inputs=[prompt, option_a, option_b, option_c, option_d, option_e],
        outputs=[top3_output, details_output],
        api_name="predict_top3",
    )

    gr.Examples(
        examples=[
            [
                "Which process in plants primarily converts light energy into chemical energy?",
                "Photosynthesis",
                "Transpiration",
                "Respiration",
                "Fermentation",
                "Diffusion",
            ],
            [
                "What is the SI unit of electric resistance?",
                "Volt",
                "Ohm",
                "Ampere",
                "Watt",
                "Coulomb",
            ],
        ],
        inputs=[prompt, option_a, option_b, option_c, option_d, option_e],
    )


if __name__ == "__main__":
    demo.queue(max_size=32, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
