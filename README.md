# Introduction to DL and GenAI Project

This repository contains milestone notebooks, model experiments, and shared competition data for a multiple-choice question answering task.

## Student Details

- Name: Chinmayee Milind Awale
- Roll Number: 26DS2000018
- Email: 26ds2000018@ds.study.iitm.ac.in

## Problem Statement

Multiple choice question answering is a benchmark for reasoning, language understanding, and answer ranking. Each sample contains a prompt and five answer choices labeled A, B, C, D, and E. The task is to predict the top three most probable answers for every question.

The project is evaluated with Mean Average Precision at 3 (MAP@3), which rewards models that rank the correct option near the top of the prediction list.

## Repository Structure

The repository is organized to keep milestone work and model work separate while using one common dataset.

- `milestone/` contains milestone notebooks and submission artifacts.
- `models/` contains model experiments, baseline architectures, and fine-tuning outputs.
- `dataset/` contains the shared competition data used by both milestone and model workflows.

### Current Folder Layout

```text
dlgenaiproject/
|-- milestone/
|   |-- milestone-4.ipynb
|   |-- milestone-5.ipynb
|   |-- submission.csv
|-- models/
|   |-- deberta-large-v3/
|   |   |-- deberta-large-v3.ipynb
|   |   |-- results/
|   |   |   |-- mcq_lora_adapter/
|   |   |   |   |-- adapter_config.json
|   |   |   |   |-- adapter_model.safetensors
|   |   |   |   |-- README.md
|   |-- LSTM/
|   |   |-- LSTM_1v&5v.ipynb
|   |   |-- submission.csv
|   |   |-- lstm_confusion_matrix.png
|-- dataset/
|   |-- train.csv
|   |-- test.csv
|   |-- sample_submission.csv
|-- README.md
```

## Shared Dataset Policy

The same dataset is used across the milestone notebooks and the model notebooks.

- Training data: `dataset/train.csv`
- Test data: `dataset/test.csv`
- Sample submission: `dataset/sample_submission.csv`

Keeping one shared dataset directory avoids duplication and makes results easier to compare across experiments.

## Model Strategy

This project uses two different modeling approaches on purpose: one strong pretrained transformer for performance, and one from-scratch recurrent baseline for comparison.

### DeBERTa-v3-large with LoRA

The DeBERTa experiment uses the pretrained `microsoft/deberta-v3-large` backbone and adapts it with LoRA, stored under `models/deberta-large-v3/results/mcq_lora_adapter/`.

Why this model is used:

- DeBERTa-v3-large is a strong language understanding model, which makes it a good fit for MCQ ranking.
- The task needs contextual comparison between the prompt and answer choices, which transformer attention handles well.
- LoRA fine-tuning lets the model adapt to the competition without updating every parameter, which reduces memory use and training cost.

How it is written:

- The notebook uses a Hugging Face transformer multiple-choice setup rather than a custom classifier from scratch.
- The adapter configuration targets attention projections such as `query_proj` and `value_proj`, which is a parameter-efficient way to specialize the pretrained model.
- Only small adapter weights are trained, while the main backbone stays largely frozen, making the approach efficient and easier to reproduce.

This is a fine-tuning approach, not training from scratch.

### LSTM Baseline

The LSTM notebook under `models/LSTM/LSTM_1v&5v.ipynb` is a from-scratch baseline.

Why this model is used:

- It provides a lightweight comparison point against the transformer model.
- It is useful for checking how much performance comes from the architecture itself versus pretrained language knowledge.
- It is easier to run and debug than a large transformer, especially for baseline experimentation.

How it is written:

- The notebook builds its own vocabulary from the text data instead of relying on a pretrained tokenizer and encoder.
- It uses embeddings, a bidirectional LSTM encoder, pooling, and a scoring head to compare the prompt with each answer choice.
- The model is written from scratch so the architecture is fully controlled and transparent.
- This makes it a clean baseline for studying sequence modeling and answer ranking behavior.

This is a training-from-scratch approach, not a fine-tuned pretrained model.

## Workflow Recommendation

1. Keep milestone-specific work inside `milestone/`.
2. Keep model experiments and saved adapter results inside `models/`.
3. Reuse the shared dataset from `dataset/` for both workflows.
4. Use the DeBERTa experiment when the goal is stronger ranking performance.
5. Use the LSTM experiment when the goal is a simple, reproducible baseline.

## Evaluation Metric

- Primary metric: MAP@3
- Objective: rank the correct answer as high as possible in the top three predictions.

## Notes

- Milestone artifacts stay in `milestone/`.
- Model artifacts stay in `models/`.
- The dataset should remain in one common folder so the project stays consistent.
