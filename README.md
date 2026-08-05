# Introduction to DL and GenAI Project

This repository contains milestone submissions and model experiments for a multiple-choice question answering competition.

## Student Details

- Name: Chinmayee Milind Awale
- Roll Number: 26DS2000018
- Email: 26ds2000018@ds.study.iitm.ac.in

## Problem Statement

### Description

Multiple choice question answering remains an important benchmark for evaluating reasoning, language understanding, and answer ranking capabilities of modern AI systems. In many real-world scenarios, intelligent models must not only identify the correct answer but also rank alternative answers effectively based on confidence and contextual understanding.

In this competition, participants are provided with a collection of challenging MCQ-style questions. Each question includes a prompt along with five possible answer choices labeled A, B, C, D, and E. The task is to predict the top three most probable correct answers for every question.

The competition is designed to encourage experimentation with a variety of approaches including transformer-based models, retrieval-augmented systems, fine-tuned language models, prompt engineering, ensemble methods, and efficient inference pipelines.

Submissions are evaluated using Mean Average Precision at 3 (MAP@3), which rewards models that rank the correct answer higher in their predictions.

## Repository Organization

The repository is organized to keep milestone work and model work clearly separated.

- `milestone/` contains milestone notebooks and milestone submission artifacts.
- `models/` contains model-specific experiments, architecture-focused work, and adapter artifacts.
- `dataset/` contains shared competition data used by both milestone and model workflows.

This keeps progress tracking clean while ensuring model development remains modular.

## Shared Dataset Policy

The same dataset is used across both milestone and model workflows.

- Training data: `train.csv`
- Test data: `test.csv`
- Sample submission: `sample_submission.csv`

Current dataset location: `dataset/`

When running model experiments from `models/`, use this same dataset to keep results consistent and comparable.

## Current Folder Layout

```text
dlgenaiproject/
|-- milestone/
|   |-- milestone-4.ipynb
|   |-- milestone-5.ipynb
|   |-- submission.csv
|-- models/
|   |-- deberta_large_v3/
|   |   |-- deberta_v3_large.ipynb
|   |-- lora_adapter/
|   |   |-- adapter_config.json
|   |   |-- README.md
|   |   |-- 97_leaderboard_73/
|   |   |-- 99_leaderboard_75/
|   |-- LSTM/
|       |-- LSTM_1v&5v.ipynb
|-- dataset/
|   |-- train.csv
|   |-- test.csv
|   |-- sample_submission.csv
|-- README.md
```

## Workflow Recommendation

1. Perform milestone-related experiments and reporting inside `milestone/`.
2. Keep architecture/model-specific R&D inside `models/`.
3. Reuse the common dataset for both to maintain fair comparison.
4. Save generated outputs (for example submissions and adapters) in their respective milestone or model context folders.

## Evaluation Metric

- Primary metric: MAP@3
- Objective: include the correct answer as high as possible in the top-3 ranked predictions for each question.

## Notes

- Keep folder boundaries strict: milestone artifacts in `milestone/`, model artifacts in `models/`.
- Avoid duplicating datasets in multiple locations unless required for a specific reproducibility reason.
