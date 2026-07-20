# Finance RAG Assistant - Quick Start Guide

## 🎯 Project Overview
Fine-tune **Mistral-7B** on **FinQA** dataset using **QLoRA** for financial question-answering with RAG integration and Ollama deployment.

---

## 📋 Execution Checklist (7 Phases)

### ✅ Phase 0: Environment Setup (COMPLETED)
**Notebook**: `00_environment_setup.ipynb`
- ✓ All dependencies installed and verified
- ✓ GPU/CUDA status checked
- ✓ Project structure confirmed

**Status**: Ready to proceed

---

### 🔄 Phase 1: Dataset Exploration
**Notebook**: `01_dataset_exploration.ipynb`
- [ ] Load FinQA dataset (13K samples)
- [ ] Inspect dataset structure and quality
- [ ] Save locally for faster processing
- **Expected time**: 5-10 minutes (first run with download)
- **Next**: Run all cells in order

---

### 🔄 Phase 2: Data Preparation
**Notebook**: `02_data_preparation.ipynb`
- [ ] Load FinQA training data
- [ ] Format into instruction-following format
- [ ] Split: 80% train / 10% val / 10% test
- [ ] Save as JSONL files
- **Expected time**: 2-5 minutes
- **Output**: `../data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`

---

### 🔄 Phase 3-4: QLoRA Training (Pilot)
**Notebook**: `03_qlora_training_pilot.ipynb`
- [ ] Configure Mistral-7B with 4-bit quantization
- [ ] Apply LoRA adapters
- [ ] Load & tokenize data (100 samples for pilot)
- [ ] Run pilot training (1 epoch)
- **Expected time**: 5-15 minutes (GPU) / 30-60 min (CPU)
- **Output**: `../models/finance-adapter-pilot/`
- **W&B Tracking**: Real-time metrics at wandb.ai

**⚠️ CPU Warning**: Currently using CPU (CUDA not available). Training will be slow.

---

### 🔄 Phase 5: Model Evaluation
**Notebook**: `05_evaluation.ipynb`
- [ ] Load fine-tuned model + adapter
- [ ] Generate predictions on test set (10 samples)
- [ ] Compute ROUGE metrics
- [ ] Evaluate response quality
- **Expected time**: 5-10 minutes
- **Metrics**: ROUGE-1, ROUGE-L, response length

---

### 🔄 Phase 6: Export to Ollama
**Notebook**: `06_export_to_ollama.ipynb`
- [ ] Merge LoRA adapter with base model
- [ ] Convert to GGUF format (quantized)
- [ ] Export for Ollama deployment
- **Expected time**: 10-20 minutes
- **Output**: Model ready for `ollama run finance-rag`

---

### 🔄 Phase 7a: RAG Index Build
**Notebook**: `08_rag_index_build.ipynb`
- [ ] Load financial corpus from training data
- [ ] Generate embeddings (all-MiniLM-L6-v2)
- [ ] Build FAISS vector index
- **Expected time**: 5-15 minutes
- **Output**: `../data/corpus/faiss_index/financial_corpus.index`

---

### 🔄 Phase 7b: RAG Query & Grounding
**Notebook**: `09_rag_query_and_grounding.ipynb`
- [ ] Load FAISS index + fine-tuned LLM
- [ ] Test retrieval pipeline
- [ ] Generate responses with cited sources
- [ ] Validate end-to-end RAG pipeline
- **Expected time**: 5-10 minutes
- **Test queries**: 3 financial questions with ground truth

---

## 🚀 Quick Start (Recommended Order)

```bash
# 1. Download & explore dataset
jupyter notebook 01_dataset_exploration.ipynb
# Run all cells

# 2. Prepare data
jupyter notebook 02_data_preparation.ipynb
# Run all cells

# 3. Train pilot (validation run)
jupyter notebook 03_qlora_training_pilot.ipynb
# Run all cells (watch W&B dashboard)

# 4. Evaluate
jupyter notebook 05_evaluation.ipynb
# Run all cells

# 5. Build RAG index
jupyter notebook 08_rag_index_build.ipynb
# Run all cells

# 6. Test RAG pipeline
jupyter notebook 09_rag_query_and_grounding.ipynb
# Run all cells
```

---

## 📊 Configuration Summary

| Component | Setting | Details |
|-----------|---------|---------|
| **Base Model** | Mistral-7B | `mistralai/Mistral-7B` |
| **Dataset** | FinQA | 13K financial Q&A pairs |
| **Training Method** | QLoRA | 4-bit + LoRA (efficient) |
| **LoRA Rank** | 16 | Balance: capacity vs memory |
| **Batch Size** | 2 | Adjust if OOM |
| **Learning Rate** | 2e-4 | Standard for fine-tuning |
| **Epochs** | 1 (pilot) / 3 (full) | Prevent overfitting |
| **Embedding Model** | all-MiniLM-L6-v2 | 384-dim vectors |
| **Vector DB** | FAISS | L2 distance metric |

---

## ⚠️ Known Issues & Solutions

### Issue: CUDA not available
- **Cause**: PyTorch CPU version installed
- **Solution**: For GPU training, install `torch[cuda12.1]`
- **Impact**: Training will use CPU (very slow)

### Issue: Out of Memory (OOM)
- **Solution 1**: Reduce `per_device_train_batch_size` to 1
- **Solution 2**: Increase `gradient_accumulation_steps` to 8
- **Solution 3**: Use smaller model (Phi-2 instead of Mistral-7B)

### Issue: W&B not connected
- **Solution**: Run `wandb login` in terminal with your API key
- **Or**: Disable W&B: Set `report_to=[]` in TrainingArguments

---

## 📁 Project Structure

```
model_1/
├── notebooks/
│   ├── 00_environment_setup.ipynb        ✅ (COMPLETE)
│   ├── 01_dataset_exploration.ipynb      ⏳ (PENDING)
│   ├── 02_data_preparation.ipynb         ⏳ (PENDING)
│   ├── 03_qlora_training_pilot.ipynb     ⏳ (PENDING)
│   ├── 04_qlora_training_full.ipynb      📝 (OPTIONAL)
│   ├── 05_evaluation.ipynb               ⏳ (PENDING)
│   ├── 06_export_to_ollama.ipynb         ⏳ (PENDING)
│   ├── 07_public_corpus_ingestion.ipynb  📝 (ADVANCED)
│   ├── 08_rag_index_build.ipynb          ⏳ (PENDING)
│   └── 09_rag_query_and_grounding.ipynb  ⏳ (PENDING)
├── data/
│   ├── finqa/                            (raw dataset)
│   ├── processed/                        (formatted for training)
│   └── corpus/                           (RAG vector index)
├── models/
│   ├── finance-adapter-pilot/            (LoRA weights)
│   ├── finance-rag-merged/               (merged for Ollama)
│   └── logs/                             (training logs)
├── configs/
├── src/
├── reports/
├── requirements.txt
└── QUICKSTART.md (this file)
```

---

## 🔗 Next Steps After Pilot

1. **Full Training**: Run `04_qlora_training_full.ipynb` (3 epochs on full dataset)
2. **Export to Ollama**: Run `06_export_to_ollama.ipynb`
3. **Deploy Locally**:
   ```bash
   ollama create finance-rag -f Modelfile
   ollama run finance-rag
   ```
4. **Public Corpus** (optional): Run `07_public_corpus_ingestion.ipynb` to add SEC filings, earnings reports, etc.

---

## 📚 References

- **QLoRA Paper**: [Efficient Fine-Tuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **Mistral Docs**: https://docs.mistral.ai/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Weights & Biases**: https://wandb.ai/
- **Ollama**: https://ollama.ai/

---

## ✨ What You're Building

By completing this pipeline, you'll have:

✅ **Fine-tuned financial LLM** - Specialized for Q&A on financial documents  
✅ **RAG system** - Retrieves relevant financial context before generating answers  
✅ **Vector database** - Fast semantic search over financial corpus  
✅ **Local Ollama model** - Deploy without API costs  
✅ **Metrics & evaluation** - Understand model performance  

This is production-ready for:
- Financial chatbot
- Investment research assistant
- SEC filing analyzer
- Earnings call summarizer

---

**Happy training! 🚀**
