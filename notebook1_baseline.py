"""
7043SCN - Task 1: LLM Fine-Tuning Study
Notebook 1: Baseline Evaluation (No Fine-Tuning)
Model: Flan-T5-Base
Task:  Medical Question Answering (MedQuAD / MedQA subset)
"""

# ── Commit: milestone1_commit ──────────────────────────────────────────────
# pip install -q transformers datasets evaluate rouge_score sacrebleu

import json
import time
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/flan-t5-base"
DATASET_NAME = "medalpaca/medical_meadow_medqa"   # HuggingFace medical QA
MAX_INPUT    = 256
MAX_TARGET   = 64
BATCH_SIZE   = 16
TEST_SAMPLES = 200          # keep small for quick baseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load tokeniser & model (zero-shot, no fine-tuning) ─────────────────────
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(f"Model loaded: {MODEL_NAME}")

# ── Load dataset ───────────────────────────────────────────────────────────
# ── Commit: milestone2_commit ──────────────────────────────────────────────
raw = load_dataset(DATASET_NAME, split="train")
raw = raw.shuffle(seed=42).select(range(TEST_SAMPLES))

def build_prompt(example):
    """Format as instruction-style prompt."""
    return f"Answer the medical question: {example['input']}"

# ── Batch inference ────────────────────────────────────────────────────────
# ── Commit: milestone3_commit ──────────────────────────────────────────────
rouge  = evaluate.load("rouge")
bleu   = evaluate.load("sacrebleu")

predictions = []
references  = []

start = time.time()
for i in range(0, len(raw), BATCH_SIZE):
    batch = raw[i : i + BATCH_SIZE]
    prompts  = [build_prompt({"input": q}) for q in batch["input"]]
    targets  = batch["output"]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT,
    ).to(DEVICE)

    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=MAX_TARGET,
            num_beams=4,
            early_stopping=True,
        )

    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    predictions.extend(decoded)
    references.extend(targets)
    print(f"  Processed {min(i+BATCH_SIZE, len(raw))}/{len(raw)}")

elapsed = time.time() - start
print(f"\nInference time: {elapsed:.1f}s")

# ── Metrics ────────────────────────────────────────────────────────────────
rouge_scores = rouge.compute(predictions=predictions, references=references)
bleu_score   = bleu.compute(
    predictions=predictions,
    references=[[r] for r in references]
)["score"]

results = {
    "stage":        "Baseline (Zero-Shot)",
    "model":        MODEL_NAME,
    "samples":      TEST_SAMPLES,
    "rouge1":       round(rouge_scores["rouge1"], 4),
    "rouge2":       round(rouge_scores["rouge2"], 4),
    "rougeL":       round(rouge_scores["rougeL"], 4),
    "bleu":         round(bleu_score, 2),
    "inference_s":  round(elapsed, 1),
}

print("\n=== BASELINE RESULTS ===")
for k, v in results.items():
    print(f"  {k}: {v}")

# ── Save ───────────────────────────────────────────────────────────────────
# ── Commit: final_commit ───────────────────────────────────────────────────
with open("results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("results/baseline_predictions.json", "w") as f:
    json.dump(
        [{"question": q, "prediction": p, "reference": r}
         for q, p, r in zip(raw["input"], predictions, references)],
        f, indent=2
    )

print("\nSaved to results/baseline_results.json")
