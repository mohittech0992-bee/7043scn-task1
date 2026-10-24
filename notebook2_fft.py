"""
7043SCN - Task 1: LLM Fine-Tuning Study
Notebook 2: Full Fine-Tuning (FFT)
Model: Flan-T5-Base
Task:  Medical Question Answering
"""

# pip install -q transformers datasets evaluate rouge_score sacrebleu accelerate

import json
import time
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/flan-t5-base"
DATASET_NAME = "medalpaca/medical_meadow_medqa"
MAX_INPUT    = 256
MAX_TARGET   = 64
TRAIN_SAMPLES = 2000
EVAL_SAMPLES  = 400
TEST_SAMPLES  = 200
EPOCHS        = 3
BATCH_SIZE    = 8
LR            = 5e-5
OUTPUT_DIR    = "results/fft_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load ───────────────────────────────────────────────────────────────────
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

raw = load_dataset(DATASET_NAME, split="train").shuffle(seed=42)
train_ds = raw.select(range(TRAIN_SAMPLES))
eval_ds  = raw.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + EVAL_SAMPLES))
test_ds  = raw.select(range(TRAIN_SAMPLES + EVAL_SAMPLES,
                            TRAIN_SAMPLES + EVAL_SAMPLES + TEST_SAMPLES))

# ── Tokenise ───────────────────────────────────────────────────────────────
def preprocess(batch):
    inputs  = [f"Answer the medical question: {q}" for q in batch["input"]]
    targets = batch["output"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=MAX_TARGET, truncation=True, padding="max_length"
    )
    # replace padding token id with -100 so loss ignores it
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in lab]
        for lab in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(preprocess,  batched=True, remove_columns=eval_ds.column_names)
test_tok  = test_ds.map(preprocess,  batched=True, remove_columns=test_ds.column_names)

# ── Training args ──────────────────────────────────────────────────────────
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    dec_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=dec_preds, references=dec_labels)
    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=100,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    fp16=(DEVICE == "cuda"),
    report_to="none",
)

collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Train ──────────────────────────────────────────────────────────────────
print("Starting FFT training...")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"Training time: {elapsed:.1f}s")

# ── Test evaluation ────────────────────────────────────────────────────────
bleu = evaluate.load("sacrebleu")

test_outputs = trainer.predict(test_tok, max_new_tokens=MAX_TARGET)
preds   = np.where(test_outputs.predictions != -100,
                   test_outputs.predictions,
                   tokenizer.pad_token_id)
dec_preds  = tokenizer.batch_decode(preds,      skip_special_tokens=True)
dec_labels = tokenizer.batch_decode(
    [[l for l in lab if l != -100] for lab in test_outputs.label_ids],
    skip_special_tokens=True
)

rouge_scores = rouge.compute(predictions=dec_preds, references=dec_labels)
bleu_score   = bleu.compute(
    predictions=dec_preds,
    references=[[r] for r in dec_labels]
)["score"]

results = {
    "stage":       "FFT (Full Fine-Tuning)",
    "model":       MODEL_NAME,
    "train_samples": TRAIN_SAMPLES,
    "test_samples":  TEST_SAMPLES,
    "epochs":      EPOCHS,
    "rouge1":      round(rouge_scores["rouge1"], 4),
    "rouge2":      round(rouge_scores["rouge2"], 4),
    "rougeL":      round(rouge_scores["rougeL"], 4),
    "bleu":        round(bleu_score, 2),
    "train_time_s": round(elapsed, 1),
}

print("\n=== FFT RESULTS ===")
for k, v in results.items():
    print(f"  {k}: {v}")

with open("results/fft_results.json", "w") as f:
    json.dump(results, f, indent=2)

trainer.save_model(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")
