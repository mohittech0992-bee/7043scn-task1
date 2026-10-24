"""
7043SCN - Task 1: LLM Fine-Tuning Study
Notebook 4: Comparison Table + Plots
Reads the three result JSON files and produces:
  - results/comparison_table.json
  - results/comparison_chart.png
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load results ───────────────────────────────────────────────────────────
# If you haven't run notebooks 1-3 yet, use the mock values below as placeholders.

def load_or_mock(path, mock):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"[WARN] {path} not found – using mock values for plotting")
    return mock

baseline = load_or_mock("results/baseline_results.json", {
    "stage": "Baseline (Zero-Shot)",
    "rouge1": 0.142, "rouge2": 0.031, "rougeL": 0.123, "bleu": 4.2,
    "trainable_pct": None, "train_time_s": 0,
})
fft = load_or_mock("results/fft_results.json", {
    "stage": "FFT",
    "rouge1": 0.381, "rouge2": 0.189, "rougeL": 0.347, "bleu": 21.5,
    "trainable_pct": 100.0, "train_time_s": 3200,
})
lora = load_or_mock("results/lora_results.json", {
    "stage": "LoRA (PEFT)",
    "rouge1": 0.362, "rouge2": 0.174, "rougeL": 0.331, "bleu": 19.8,
    "trainable_pct": 0.52, "train_time_s": 2900,
})

# ── Comparison table ───────────────────────────────────────────────────────
table = []
for r in [baseline, fft, lora]:
    table.append({
        "Stage":            r["stage"],
        "ROUGE-1":          r.get("rouge1", "N/A"),
        "ROUGE-2":          r.get("rouge2", "N/A"),
        "ROUGE-L":          r.get("rougeL", "N/A"),
        "BLEU":             r.get("bleu",   "N/A"),
        "Trainable %":      r.get("trainable_pct", "100%"),
        "Train Time (s)":   r.get("train_time_s", 0),
    })

with open("results/comparison_table.json", "w") as f:
    json.dump(table, f, indent=2)

print("=== COMPARISON TABLE ===")
for row in table:
    print(json.dumps(row))

# ── Bar chart ──────────────────────────────────────────────────────────────
stages   = [r["stage"] for r in [baseline, fft, lora]]
rouge1   = [r.get("rouge1", 0) for r in [baseline, fft, lora]]
rouge2   = [r.get("rouge2", 0) for r in [baseline, fft, lora]]
rougeL   = [r.get("rougeL", 0) for r in [baseline, fft, lora]]
bleu_sc  = [r.get("bleu",   0) / 100 for r in [baseline, fft, lora]]  # normalise 0–1

x     = np.arange(len(stages))
width = 0.18

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - 1.5*width, rouge1,  width, label="ROUGE-1",  color="#4C72B0")
ax.bar(x - 0.5*width, rouge2,  width, label="ROUGE-2",  color="#DD8452")
ax.bar(x + 0.5*width, rougeL,  width, label="ROUGE-L",  color="#55A868")
ax.bar(x + 1.5*width, bleu_sc, width, label="BLEU/100", color="#C44E52")

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel("Score (0–1)")
ax.set_title("Flan-T5-Base on Medical QA – Baseline vs FFT vs LoRA", fontsize=12)
ax.legend()
ax.set_ylim(0, 0.6)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/comparison_chart.png", dpi=150)
plt.close()
print("\nChart saved to results/comparison_chart.png")

# ── Learning curve (stub – fill with actual epoch logs) ───────────────────
fft_log  = [0.21, 0.31, 0.35]   # replace with trainer.state.log_history rougeL per epoch
lora_log = [0.20, 0.30, 0.33]

epochs = list(range(1, len(fft_log) + 1))
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(epochs, fft_log,  "o-", label="FFT",  color="#DD8452")
ax2.plot(epochs, lora_log, "s-", label="LoRA", color="#55A868")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("ROUGE-L (eval)")
ax2.set_title("Learning Curves – FFT vs LoRA")
ax2.legend()
ax2.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/learning_curves.png", dpi=150)
plt.close()
print("Learning curve saved to results/learning_curves.png")
