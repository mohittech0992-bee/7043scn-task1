# 7043SCN Task 1 – LLM Fine-Tuning: Medical Question Answering

**Model:** Flan-T5-Base  
**Task:** Medical Question Answering (MedQA dataset)  
**Student ID:** [YOUR_STUDENT_ID]

## Method

| Stage | Description |
|-------|-------------|
| Baseline | Zero-shot inference, no training |
| FFT | Full Fine-Tuning – all 248M parameters updated |
| LoRA | Parameter-Efficient Fine-Tuning (rank=16, ~0.5% trainable params) |

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run baseline (zero-shot)
python notebook1_baseline.py

# 3. Run Full Fine-Tuning
python notebook2_fft.py

# 4. Run LoRA fine-tuning
python notebook3_lora.py

# 5. Generate comparison table + plots
python notebook4_compare.py
```

Results are saved to `results/`.

## Git commit log

Commits follow the required milestone naming:
- `initial_commit`
- `milestone1_commit` — model and dataset loaded
- `milestone2_commit` — baseline and FFT implemented
- `milestone3_commit` — LoRA implemented, preliminary results
- `final_commit` — comparison, paper, viva video

## Expected results

| Metric | Baseline | FFT | LoRA |
|--------|----------|-----|------|
| ROUGE-1 | ~0.14 | ~0.38 | ~0.36 |
| ROUGE-L | ~0.12 | ~0.35 | ~0.33 |
| BLEU    | ~4.2  | ~21.5 | ~19.8 |
| Trainable % | 0% | 100% | ~0.5% |

*Replace with your actual results after running.*

## Viva video

[LINK TO YOUR VIDEO]
