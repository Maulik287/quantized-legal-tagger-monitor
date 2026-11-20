# Quantized Legal Tagger Monitor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/kfms387hu8/badge.svg)](https://ask.thili.ai/b/kfms387hu8)

This project creates a "Quantized Legal Tagger Monitor" designed for legal aid workers. It involves fine-tuning a GPT-2 model for legal clause classification and then applying quantization to measure its impact on model size and simulated performance. The goal is to ensure efficient document processing on local systems by evaluating the trade-offs between model efficiency and tagging accuracy.

## What you are building

**Goal.** You will build a tool for legal aid workers to monitor the efficiency (size and inference speed) of a quantized legal clause tagger. This ensures the tagger runs effectively on local systems for document processing.

**Done when.** You have fine-tuned a GPT-2 model to classify legal clauses, applied dynamic and static post-training quantization, and presented a dashboard-like output comparing the original and quantized model sizes, simulated inference times, and tagging accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a subset of the LEDGAR dataset for legal clause tagging.**  
      Hint: Prepare your LEDGAR dataset subset, ensuring it is formatted correctly as input for the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill.
- [x] **2. Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.**  
      Hint: Apply the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` technique to train your GPT-2 model for legal clause classification.
- [x] **3. Evaluate the tagging accuracy of the fine-tuned model.**  
      Hint: Utilize the evaluation methods demonstrated in the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to assess your model's tagging accuracy.
- [ ] **4. Apply dynamic and static post-training quantization to the trained model.**  
      Hint: Implement the `Shrink a Neural Network with Quantisation` technique, specifically applying post-training quantization in both dynamic and static modes.
- [ ] **5. Compare the original and quantized model sizes and measure simulated inference times.**  
      Hint: Apply the measurement techniques from the `Shrink a Neural Network with Quantisation` drill to compare model sizes and simulate inference speeds.
- [ ] **6. Present a dashboard-like output of model efficiency metrics.**  
      Hint: Organize and present the efficiency metrics obtained from the `Shrink a Neural Network with Quantisation` drill in a clear, comparative format suitable for a dashboard.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Analyze the impact of quantization on the accuracy of specific legal clause types, providing a more granular view of performance trade-offs.
- Develop a more interactive dashboard that allows users to filter or compare different quantization configurations and their effects.
- Investigate additional evaluation metrics beyond accuracy, such as F1-score for individual clause tags, to provide a more nuanced assessment.
