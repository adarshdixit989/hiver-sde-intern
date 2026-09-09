# Hiver SDE Intern — AI Support Agent

A reproducible customer-support agent built on the Customer Support on Twitter (TWCS) dataset for one brand: AppleSupport.

## What it does

1. Classifies inbound AppleSupport customer messages into brand-specific intents.
2. Retrieves similar historical customer/support pairs to ground a draft response.
3. Decides whether to auto-handle or escalate, with an explicit reason.

## Dataset

Use the Kaggle Customer Support on Twitter dataset. Keep the raw `twcs.csv` local; it is intentionally excluded from Git.

Place it at:

```text
data/twcs.csv
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Reproduce the pipeline

Extract AppleSupport customer → support pairs:

```bash
python src/prepare_data.py --input data/twcs.csv --output data/apple_pairs.csv --brand AppleSupport
```

Train the intent model:

```bash
mkdir -p models
python src/train.py --input data/apple_pairs.csv --model-dir models
```

Smoke-test the agent:

```bash
python src/agent.py --model-dir models --message "My iPhone battery is draining after the latest iOS update"
```

Create the 200-row annotation queue:

```bash
python evaluation/make_golden_set.py --input data/apple_pairs.csv --output evaluation/golden_annotation.csv --n 200
```

Run evaluation after human verification of the golden labels:

```bash
python evaluation/evaluate.py --input data/apple_pairs.csv --gold evaluation/golden_annotation_completed.csv --model-dir models
```

## Evaluation honesty

The 200-example golden set is intended to be human-reviewed. AI-assisted preannotation is provided only to reduce clerical effort; it must not be described as independent human ground truth until verified by a person.

The headline number should be interpreted alongside per-intent performance, the escalation policy, reply-quality/grounding judgments, and known failure modes.

## Repository contents

- `src/prepare_data.py` — dataset filtering and conversation reconstruction
- `src/train.py` — bootstrap intent labels + TF-IDF/Logistic Regression classifier
- `src/agent.py` — classification, retrieval, grounded response template, escalation
- `evaluation/` — golden-set generation and evaluation harness
- `report/report.md` — report draft and methodology
- `decision_log.md` — non-obvious decisions and rationale

## Not included

The raw Kaggle dataset and generated large local artifacts are not committed to GitHub.
