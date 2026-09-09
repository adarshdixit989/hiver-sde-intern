# Hiver SDE Intern — AppleSupport AI Support Agent

A reproducible customer-support agent for the Hiver take-home, built on the **Customer Support on Twitter (TWCS)** dataset for one brand: **AppleSupport**.

## What it does

1. Classifies an inbound customer message into a small AppleSupport-specific intent taxonomy.
2. Retrieves a similar historical customer/support pair as evidence.
3. Drafts a grounded response or escalates to a human when confidence/evidence is weak.

## Dataset

Download the official TWCS dataset from Kaggle and place the raw file at:

```text
data/twcs.csv
```

The raw dataset and large generated artifacts are intentionally excluded from Git.

## Quickstart (Windows)

```bat
python -m pip install -r requirements.txt
python src\prepare_data.py --input data\twcs.csv --output data\apple_pairs.csv --brand AppleSupport
mkdir artifacts\model
python src\train.py --input data\apple_pairs.csv --model-dir artifacts\model
python src\agent.py --message "My iPhone battery is draining after the latest iOS update"
```

The local full-data run used **2,811,774 rows** and produced **103,832 AppleSupport customer→support pairs**.

## Golden evaluation set

The take-home requires 150–250 hand-labelled examples. Generate a 200-example annotation queue:

```bat
python evaluation\make_golden_set.py --input data\apple_pairs.csv --output evaluation\golden_annotation.csv --n 200
```

Fill `gold_intent`, `gold_reply_quality`, `gold_grounding`, `gold_escalate`, and `gold_escalation_reason` with human-reviewed labels before using the file as official ground truth.

AI-assisted preannotation may be used to reduce clerical effort, but it must not be described as independent human ground truth until reviewed.

## Baselines and evaluation

Trivial baseline: majority class.

Simple baseline: TF-IDF + Logistic Regression.

Primary system: TF-IDF/LogReg intent classifier + historical retrieval + conservative escalation.

```bat
python evaluation\baselines.py --gold evaluation\golden_annotation.csv
python evaluation\evaluate.py --gold evaluation\golden_annotation.csv --model artifacts\model\intent_model.joblib
```

Do not hard-code benchmark scores in the report. Record results from an actual run on the final human-verified golden set.

## LLM-as-judge

`evaluation/judge.py` is optional and requires an OpenAI API key and the `openai` package. Calibrate the judge on a human-rated subset before trusting its scores. Use `evaluation/human_judge_agreement.py` to report weighted Cohen’s kappa and exact agreement.

## Escalation policy

Auto-handle only when intent confidence is high, historical similarity is sufficiently strong, and the predicted intent is not the fallback class. Otherwise hand off rather than inventing a fix.

## Failure modes to inspect

- Ambiguous short messages
- Multiple issues in one tweet
- Lexically similar but causally different historical cases
- Missing prior-turn context
- Historical replies whose context does not transfer

## What is misleading about my headline number?

Overall accuracy can hide class imbalance, repeated templates, and leakage among near-duplicate examples. It also does not measure reply safety, grounding, or escalation quality. Report macro-F1, per-intent F1, reply-quality/grounding, unsupported-claim rate, and escalation precision/recall alongside accuracy.

## Sources

- TWCS dataset: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- AppleSupport/TWCS processing reference: https://sofiadutta.github.io/datascience-ipynbs/capstone/QABot.html

## Repository hygiene

Do not commit:
- `data/twcs.csv`
- `data/apple_pairs.csv`
- model/artifact directories
- API keys or `.env` files
