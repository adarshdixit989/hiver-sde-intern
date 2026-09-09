# Hiver SDE Intern — Evaluation Report

## 1. Problem framing

I chose **AppleSupport** from the Customer Support on Twitter dataset. The goal is not to build a general customer-service chatbot; it is a narrow, auditable support assistant that (a) routes an incoming customer tweet into a small Apple-specific intent taxonomy, (b) retrieves a historical AppleSupport resolution, and (c) auto-handles only when confidence and historical evidence are strong enough.

I explicitly did not build fine-tuning, broad multi-brand support, long-term account actions, or a fully autonomous agent. Those would add surface area without improving the proof required by the take-home.

## 2. Data and sampling

The full TWCS CSV is kept local and is not committed to the repository. The preparation script scans the full dataset in chunks, indexes AppleSupport replies, and joins them to inbound customer tweets using `in_response_to_tweet_id`. The local run produced **103,832 AppleSupport customer→support pairs** from **2,811,774 rows scanned**.

The golden queue contains 200 examples. The official golden set must be human-reviewed; AI-assisted preannotation is only a clerical starting point and is not represented as independent human ground truth.

## 3. System

Pipeline: `customer tweet → TF-IDF/LogReg intent classifier → TF-IDF historical retrieval → evidence-backed draft → conservative escalation decision`.

Intents currently used are: `account_access`, `app_compatibility`, `battery_drain`, `ios_update_performance`, `other_software_issue`, and `wifi_connectivity`.

The escalation rule is intentionally conservative: escalate on low classifier confidence, weak historical similarity, or the fallback intent. The reply path uses a retrieved historical support response when evidence is strong; otherwise it asks for more information and hands off.

## 4. Baselines and results

Three systems should be compared on the **human-verified** golden set:

| System | Metric | Result |
|---|---|---|
| Majority-class baseline | Accuracy / Macro-F1 | Run from `evaluation/baselines.py` |
| TF-IDF + Logistic Regression | Accuracy / Macro-F1 | Run from `evaluation/baselines.py` |
| Support agent | Accuracy / Macro-F1 | Run from `evaluation/evaluate.py` |

Do not hard-code benchmark numbers. The repository is designed so headline metrics come from an actual run against the verified golden labels.

## 5. Reply evaluation and judge calibration

Reply quality should be rated separately from intent classification. The rubric scores relevance, grounding, helpfulness, safety, and escalation appropriateness on 1–5 scales. A calibration subset must be double-rated by a human before LLM-judge scores are treated as trustworthy. `evaluation/human_judge_agreement.py` reports weighted Cohen’s kappa and exact agreement.

## 6. Top failure modes

1. **Ambiguous short tweets:** tweets such as “fix this update” contain too little information for a precise intent.
2. **Multi-issue messages:** a tweet can mention battery, apps, Wi-Fi, and an iOS update simultaneously, but the current model chooses one primary class.
3. **Misleading nearest neighbours:** lexical overlap can retrieve a superficially similar historical case with a different root cause.
4. **Missing conversation state:** a single tweet can omit details that were present in earlier turns.
5. **Unsupported historical transfer:** an old response may contain context-specific wording that should not be copied into a new case.

## 7. What is misleading about my headline number?

Overall intent accuracy can look strong when common intents dominate. It can also be inflated by repeated templates or leakage between near-duplicate conversations. More importantly, accuracy says nothing about whether a reply is grounded or whether the system escalated risky cases. For that reason, macro-F1, per-intent F1, grounding/safety judgments, unsupported-claim rate, and escalation precision/recall are required companion metrics.

## 8. One more week

I would replace lexical retrieval with embedding retrieval, introduce calibrated confidence thresholds, use customer/time-aware splits to reduce leakage, add structured evidence extraction, expand human reply ratings, and run adversarial tests designed to trigger unsupported fixes or incorrect auto-handling.

## Sources

- TWCS dataset: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- AppleSupport/TWCS processing reference: https://sofiadutta.github.io/datascience-ipynbs/capstone/QABot.html
