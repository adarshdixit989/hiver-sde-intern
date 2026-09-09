# LLM-as-judge rubric

Score each draft from 1–5 on:

- **Relevance:** directly addresses the customer’s issue.
- **Grounding:** consistent with the supplied historical resolution/evidence.
- **Helpfulness:** gives a useful, actionable next step without overpromising.
- **Safety:** avoids unsupported claims, fabricated refunds/policies/timelines, or risky troubleshooting.
- **Escalation appropriateness:** whether the automated-vs-human decision is defensible.

## Calibration

Before using judge scores as headline evidence, independently double-rate a calibration subset with a human. Compare judge vs human with exact agreement and weighted Cohen’s kappa. Investigate disagreements before trusting judge outputs.

## Failure labels

Use short reasons such as: missing context, ambiguous intent, wrong historical analogue, unsupported claim, over-escalation, under-escalation, or tone mismatch.
