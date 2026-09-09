# Decision log

1. **One brand: AppleSupport.** It has a coherent support style and a large historical corpus in TWCS.
2. **Conservative escalation.** When intent confidence or historical evidence is weak, the system hands off instead of guessing.
3. **Data-derived intent taxonomy.** We use brand-specific intents rather than forcing Banking77 labels onto Apple support.
4. **TF-IDF + Logistic Regression baseline.** It is cheap, deterministic, interpretable, and appropriate for short noisy support messages.
5. **Historical retrieval.** Replies are grounded in prior AppleSupport resolutions and expose the selected evidence for auditability.
6. **No fine-tuning in v1.** Retrieval is easier to reproduce and inspect within the take-home time budget.
7. **Human golden set requirement is preserved.** Weak/AI-assisted labels are for bootstrapping only; official evaluation requires human verification.
8. **Deduplicate customer messages.** Prevents repeated templates from disproportionately affecting the golden queue.
9. **Reply quality is separate from intent accuracy.** Correct intent does not imply a safe or useful reply.
10. **Escalation is a first-class evaluation target.** A high auto-handle rate is not useful if unsafe cases are handled automatically.
11. **Raw TWCS data stays out of Git.** The repository contains code and a small sample rather than the large downloaded dataset.
12. **LLM judge must be calibrated against humans.** Agreement is measured before treating judge scores as trustworthy.
