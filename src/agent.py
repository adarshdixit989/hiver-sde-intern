import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--model-dir", default="artifacts/model")
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "intent_model.joblib")
    df = pd.read_csv(model_dir / "training_pairs.csv")
    msg = args.message
    probs = model.predict_proba([msg])[0]
    intent = model.classes_[probs.argmax()]
    conf = float(probs.max())
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(df.customer_message)
    q = vec.transform([msg])
    scores = cosine_similarity(q, X)[0]
    i = scores.argmax()
    sim = float(scores[i])
    ev = df.iloc[i]
    escalate = conf < 0.72 or sim < 0.18 or intent == "other_software_issue"
    if escalate:
        reason = "Low confidence or weak historical evidence; escalate rather than invent a fix."
        reply = ("Thanks for reaching out. We'd like to look into this with you. "
                 "Please send us a DM with your device model and iOS version so the support team can investigate.")
    else:
        reason = "High-confidence intent and a close historical resolution."
        reply = str(ev.historical_reply)
    print({
        "intent": intent,
        "confidence": round(conf, 3),
        "retrieval_similarity": round(sim, 3),
        "escalate": escalate,
        "reason": reason,
        "draft_reply": reply,
        "evidence_message": ev.customer_message,
        "evidence_reply": ev.historical_reply,
    })

if __name__ == "__main__":
    main()
