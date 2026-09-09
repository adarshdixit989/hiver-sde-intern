import argparse, re
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RULES = [
    ("battery_drain", r"\bbattery\b|\bdrain\b|\bcharge\b|\bcharging\b"),
    ("wifi_connectivity", r"\bwifi\b|\bwi-fi\b|\bbluetooth\b|\bdisconnect"),
    ("ios_update_performance", r"\bios\b|\bupdate\b|\bslow\b|\bfreeze\b|\bcrash\b|\bfreez"),
    ("app_compatibility", r"\bapp\b|\bmusic\b|\bwhatsapp\b|\bapp store\b"),
    ("account_access", r"\blog ?in\b|\bpassword\b|\bapple id\b|\baccount\b"),
]

def bootstrap_label(text):
    text = str(text).lower()
    for name, pattern in RULES:
        if re.search(pattern, text):
            return name
    return "other_software_issue"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    if df.empty:
        raise SystemExit("No training pairs found. Check prepare_data output.")
    df["bootstrap_label"] = df.customer_message.map(bootstrap_label)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, max_features=50000)),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced")),
    ])
    pipe.fit(df.customer_message, df.bootstrap_label)
    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "intent_model.joblib")
    df.to_csv(out / "training_pairs.csv", index=False)
    print("trained:", len(df))
    print("classes:", sorted(df.bootstrap_label.unique()))

if __name__ == "__main__":
    main()
