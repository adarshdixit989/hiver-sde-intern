import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.gold)
    d = d[d.gold_intent.notna() & (d.gold_intent != "")].copy()
    if d.empty:
        raise SystemExit("No labelled rows found.")
    model = joblib.load(args.model)
    pred = model.predict(d.customer_message)
    print("accuracy:", round(accuracy_score(d.gold_intent, pred), 4))
    print("macro_f1:", round(f1_score(d.gold_intent, pred, average="macro"), 4))
    print(classification_report(d.gold_intent, pred, digits=3, zero_division=0))

if __name__ == "__main__":
    main()
