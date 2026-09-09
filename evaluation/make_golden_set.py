import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    df["created_at"] = pd.to_datetime(df.created_at, errors="coerce")
    df = df.sort_values("created_at").drop_duplicates("customer_message")
    if len(df) < a.n:
        raise SystemExit(f"Only {len(df)} usable rows available; need {a.n}.")
    step = max(1, len(df) // a.n)
    out = df.iloc[::step].head(a.n).copy()
    for c in ["gold_intent", "gold_reply_quality", "gold_grounding", "gold_escalate", "gold_escalation_reason"]:
        out[c] = ""
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print("annotation queue:", len(out))

if __name__ == "__main__":
    main()
