import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--human-col", default="human_reply_quality")
    ap.add_argument("--judge-col", default="judge_reply_quality")
    args = ap.parse_args()
    d = pd.read_csv(args.input)
    d = d[d[args.human_col].notna() & d[args.judge_col].notna()].copy()
    if d.empty:
        raise SystemExit("No overlapping human/judge labels.")
    print("n:", len(d))
    print("weighted_kappa:", round(cohen_kappa_score(d[args.human_col], d[args.judge_col], weights="quadratic"), 4))
    print("exact_agreement:", round((d[args.human_col].astype(str) == d[args.judge_col].astype(str)).mean(), 4))

if __name__ == "__main__": main()
