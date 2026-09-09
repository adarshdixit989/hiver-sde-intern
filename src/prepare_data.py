"""Extract customer -> support reply pairs for one TWCS support account."""
import argparse
import re
from pathlib import Path
import pandas as pd

REQUIRED = {"tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"}

def norm_id(x):
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s

def clean(x):
    x = re.sub(r"https?://\S+", " ", str(x))
    x = re.sub(r"@\w+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def iter_chunks(path, chunksize, max_rows=None):
    seen = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if max_rows is not None:
            remaining = max_rows - seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        seen += len(chunk)
        yield chunk

def main():
    ap = argparse.ArgumentParser(description="Extract customer -> support reply pairs from TWCS.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--brand", default="AppleSupport")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()

    support_by_parent = {}
    total = 0
    support_rows = 0
    for chunk in iter_chunks(args.input, args.chunksize, args.max_rows):
        total += len(chunk)
        missing = REQUIRED - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing TWCS columns: {sorted(missing)}")
        inbound = chunk["inbound"].astype(str).str.strip().str.lower().isin(["true", "1"])
        author = chunk["author_id"].astype(str).str.strip()
        support = chunk[(~inbound) & author.eq(args.brand)]
        support_rows += len(support)
        for r in support.itertuples(index=False):
            parent = norm_id(r.in_response_to_tweet_id)
            if parent and parent.lower() not in {"nan", "none"}:
                support_by_parent.setdefault(parent, str(r.text))

    records = []
    for chunk in iter_chunks(args.input, args.chunksize, args.max_rows):
        inbound = chunk["inbound"].astype(str).str.strip().str.lower().isin(["true", "1"])
        for r in chunk[inbound].itertuples(index=False):
            tid = norm_id(r.tweet_id)
            if tid not in support_by_parent:
                continue
            msg = clean(r.text)
            reply = clean(support_by_parent[tid])
            if len(msg) > 8 and len(reply) > 8:
                records.append({
                    "customer_tweet_id": tid,
                    "customer_message": msg,
                    "historical_reply": reply,
                    "created_at": getattr(r, "created_at", ""),
                })

    out = pd.DataFrame(records).drop_duplicates("customer_tweet_id")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"brand: {args.brand}")
    print(f"rows scanned: {total:,}")
    print(f"support rows indexed: {support_rows:,}")
    print(f"reply parents indexed: {len(support_by_parent):,}")
    print(f"pairs: {len(out):,}")
    print(f"output: {args.output}")

if __name__ == "__main__":
    main()
