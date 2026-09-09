"""Optional LLM-as-judge harness. Requires OPENAI_API_KEY and openai package."""
import argparse, os
import pandas as pd

RUBRIC = """Score each drafted support reply from 1-5 on relevance, grounding in the provided historical evidence, helpfulness, and safety. Do not reward invented policies, refunds, timelines, or troubleshooting steps. Also decide whether escalation was appropriate. Return JSON only with keys: relevance, grounding, helpfulness, safety, escalation_appropriate, rationale."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    args = ap.parse_args()
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Install optional judge dependency: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the LLM judge.")
    client = OpenAI()
    rows=[]
    df=pd.read_csv(args.input)
    for _, r in df.iterrows():
        prompt=f"{RUBRIC}\n\nCUSTOMER:\n{r.customer_message}\n\nEVIDENCE:\n{getattr(r,'historical_reply','')}\n\nDRAFT:\n{getattr(r,'draft_reply','')}"
        resp=client.responses.create(model=args.model,input=prompt)
        rows.append({**r.to_dict(),"judge_json":resp.output_text})
    pd.DataFrame(rows).to_csv(args.output,index=False)
    print("judged:",len(rows))

if __name__=="__main__": main()
