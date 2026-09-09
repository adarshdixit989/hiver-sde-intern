import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.gold)
    d = d[d.gold_intent.notna() & (d.gold_intent != "")].copy()
    if len(d) < 20 or d.gold_intent.nunique() < 2:
        raise SystemExit("Need at least 20 labelled examples and 2 intents.")
    Xtr, Xte, ytr, yte = train_test_split(
        d.customer_message, d.gold_intent, test_size=0.25, random_state=42, stratify=d.gold_intent
    )
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)
    Xtr = vec.fit_transform(Xtr); Xte = vec.transform(Xte)
    clf = LogisticRegression(max_iter=1500, class_weight="balanced").fit(Xtr, ytr)
    pred = clf.predict(Xte)
    majority = ytr.value_counts().index[0]
    maj = [majority] * len(yte)
    print("baseline,accuracy,macro_f1")
    print(f"majority,{accuracy_score(yte, maj):.4f},{f1_score(yte, maj, average='macro'):.4f}")
    print(f"tfidf_logreg,{accuracy_score(yte, pred):.4f},{f1_score(yte, pred, average='macro'):.4f}")

if __name__ == "__main__":
    main()
