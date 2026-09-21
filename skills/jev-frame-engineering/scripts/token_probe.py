#!/usr/bin/env python3
"""Probe usage: fitted baseline, incremental minimal-question cost, and character
ratios on synthetic prose and criteria. These are not universal billing constants.

  token_probe.py [--model M]

Re-run after model changes and confirm estimates on representative real requests.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jevlib as J


def toks(state, qs, model):
    d, _ = J.post(state, qs, model); return d["usage"]["input_tokens"], d.get("model")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=J.DEFAULT_MODEL); a = ap.parse_args()
    try:
        print("A. same one-word state, k identical minimal Nouls")
        rows = []
        for k in (1, 2, 4, 8, 16):
            qs = {f"q{i}": {"type": "noul", "instructions": "Is `items[0]` empty?"} for i in range(k)}
            t, m = toks({"items": ["a"]}, qs, a.model); rows.append((k, t)); print(f"   {k:>2} questions -> {t:>5} tokens")
        per_q = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0]); fixed = rows[0][1] - per_q
        print(f"   => ~{per_q:.0f} tokens per minimal question; fitted baseline ~{fixed:.0f} tokens (includes test state/serialization; model {m})")

        print("B. one minimal Noul, prose state of growing length")
        sent = "The consumer stops pulling messages after the broker restarts and jobs pile up until someone restarts the process. "
        q = {"q": {"type": "noul", "instructions": "Does `text` describe a failure?"}}; rows = []
        for reps in (1, 5, 20, 50):
            chars = len(sent) * reps; t, _ = toks({"text": sent * reps}, q, a.model); rows.append((chars, t)); print(f"   {chars:>6} chars -> {t:>5} tokens")
        slope = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])
        print(f"   => prose: {1/slope:.1f} chars per token at the margin")

        print("C. one Choice with 6 options, criteria text of growing length")
        rows = []
        for reps in (1, 10, 40):
            crit = {k: ("ready to buy or order now, names a purchase action " * reps).strip() for k in ("a", "b", "c", "d", "e", "f")}
            chars = sum(len(v) for v in crit.values())
            t, _ = toks({"text": "hello"}, {"q": {"type": "choice", "instructions": "Classify `text`.", "criteria": crit}}, a.model)
            rows.append((chars, t)); print(f"   {chars:>6} chars -> {t:>5} tokens")
        slope = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])
        print(f"   => criteria text: {1/slope:.1f} chars per token at the margin")
        print("\nThese text slopes are not interchangeable or exact request estimates; use reported usage.")
        print(f"Reference price for {J.DEFAULT_MODEL}: ${J.PRICE_PER_MTOK}/M input tokens; check the models page for the model being probed.")
    except J.JevError as e:
        print(f"error: {J.redact(str(e))}", file=sys.stderr); sys.exit(2)


if __name__ == "__main__":
    main()
