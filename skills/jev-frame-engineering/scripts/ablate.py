#!/usr/bin/env python3
"""Ablate one question's criteria fields and report accuracy against tokens, per item set.

  ablate.py --questions Q.json --question QID --items dev.jsonl --items val.jsonl
            [--policy policy.json] [--fields not_for,examples,notes,inspect] [--model M]

Only the target question is sent, so tokens/item isolates that question's cost. Variants:
full, minus each field, minus all. Select on tuning data and report on an untouched holdout.
"""
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jevlib as J


def strip(q, field):
    q = copy.deepcopy(q)
    crit = q.get("criteria")
    if isinstance(crit, dict):
        for v in crit.values():
            if isinstance(v, dict):
                v.pop(field, None)
    elif isinstance(crit, list):
        for v in crit:
            if isinstance(v, dict):
                v.pop(field, None)
    if isinstance(q.get("instructions"), dict):
        q["instructions"].pop(field, None)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True); ap.add_argument("--question", required=True)
    ap.add_argument("--items", action="append", required=True); ap.add_argument("--policy")
    ap.add_argument("--fields", default="not_for,examples,notes,inspect"); ap.add_argument("--model"); ap.add_argument("--sleep", type=float, default=0.05)
    a = ap.parse_args()
    qset = J.load_json(a.questions); qs = J.questions_of(qset); model = a.model or (qset.get("model") if isinstance(qset, dict) else None)
    policy = J.load_json(a.policy) if a.policy else {}
    base = qs[a.question]; asked = [f for f in a.fields.split(",") if f]
    # Only ablate fields the question actually has. A "minus examples" row for a question with no
    # examples repeats the full row and reads as a measured null result instead of an absent field.
    fields = [f for f in asked if json.dumps(strip(base, f), sort_keys=True) != json.dumps(base, sort_keys=True)]
    absent = [f for f in asked if f not in fields]
    variants = [("full", base)] + [(f"minus {f}", strip(base, f)) for f in fields]
    allq = base
    for f in fields:
        allq = strip(allq, f)
    variants.append(("minus all", allq))
    sets = {p: J.load_items(p) for p in a.items}
    if absent:
        print(f"note: {a.question} has no " + ", ".join(absent) + " — those variants are skipped, not measured as null")
    if not fields:
        print("nothing to ablate: the question carries none of the requested fields"); return
    print(f"question {a.question}; sets: " + ", ".join(f"{os.path.basename(p)} ({len(v)})" for p, v in sets.items()))
    print(f"{'variant':<18} " + "  ".join(f"{os.path.basename(p)[:12]:<32}" for p in sets) + "  tokens/item")
    try:
        for name, q in variants:
            cells = []; toks = []
            for p, items in sets.items():
                res = J.run_single(items, {a.question: q}, model, a.sleep); rep = J.score(res, {a.question: q}, policy)
                r = rep["questions"].get(a.question, {}); toks.append(rep["tokens_per_item"] or 0)
                if not r:
                    cells.append(f"no scored answers (failed {rep['failed']}/{rep['items']})")
                elif r.get("type") == "choice":
                    cells.append(f"fine {r['fine']:>2}/{r['n']} " + (f"coarse {r['coarse']:>2} " if r["coarse"] is not None else "") + f"≥thr {r['pass']:>2}({r['pass_correct']:>2}) conf {r['mean_conf']:.2f}")
                elif r.get("type") == "noul":
                    cells.append(f"caught {r['caught']}/{r['positives']} fp {r['false_positives']}/{r['negatives']} margin {J.fmt(r['lowest_true'])}/{J.fmt(r['highest_false'])}")
                else:
                    cells.append(f"MAE {r.get('mae', 0):.2f}")
                if r and rep["failed"]:
                    cells[-1] += f" failed {rep['failed']}/{rep['items']}"
            print(f"{name:<18} " + "  ".join(f"{c:<32}" for c in cells) + f"  {sum(toks)/len(toks):6.0f}")
    except J.JevError as e:
        print(f"error: {J.redact(str(e))}", file=sys.stderr); sys.exit(2)


if __name__ == "__main__":
    main()
