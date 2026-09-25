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
    try:
        qset = J.load_json(a.questions); qs = J.questions_of(qset) if isinstance(qset, dict) else None
        model = a.model or (qset.get("model") if isinstance(qset, dict) else None)
        policy = J.load_json(a.policy) if a.policy else {}
        if not isinstance(qs, dict) or a.question not in qs:
            raise J.JevError(f"{a.questions} has no question {a.question!r}"
                             + (f"; its questions are {', '.join(qs)}" if isinstance(qs, dict) and qs else ""))
        base = qs[a.question]; asked = [f for f in a.fields.split(",") if f]
        # the target question alone, wrapped, so a question whose id is "questions" is not read as the wrapper
        one = lambda q: {"questions": {a.question: q}}
        # the question and every label are checked before the first billed request; only the target question is
        # sent, so only its labels are
        sets = dict(J.load_labelled(a.items, one(base)))
        # Only ablate fields the question actually has. A "minus examples" row for a question with no
        # examples repeats the full row and reads as a measured null result instead of an absent field.
        fields = [f for f in asked if json.dumps(strip(base, f), sort_keys=True) != json.dumps(base, sort_keys=True)]
        absent = [f for f in asked if f not in fields]
        variants = [("full", base)] + [(f"minus {f}", strip(base, f)) for f in fields]
        allq = base
        for f in fields:
            allq = strip(allq, f)
        variants.append(("minus all", allq))
        if absent:
            print(f"note: {a.question} has no " + ", ".join(absent) + " — those variants are skipped, not measured as null")
        if not fields:
            print("nothing to ablate: the question carries none of the requested fields"); return
        print(f"question {a.question}; sets: " + ", ".join(f"{os.path.basename(p)} ({len(v)})" for p, v in sets.items()))
        if base.get("type") == "choice":   # the legend describes a cell only a Choice question prints
            print("low = the Choice option with the lowest recall, hits/labelled; overall accuracy can hold while one class collapses")
        # column width: the widest cell this question can produce over these sets (every count at its maximum,
        # its longest option name, the largest MAE on its rubric, coarse only when the policy rolls it up) or
        # the no-answers note, measured rather than guessed so the tokens/item column stays aligned
        n_max = max(len(v) for v in sets.values())
        W = max(len(J.cell(J.worst_case(base, n_max, bool(policy.get(a.question, {}).get("parents")), a.question))),
                len(f"no scored answers (failed {n_max}/{n_max})"))
        print(f"{'variant':<18} " + "  ".join(f"{os.path.basename(p)[:12]:<{W}}" for p in sets) + "  tokens/item")
        for name, q in variants:
            cells = []; toks = []; notes = []
            for p, items in sets.items():
                res = J.run_single(items, one(q), model, a.sleep); rep = J.score(res, one(q), policy)
                r = rep["questions"].get(a.question)
                if rep["tokens_per_item"] is not None:   # a set that answered nothing has no cost to average in;
                    toks.append(rep["tokens_per_item"])  # counting it as 0 made the variant look cheaper than it is
                if not r:
                    cells.append(f"no scored answers (failed {rep['failed']}/{rep['items']})")
                else:
                    cells.append(J.cell(r))
                    if rep["failed"]:   # after the fixed-width columns, so a partial failure does not shift them
                        notes.append(f"{os.path.basename(p)} failed {rep['failed']}/{rep['items']}")
            print(f"{name:<18} " + "  ".join(f"{c:<{W}}" for c in cells)
                  + f"  {J.fmt(sum(toks) / len(toks) if toks else None, 0):>6}"
                  + (f"  {'; '.join(notes)}" if notes else ""))
    except J.JevError as e:
        print(f"error: {J.redact(str(e))}", file=sys.stderr); sys.exit(2)


if __name__ == "__main__":
    main()
