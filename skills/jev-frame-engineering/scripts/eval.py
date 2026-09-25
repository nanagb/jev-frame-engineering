#!/usr/bin/env python3
"""Evaluate a Jev question set against labelled items.

  eval.py --questions Q.json --items dev.jsonl [--items val.jsonl] [--policy policy.json]
          [--batch-template T.json --batch-size N] [--model M] [--repeat K] [--sleep S]
          [--json out.json] [--verbose] [--permute-options]

Each --items file is reported separately (keep dev and validation apart). --repeat runs the
same set again and reports label agreement (run variation, not population uncertainty).
--permute-options (single mode) runs the set once more with every Choice's options listed in
reverse order and reports, per Choice, the label flips, the items that crossed the policy
threshold, and the mean |Δconf| against the first run to test option-order sensitivity.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jevlib as J


def reversed_options(qset):
    """A copy of the question set with every Choice's criteria in reverse order (values untouched)."""
    rev = json.loads(json.dumps(qset))
    for q in J.questions_of(rev).values():
        if q.get("type") == "choice" and isinstance(q.get("criteria"), dict):
            q["criteria"] = {k: q["criteria"][k] for k in reversed(list(q["criteria"]))}
    return rev


def report_permutation(base, perm, qset, policy):
    """Per Choice: label flips, items that crossed the policy threshold, mean |Δconf| between the
    original and reversed option order. This detects sensitivity, not its cause."""
    for qid, q in J.questions_of(qset).items():
        if q.get("type") != "choice":
            continue
        thr = J.threshold(policy, qid, "choice")
        pairs = [(x, x["answers"][qid], y["answers"][qid]) for x, y in zip(base, perm)
                 if x.get("answers") and y.get("answers") and qid in x["answers"] and qid in y["answers"]]
        if not pairs:
            continue
        flips = [(x["id"], p["choice"], r["choice"]) for x, p, r in pairs if p["choice"] != r["choice"]]
        crossed = sum(1 for _, p, r in pairs if (p["confidence"] >= thr) != (r["confidence"] >= thr))
        dconf = sum(abs(p["confidence"] - r["confidence"]) for _, p, r in pairs) / len(pairs)
        print(f"  option order [{qid}]: {len(flips)} label flips of {len(pairs)}, {crossed} crossed ≥{thr}, mean |Δconf| {dconf:.3f}")
        for i, x, y in flips[:10]:
            print(f"    {str(i)[:44]:<44} {x} -> {y}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", help="question set JSON (single-item mode)")
    ap.add_argument("--items", action="append", required=True, help="labelled JSONL; repeat for dev and val")
    ap.add_argument("--policy", help="thresholds and roll-up parents per question")
    ap.add_argument("--batch-template", help="batched mode: template with array_field, reference (keyed|quoted|index) and {ref} questions")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model"); ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.05); ap.add_argument("--json"); ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--permute-options", action="store_true", help="single mode: re-run with every Choice's options reversed and report flips and threshold crossings")
    a = ap.parse_args()
    if not a.questions and not a.batch_template:
        ap.error("give --questions (single mode) or --batch-template (batched mode)")
    if a.questions and a.batch_template:
        ap.error("choose --questions or --batch-template, not both")
    if a.repeat < 1 or a.batch_size < 1:
        ap.error("--repeat and --batch-size must be positive")
    if a.permute_options and a.batch_template:
        ap.error("--permute-options works in single mode (--questions)")
    dump = {}
    try:
        policy = J.load_json(a.policy) if a.policy else {}
        qset = J.load_json(a.batch_template) if a.batch_template else J.load_json(a.questions)
        if a.batch_template:
            J.check_template(qset)
        for path, items in J.load_labelled(a.items, qset):   # every file is checked before the first billed request
            runs = []
            for k in range(a.repeat):
                if a.batch_template:
                    res = J.run_batched(items, qset, a.batch_size, a.model, a.sleep)
                else:
                    res = J.run_single(items, qset, a.model, a.sleep)
                runs.append(res)
                title = f"{os.path.basename(path)}" + (f" run {k+1}" if a.repeat > 1 else "") + (f"  batch {a.batch_size}" if a.batch_template else "")
                J.print_report(title, J.score(res, qset, policy), a.verbose)
            if a.repeat > 1:
                for k, later in enumerate(runs[1:], 2):
                    print(f"  repeatability run1 vs run{k}: {json.dumps(J.agreement(runs[0], later, qset))}")
            if a.permute_options:
                rev = reversed_options(qset)
                res = J.run_single(items, rev, a.model, a.sleep)
                J.print_report(f"{os.path.basename(path)}  options reversed", J.score(res, rev, policy), a.verbose)
                report_permutation(runs[0], res, qset, policy)
                runs.append(res)
            dump[path] = runs
    except J.JevError as e:
        print(f"error: {J.redact(str(e))}", file=sys.stderr); sys.exit(2)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dump, f, indent=1)
        print(f"raw answers written to {a.json}")


if __name__ == "__main__":
    main()
