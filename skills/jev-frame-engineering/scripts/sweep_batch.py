#!/usr/bin/env python3
"""Compare N items per request across sizes, including position-quarter accuracy.
Run the reference form you intend to ship; flags are heuristics, not significance tests
or diagnoses of the cause of a change.

  sweep_batch.py --batch-template T.json --items dev.jsonl [--items val.jsonl]
                 [--sizes 1,4,8,12,16,20] [--policy policy.json] [--model M]

All item files are concatenated. Use tuning data while selecting a size and evaluate
the chosen configuration on a separate final holdout.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jevlib as J


def lost_items(best, rate, n):
    """Items lost against a best (rate, n): the rate drop over the smaller of the two labelled counts. A drop is
    only as credible as the smaller sample, so a one-item class can never lose two items, a failed batch that
    shrinks n is judged on its own count, and with equal counts this is exactly hits_best - hits."""
    return round((best[0] - rate) * min(best[1], n), 6)   # rounded so 3/22 of 22 counts as three items


def fold_best(front, pair):
    """Add a (rate, n) pair to the frontier of earlier sizes, dropping any pair it matches or beats on both rate
    and labelled count. A dropped pair can never lose more items than the pair that beat it, so nothing is lost."""
    if any(p[0] >= pair[0] and p[1] >= pair[1] for p in front):
        return front
    front[:] = sorted([p for p in front if not (pair[0] >= p[0] and pair[1] >= p[1])] + [pair], reverse=True)
    return front


def choice_drop(r, b):
    """Whether Choice report row r fell from the sizes seen so far in b, then fold r into b (start with {}).
    A drop is accuracy three items below some earlier size, mean confidence 0.05 below its best, or one option
    whose recall fell 0.25 from some earlier size while losing at least two items against that same size. Items
    lost are measured by lost_items, so a failed batch that shrinks n, or a size where only one item of a class
    was scored, cannot set off the flag on its own. Earlier sizes are kept as a frontier of (rate, n) pairs rather
    than one best, so a perfect score on a few items cannot hide a later fall from a large sample. Each option is
    measured against itself, so a large class collapsing is not compared with a small one."""
    acc = r["fine"] / r["n"]; classes = b.setdefault("classes", {})
    pc = {lab: c for lab, c in (r.get("per_class") or {}).items() if c["n"]}
    collapsed = [lab for lab, c in pc.items() if any(c["recall"] <= p[0] - 0.25 and lost_items(p, c["recall"], c["n"]) >= 2
                                                     for p in classes.get(lab, ()))]
    drop = bool(collapsed) or ("acc" in b and (any(lost_items(p, acc, r["n"]) >= 3 for p in b["acc"])
                                               or r["mean_conf"] <= b["conf"] - 0.05))
    fold_best(b.setdefault("acc", []), (acc, r["n"])); b["conf"] = max(b.get("conf", 0.0), r["mean_conf"])
    for lab, c in pc.items():
        fold_best(classes.setdefault(lab, []), (c["recall"], c["n"]))
    return drop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-template", required=True); ap.add_argument("--items", action="append", required=True)
    ap.add_argument("--sizes", default="1,4,8,12,16,20"); ap.add_argument("--policy"); ap.add_argument("--model")
    ap.add_argument("--sleep", type=float, default=0.05); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    tpl = J.load_json(a.batch_template); policy = J.load_json(a.policy) if a.policy else {}
    sizes = [int(s) for s in a.sizes.split(",") if s]
    if not sizes or any(n < 1 for n in sizes):
        ap.error("--sizes must contain positive integers")
    tq = tpl["questions"]; qids = list(tq)
    try:
        items = []
        for p in a.items:   # every label is checked against the template before the first billed request
            loaded = J.load_items(p); J.check_labels(loaded, tpl, p); items.extend(loaded)
        print(f"{len(items)} items; questions per item: {len(qids)} ({', '.join(qids)})")
        if any(isinstance(tq[q], dict) and tq[q].get("type") == "choice" for q in qids):   # only Choice cells print it
            print("low = the Choice option with the lowest recall, hits/labelled; an option's recall falling is flagged even when fine holds")
        # one column width per question: the widest cell its own type can print for this many items (every
        # count at its maximum, its longest option name, the largest MAE on its rubric) or the no-answers
        # note, measured rather than guessed so the columns after it stay aligned
        W = {q: max(len(J.cell(J.worst_case(tq[q], len(items), bool(policy.get(q, {}).get("parents"))))), len("(no scored answers)"))
             for q in qids}
        print(f"{'N':>3} {'q/req':>5} " + " ".join(f"{q[:W[q]]:<{W[q]}}" for q in qids)
              + f" {'Q1 acc':>6} {'Q4 acc':>6} {'Q4 conf':>7} {'tok/item':>8} {'ms/item':>7} {'requests':>8}")
        best = {}
        for n in sizes:
            res = J.run_batched(items, tpl, n, a.model, a.sleep); rep = J.score(res, tpl, policy)
            cells = []; flags = []
            for q in qids:
                r = rep["questions"].get(q)
                if not r:
                    cells.append(f"{'(no scored answers)':<{W[q]}}"); continue
                cells.append(J.cell(r).ljust(W[q]))
                if r["type"] == "choice":
                    if choice_drop(r, best.setdefault(q, {})):
                        flags.append(q)   # the per-class rule catches one option collapsing while fine holds
                elif r["type"] == "noul":
                    b = best.setdefault(q, {"fp": r["false_positives"], "lo": r["lowest_true"], "hi": r["highest_false"]})
                    if (r["false_positives"] > b["fp"]
                            or (r["lowest_true"] is not None and r["lowest_true"] <= (b["lo"] or 0) - 0.1)
                            or (r["highest_false"] is not None and r["highest_false"] >= (b["hi"] if b["hi"] is not None else 1) + 0.2)):
                        flags.append(q)   # a rising highest-false score is the early warning before false positives appear
                    b["fp"] = min(b["fp"], r["false_positives"]); b["lo"] = max(b["lo"] or 0, r["lowest_true"] or 0)
                    b["hi"] = min(b["hi"] if b["hi"] is not None else 1, r["highest_false"] if r["highest_false"] is not None else 1)
            q1 = q4 = c4 = None
            for q in qids:   # first Choice with a by-position readout supplies the columns
                bp = (rep["questions"].get(q) or {}).get("by_position")
                if bp:
                    q1, q4, c4 = bp[0]["acc"], bp[3]["acc"], bp[3]["mean_conf"]; break
            if q1 is not None and q4 is not None and q4 <= q1 - 0.15:
                flags.append("last-quarter")
            print(f"{n:>3} {J.fmt(rep['questions_per_request']):>5} " + " ".join(cells)
                  + f" {J.fmt(q1):>6} {J.fmt(q4):>6} {J.fmt(c4):>7} {J.fmt(rep['tokens_per_item'], 0):>8} {J.fmt(rep['ms_per_item'], 0):>7} {rep['requests']:>8}"
                  + f"  failed {rep['failed']}/{rep['items']}"
                  + (f"   <-- investigate (heuristic): {', '.join(flags)}" if flags else ""))
            if a.verbose:
                J.print_report(f"N={n}", rep, True)
    except J.JevError as e:
        print(f"error: {J.redact(str(e))}", file=sys.stderr); sys.exit(2)


if __name__ == "__main__":
    main()
