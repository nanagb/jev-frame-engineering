#!/usr/bin/env python3
"""Compare N items per request across sizes, including position-quarter accuracy.
Run the reference form you intend to ship; flags are heuristics, not significance tests
or diagnoses of the cause of a change. Each size is compared with every earlier size on the
items both of them scored, so a failed batch neither raises nor hides a flag, and a flag
names what fell: a Choice's fine accuracy, confidence or one option's recall; a Noul's false
positives, lowest true or highest false score; or last-quarter accuracy within the size.

  sweep_batch.py --batch-template T.json --items dev.jsonl [--items val.jsonl]
                 [--sizes 1,4,8,12,16,20] [--policy policy.json] [--model M]

All item files are concatenated. Use tuning data while selecting a size and evaluate
the chosen configuration on a separate final holdout.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jevlib as J


def outcomes(res, qid, q):
    """One question's scored items at one batch size, keyed by position in the sweep (run_batched returns one row
    per item, in order, and ids can repeat across --items files): (label, hit, confidence) for a Choice and
    (label, p) for a Noul. Failed items and items the files leave unlabelled are absent, so two sizes are compared
    only on the items both of them scored."""
    out = {}
    for i, r in enumerate(res):
        a = (r.get("answers") or {}).get(qid)
        if a is None or qid not in r["expected"] or not J.valid_label(q, r["expected"][qid]):
            continue
        lab = r["expected"][qid]
        out[i] = (lab, a["choice"] == lab, a["confidence"]) if q["type"] == "choice" else (lab, a["noul"])
    return out


def choice_drop(now, earlier):
    """What fell for a Choice at this size against any earlier size, as the flag's reasons ([] when nothing did):
    'fine' when three more items are wrong, 'confidence' when mean confidence is 0.05 lower, and '<option> recall'
    when an option loses at least two items and at least a quarter of its items. now and each earlier entry come
    from outcomes(), and each pair of sizes is compared only on the items both scored: a failed batch can neither
    raise a flag (the answers it lost are compared with nothing) nor hide one, and a one-item class cannot lose two
    items. Each option is measured against itself, so a large class collapsing is not averaged away."""
    fine = conf = False; classes = set()
    for before in earlier:
        common = before.keys() & now.keys()
        if not common:
            continue
        fine = fine or sum(before[i][1] - now[i][1] for i in common) >= 3
        # a float mean is rounded before the comparison, so 0.95 -> 0.90 is the 0.05 drop it looks like
        conf = conf or round(sum(before[i][2] - now[i][2] for i in common) / len(common), 9) >= 0.05
        per = {}   # option -> (items, hits lost), counted in whole items so 4/12 -> 1/12 is exactly a quarter
        for i in common:
            n, lost = per.get(before[i][0], (0, 0)); per[before[i][0]] = (n + 1, lost + before[i][1] - now[i][1])
        classes.update(lab for lab, (n, lost) in per.items() if lost >= 2 and 4 * lost >= n)
    return (["fine"] if fine else []) + (["confidence"] if conf else []) + [f"{lab} recall" for lab in sorted(classes)]


def noul_drop(now, earlier, thr):
    """What fell for a Noul at this size against any earlier size, as the flag's reasons: 'false positives' when
    more negatives reach the threshold thr, 'lowest true' when the lowest positive score falls by 0.1, and
    'highest false' when the highest negative score rises by 0.2, the early warning before false positives
    appear. Each pair of sizes is compared on the items both scored, as in choice_drop."""
    fp = low = high = False
    for before in earlier:
        common = before.keys() & now.keys()
        pos = [i for i in common if before[i][0] is True]; neg = [i for i in common if before[i][0] is False]
        fp = fp or sum(now[i][1] >= thr for i in neg) > sum(before[i][1] >= thr for i in neg)
        if pos:
            low = low or round(min(before[i][1] for i in pos) - min(now[i][1] for i in pos), 9) >= 0.1
        if neg:
            high = high or round(max(now[i][1] for i in neg) - max(before[i][1] for i in neg), 9) >= 0.2
    return (["false positives"] if fp else []) + (["lowest true"] if low else []) + (["highest false"] if high else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-template", required=True); ap.add_argument("--items", action="append", required=True)
    ap.add_argument("--sizes", default="1,4,8,12,16,20"); ap.add_argument("--policy"); ap.add_argument("--model")
    ap.add_argument("--sleep", type=float, default=0.05); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    try:
        sizes = [int(s) for s in a.sizes.split(",") if s]
    except ValueError:
        sizes = []
    if not sizes or any(n < 1 for n in sizes):
        ap.error("--sizes must contain positive integers")
    try:
        tpl = J.load_json(a.batch_template); policy = J.load_json(a.policy) if a.policy else {}
        J.check_template(tpl)
        # every file is concatenated into one sweep, after the template and every label are checked
        items = [it for _, loaded in J.load_labelled(a.items, tpl) for it in loaded]
        tq = tpl["questions"]; qids = list(tq)
        print(f"{len(items)} items; questions per item: {len(qids)} ({', '.join(qids)})")
        if any(tq[q]["type"] == "choice" for q in qids):   # only Choice cells print it
            print("low = the Choice option with the lowest recall, hits/labelled; a flag names what fell against an earlier size,"
                  " compared on the items both sizes scored")
        # one column width per question: the widest cell its own type can print for this many items (every
        # count at its maximum, its longest option name, the largest MAE on its rubric), the no-answers note or
        # the whole question id, in terminal columns, measured rather than guessed so the columns after it stay
        # aligned and two ids that share a prefix stay distinguishable
        W = {q: max(J.width(J.cell(J.worst_case(tq[q], len(items), bool(policy.get(q, {}).get("parents")), q))),
                    J.width("(no scored answers)"), J.width(q))
             for q in qids}
        print(f"{'N':>3} {'q/req':>5} " + " ".join(J.pad(q, W[q]) for q in qids)
              + f" {'Q1 acc':>6} {'Q4 acc':>6} {'Q4 conf':>7} {'tok/item':>8} {'ms/item':>7} {'requests':>8}")
        earlier = {q: [] for q in qids}   # each question's outcomes() at every earlier size
        for n in sizes:
            res = J.run_batched(items, tpl, n, a.model, a.sleep); rep = J.score(res, tpl, policy)
            cells = []; flags = []
            for q in qids:
                r = rep["questions"].get(q)
                cells.append(J.pad(J.cell(r) if r else "(no scored answers)", W[q]))
                typ = tq[q]["type"]
                if typ in ("choice", "noul"):
                    now = outcomes(res, q, tq[q])
                    why = (choice_drop(now, earlier[q]) if typ == "choice"
                           else noul_drop(now, earlier[q], J.threshold(policy, q, "noul")))
                    if why:
                        flags.append(f"{q} ({', '.join(why)})")
                    earlier[q].append(now)
            q1 = q4 = c4 = None
            for q in qids:   # first Choice with a by-position readout supplies the columns
                bp = (rep["questions"].get(q) or {}).get("by_position")
                if bp:
                    q1, q4, c4 = bp[0]["acc"], bp[3]["acc"], bp[3]["mean_conf"]; break
            if q1 is not None and q4 is not None and round(q1 - q4, 9) >= 0.15:
                flags.append("last-quarter")
            print(f"{n:>3} {J.fmt(rep['questions_per_request']):>5} " + " ".join(cells)
                  + f" {J.fmt(q1):>6} {J.fmt(q4):>6} {J.fmt(c4):>7} {J.fmt(rep['tokens_per_item'], 0):>8} {J.fmt(rep['ms_per_item'], 0):>7} {rep['requests']:>8}"
                  + f"  failed {rep['failed']}/{rep['items']}"
                  + (f"   <-- investigate (heuristic): {'; '.join(flags)}" if flags else ""))
            if a.verbose:
                J.print_report(f"N={n}", rep, True)
    except J.JevError as e:
        print(f"error: {J.redact(str(e))}", file=sys.stderr); sys.exit(2)


if __name__ == "__main__":
    main()
