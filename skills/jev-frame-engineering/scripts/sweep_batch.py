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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-template", required=True); ap.add_argument("--items", action="append", required=True)
    ap.add_argument("--sizes", default="1,4,8,12,16,20"); ap.add_argument("--policy"); ap.add_argument("--model")
    ap.add_argument("--sleep", type=float, default=0.05); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    tpl = J.load_json(a.batch_template); policy = J.load_json(a.policy) if a.policy else {}
    items = [it for p in a.items for it in J.load_items(p)]
    sizes = [int(s) for s in a.sizes.split(",") if s]
    if not sizes or any(n < 1 for n in sizes):
        ap.error("--sizes must contain positive integers")
    qids = list(tpl["questions"])
    print(f"{len(items)} items; questions per item: {len(qids)} ({', '.join(qids)})")
    print(f"{'N':>3} {'q/req':>5} " + " ".join(f"{q[:22]:<34}" for q in qids) + f" {'Q1 acc':>6} {'Q4 acc':>6} {'Q4 conf':>7} {'tok/item':>8} {'ms/item':>7} {'requests':>8}")
    best = {}
    try:
        for n in sizes:
            res = J.run_batched(items, tpl, n, a.model, a.sleep); rep = J.score(res, tpl, policy)
            cells = []; flags = []
            for q in qids:
                r = rep["questions"].get(q)
                if not r:
                    cells.append(f"{'(no scored answers)':<34}"); continue
                if r["type"] == "choice":
                    cells.append(f"fine {r['fine']:>3}/{r['n']} ≥thr {r['pass']:>3}({r['pass_correct']:>3}) conf {r['mean_conf']:.2f}".ljust(34))
                    b = best.setdefault(q, {"fine": r["fine"], "conf": r["mean_conf"]})
                    if r["fine"] <= b["fine"] - 3 or r["mean_conf"] <= b["conf"] - 0.05:
                        flags.append(q)
                    best[q] = {"fine": max(b["fine"], r["fine"]), "conf": max(b["conf"], r["mean_conf"])}
                elif r["type"] == "noul":
                    cells.append(f"caught {r['caught']}/{r['positives']} fp {r['false_positives']}/{r['negatives']} m {J.fmt(r['lowest_true'])}/{J.fmt(r['highest_false'])}".ljust(34))
                    b = best.setdefault(q, {"fp": r["false_positives"], "lo": r["lowest_true"], "hi": r["highest_false"]})
                    if (r["false_positives"] > b["fp"]
                            or (r["lowest_true"] is not None and r["lowest_true"] <= (b["lo"] or 0) - 0.1)
                            or (r["highest_false"] is not None and r["highest_false"] >= (b["hi"] if b["hi"] is not None else 1) + 0.2)):
                        flags.append(q)   # a rising highest-false score is the early warning before false positives appear
                    best[q] = {"fp": min(b["fp"], r["false_positives"]), "lo": max(b["lo"] or 0, r["lowest_true"] or 0),
                               "hi": min(b["hi"] if b["hi"] is not None else 1, r["highest_false"] if r["highest_false"] is not None else 1)}
                else:
                    cells.append(f"MAE {r['mae']:.2f} within½ {r['within_half_level']}/{r['n']}".ljust(34))
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
