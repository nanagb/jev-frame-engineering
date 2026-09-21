#!/usr/bin/env python3
"""Offline tests for eval.py's option-permutation pass. No network: the runner is never called.

  python3 tests/test_eval.py

Covers reversed_options (only Choice criteria move, values and Nouls untouched, input not
mutated, a nested question set round-trips) and report_permutation (flips, threshold crossings and
mean |Δconf| counted from synthetic answer pairs; the policy threshold is the one used; items
missing an answer are skipped; Nouls are not reported).
"""
import io, json, os, sys, unittest
from contextlib import redirect_stdout
SCRIPTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "skills", "jev-frame-engineering", "scripts"))
sys.path.insert(0, SCRIPTS)
import eval as ev  # the script, imported as a module (its main() only runs under __main__)

QSET = {"questions": {
    "queue": {"type": "choice", "instructions": "route it",
              "criteria": {"billing": {"what": "money"}, "technical": {"what": "broken"}, "other": {"what": "none"}}},
    "urgent": {"type": "noul", "instructions": "is it urgent", "criteria": {"true": "yes", "false": "no"}},
}}


def item(i, choice, conf, noul=0.1):
    return {"id": f"m{i}", "expected": {"queue": choice}, "answers": {"queue": {"choice": choice, "confidence": conf}, "urgent": {"noul": noul}}}


class ReversedOptions(unittest.TestCase):
    def test_choice_criteria_reversed_values_kept(self):
        rev = ev.reversed_options(QSET)
        self.assertEqual(list(rev["questions"]["queue"]["criteria"]), ["other", "technical", "billing"])
        self.assertEqual(rev["questions"]["queue"]["criteria"]["billing"], {"what": "money"})
        self.assertEqual(rev["questions"]["queue"]["instructions"], "route it")

    def test_noul_untouched_and_input_not_mutated(self):
        before = json.dumps(QSET)
        rev = ev.reversed_options(QSET)
        self.assertEqual(rev["questions"]["urgent"], QSET["questions"]["urgent"])
        self.assertEqual(json.dumps(QSET), before)
        self.assertEqual(list(QSET["questions"]["queue"]["criteria"]), ["billing", "technical", "other"])

    def test_bare_question_map_and_double_reverse(self):
        bare = QSET["questions"]
        once = ev.reversed_options(bare)
        self.assertEqual(list(once["queue"]["criteria"]), ["other", "technical", "billing"])
        self.assertEqual(json.dumps(ev.reversed_options(once)), json.dumps(bare))

    def test_nested_question_set_round_trips(self):
        qset = {"model": "jev-1.13.0", "questions": {
            "queue": {"type": "choice",
                      "instructions": {"inspect": "message", "question": "which queue?", "notes": ["message only"]},
                      "criteria": {"billing": {"what": "money", "not_for": "pre-sales", "examples": ["charged twice"]},
                                   "technical": {"what": "broken"}, "account_access": {"what": "locked out"},
                                   "sales": {"what": "pre-sales"}, "other": {"what": "none of the above"}}},
            "urgent": {"type": "noul", "instructions": "urgent?", "criteria": {"true": "yes", "false": "no"}},
            "contains_secret": {"type": "noul", "instructions": "secret?", "criteria": {"true": "yes", "false": "no"}}}}
        rev = ev.reversed_options(qset)
        q = ev.J.questions_of(qset)["queue"]; r = ev.J.questions_of(rev)["queue"]
        self.assertEqual(list(r["criteria"]), list(reversed(list(q["criteria"]))))
        self.assertNotEqual(list(r["criteria"]), list(q["criteria"]))
        for k in q["criteria"]:
            self.assertEqual(r["criteria"][k], q["criteria"][k])
        for qid in ("urgent", "contains_secret"):
            self.assertEqual(ev.J.questions_of(rev)[qid], ev.J.questions_of(qset)[qid])


class ReportPermutation(unittest.TestCase):
    def run_report(self, base, perm, policy=None):
        out = io.StringIO()
        with redirect_stdout(out):
            ev.report_permutation(base, perm, QSET, policy)
        return out.getvalue()

    def test_counts_flips_crossings_and_mean_dconf(self):
        base = [item(0, "billing", 0.90), item(1, "technical", 0.80), item(2, "other", 0.70), item(3, "billing", 0.76)]
        perm = [item(0, "billing", 0.90), item(1, "other", 0.80), item(2, "other", 0.78), item(3, "billing", 0.74)]
        # item1 flips label (no crossing); item2 crosses up (0.70 -> 0.78); item3 crosses down (0.76 -> 0.74)
        out = self.run_report(base, perm)
        self.assertIn("option order [queue]: 1 label flips of 4, 2 crossed ≥0.75, mean |Δconf| 0.025", out)
        self.assertIn("m1", out); self.assertIn("technical -> other", out)
        self.assertNotIn("urgent", out)

    def test_policy_threshold_is_used(self):
        base = [item(0, "billing", 0.90), item(1, "technical", 0.86)]
        perm = [item(0, "billing", 0.90), item(1, "technical", 0.84)]
        self.assertIn("0 crossed ≥0.75", self.run_report(base, perm))
        self.assertIn("1 crossed ≥0.85", self.run_report(base, perm, {"queue": {"threshold": 0.85}}))

    def test_identical_runs_report_nothing_moved(self):
        base = [item(0, "billing", 0.90), item(1, "other", 0.60)]
        self.assertIn("0 label flips of 2, 0 crossed ≥0.75, mean |Δconf| 0.000", self.run_report(base, json.loads(json.dumps(base))))

    def test_unanswered_items_are_skipped(self):
        base = [item(0, "billing", 0.90), {"id": "m1", "expected": {}, "answers": None}, item(2, "other", 0.60)]
        perm = [item(0, "other", 0.90), item(1, "billing", 0.90), item(2, "other", 0.60)]
        self.assertIn("1 label flips of 2", self.run_report(base, perm))


class NoulReportFormatting(unittest.TestCase):
    """A set with no positive items yields lowest_true = None (and highest_false = None with no
    negatives). sweep_batch.py once formatted those with :.2f and crashed on the first real
    validation set that had nothing to catch. Every script must format them through jevlib.fmt."""

    def test_fmt_handles_none(self):
        self.assertEqual(ev.J.fmt(None), "-")
        self.assertEqual(ev.J.fmt(0.5), "0.50")

    def test_no_script_formats_margins_directly(self):
        import glob, re
        from pathlib import Path
        bad = []
        for path in glob.glob(os.path.join(SCRIPTS, "*.py")):
            for n, line in enumerate(Path(path).read_text().splitlines(), 1):
                if re.search(r"\[.(lowest_true|highest_false).\]:\.\d", line):
                    bad.append(f"{os.path.basename(path)}:{n}")
        self.assertEqual(bad, [], f"margins formatted without jevlib.fmt: {bad}")

    def test_score_reports_none_margins_without_positives(self):
        qset = {"questions": {"flag": {"type": "noul", "instructions": "x", "criteria": {"true": "t", "false": "f"}}}}
        meta = {"requests": 1, "tokens": 300, "ms": 250}
        res = [{"id": "a", "expected": {"flag": False}, "answers": {"flag": {"noul": 0.2}}, **meta},
               {"id": "b", "expected": {"flag": False}, "answers": {"flag": {"noul": 0.4}}, **meta}]
        rep = ev.J.score(res, qset, {})["questions"]["flag"]
        self.assertIsNone(rep["lowest_true"]); self.assertEqual(rep["highest_false"], 0.4)
        self.assertEqual(rep["positives"], 0); self.assertEqual(rep["false_positives"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
