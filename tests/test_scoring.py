"""Offline checks for jevlib.score and print_report.

  python3 tests/test_scoring.py

Covers per-class recall/precision on Choice (single and batched rows), lowest_recall, recall/precision/TNR
on Noul, labels a question cannot take (refused by check_labels before any request; never a hit, never a
class, and never a traceback when they reach score() anyway), and the table cells the scripts measure
their columns from.
"""
import io, contextlib, unittest
try:
    from . import scripts_path  # noqa: F401  (package run: python3 -m unittest tests.test_scoring)
except ImportError:
    import scripts_path  # noqa: F401  (discover -s tests, or python3 tests/test_scoring.py)
import jevlib as J

QSET = {"questions": {
    "queue": {"type": "choice", "instructions": "q", "criteria": {"a": "A", "b": "B", "c": "C"}},
    "flag": {"type": "noul", "instructions": "q"}}}


def row(i, exp_q, got_q, exp_f, p_f, conf=0.9, **batch):
    r = {"id": str(i), "state": {"t": "x"}, "expected": {"queue": exp_q, "flag": exp_f},
         "answers": {"queue": {"type": "choice", "choice": got_q, "confidence": conf},
                     "flag": {"type": "noul", "noul": p_f}},
         "model": "jev-1.13.0", "tokens": 100, "ms": 10, "requests": 1, "questions_per_request": 2}
    r.update(batch)   # pos= and batch_len= for rows that came out of a batched request
    return r


def report_text(rep, verbose=False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        J.print_report("t", rep, verbose)
    return buf.getvalue()


class SixRows(unittest.TestCase):
    """The shared scenario both question types are scored on."""
    def setUp(self):
        # queue: a expected 3 (hit 3), b expected 2 (hit 1, other -> c), c expected 1 (hit 0, -> a)
        # flag: positives 3 (caught 2), negatives 3 (1 false positive)
        self.res = [row(0, "a", "a", True, 0.9), row(1, "a", "a", True, 0.85), row(2, "a", "a", True, 0.3),
                    row(3, "b", "b", False, 0.1), row(4, "b", "c", False, 0.95), row(5, "c", "a", False, 0.2)]
        self.rep = J.score(self.res, QSET)


class ChoiceScoring(SixRows):
    def test_per_class(self):
        q = self.rep["questions"]["queue"]; pc = q["per_class"]
        self.assertEqual(pc["a"], {"n": 3, "hits": 3, "recall": 1.0, "predicted": 4, "precision": 0.75})
        self.assertEqual(pc["b"], {"n": 2, "hits": 1, "recall": 0.5, "predicted": 1, "precision": 1.0})
        self.assertEqual(pc["c"]["recall"], 0.0)
        self.assertEqual(pc["c"]["precision"], 0.0)
        self.assertEqual(q["fine"], 4)
        self.assertEqual(q["lowest_recall"], {"label": "c", "hits": 0, "n": 1, "recall": 0.0})

    def test_unseen_option_has_none_recall_and_precision(self):
        pc = J.score([row(0, "a", "a", True, 0.9)], QSET)["questions"]["queue"]["per_class"]
        self.assertIsNone(pc["b"]["recall"])
        self.assertIsNone(pc["b"]["precision"])

    def test_batched_rows_score_per_class(self):
        # rows with pos/batch_len >= 4 take the by-position path; it must not disturb the per-class pass
        labelled = [("a", "a"), ("b", "c"), ("c", "c"), ("a", "b")]
        res = [row(i, e, g, True, 0.9, pos=i, batch_len=4) for i, (e, g) in enumerate(labelled)]
        q = J.score(res, QSET)["questions"]["queue"]
        self.assertEqual(len(q["by_position"]), 4)
        self.assertEqual(q["by_position"][0], {"quarter": 1, "n": 1, "acc": 1.0, "mean_conf": 0.9})
        self.assertEqual({lab: c["hits"] for lab, c in q["per_class"].items()}, {"a": 1, "b": 0, "c": 1})
        self.assertEqual(q["per_class"]["c"]["predicted"], 2)
        self.assertEqual(q["lowest_recall"]["label"], "b")

    def test_offlist_choice_scores_as_a_miss(self):
        # the client refuses an answer outside the option list before scoring (test_jevlib covers that);
        # score() itself still treats one as a miss rather than a per-class row or a traceback
        q = J.score([row(0, "a", "zzz", True, 0.9)], QSET)["questions"]["queue"]
        self.assertEqual(list(q["per_class"]), ["a", "b", "c"])
        self.assertEqual((q["fine"], q["per_class"]["a"]["hits"], q["per_class"]["a"]["predicted"]), (0, 0, 0))

    def test_malformed_expected_values_score_as_misses(self):
        # null, an unlisted string and a list in the label file are misses in the report, not a traceback,
        # and are listed as JSON under unlisted labels rather than as per-class rows
        res = [row(0, None, "a", True, 0.9), row(1, "zzz", "a", True, 0.9), row(2, ["x"], "a", True, 0.9)]
        rep = J.score(res, QSET); q = rep["questions"]["queue"]
        self.assertEqual((q["fine"], len(q["misses"])), (0, 3))
        self.assertEqual(q["per_class"]["a"]["predicted"], 3)
        self.assertEqual(sum(c["n"] for c in q["per_class"].values()), 0)
        self.assertEqual(q["unlisted"], {'"zzz"': 1, '["x"]': 1, "null": 1})
        out = report_text(rep, verbose=True)
        self.assertIn('unlisted labels     "zzz" 1  ["x"] 1  null 1   (label-file values the question cannot take; scored as misses, counted in no class)', out)
        self.assertIn("expected ['x']", out)

    def test_coarse_accuracy_tolerates_malformed_expected_values(self):
        # a parents map used to look the raw label up, so a list or dict in the label file raised TypeError
        policy = {"queue": {"parents": {"a": "p", "b": "p"}}}
        res = [row(0, "a", "b", True, 0.9), row(1, ["x"], "a", True, 0.9), row(2, None, "a", True, 0.9), row(3, {"k": 1}, "a", True, 0.9)]
        q = J.score(res, QSET, policy)["questions"]["queue"]
        self.assertEqual((q["fine"], q["coarse"]), (0, 1))
        self.assertEqual(q["unlisted"], {'["x"]': 1, '{"k": 1}': 1, "null": 1})

    def test_a_label_naming_a_parent_is_a_coarse_miss(self):
        # "p" is a parent, not an option: rolled up it equals every child, so it once scored as coarse-correct
        # against any answer under that parent while fine counted it a miss
        policy = {"queue": {"parents": {"a": "p", "b": "p"}}}
        q = J.score([row(0, "p", "a", True, 0.9), row(1, "p", "c", True, 0.9)], QSET, policy)["questions"]["queue"]
        self.assertEqual((q["fine"], q["coarse"], q["n"]), (0, 0, 2))
        self.assertEqual(q["unlisted"], {'"p"': 2})

    def test_unlisted_labels_stay_distinct_and_never_headline(self):
        # 1, 1.0 and true are one dict key; the report must show three labels, and none of them may be
        # the lowest-recall option that ablate.py and sweep_batch.py headline
        res = [row(0, 1, "a", True, 0.9), row(1, True, "a", True, 0.9), row(2, 1.0, "a", True, 0.9), row(3, "b", "b", True, 0.9)]
        q = J.score(res, QSET)["questions"]["queue"]
        self.assertEqual(q["unlisted"], {"1": 1, "1.0": 1, "true": 1})
        self.assertEqual(q["lowest_recall"], {"label": "b", "hits": 1, "n": 1, "recall": 1.0})
        self.assertNotIn("unlisted", report_text(J.score([row(0, "a", "a", True, 0.9)], QSET)))

    def test_lowest_recall(self):
        self.assertIsNone(J.lowest_recall({}))
        self.assertEqual(J.fmt_low(None), "")
        self.assertEqual(J.fmt_low({"label": "technical", "hits": 2, "n": 3, "recall": 2 / 3}), " low technical 2/3")
        self.assertEqual(J.fmt_low({"label": "billing_dispute", "hits": 0, "n": 6, "recall": 0.0}), " low billing_dispute 0/6")
        self.assertIsNone(J.lowest_recall({"a": {"n": 0, "hits": 0, "recall": None, "predicted": 2, "precision": 0.0}}))
        tie = {"b": {"n": 2, "hits": 1, "recall": 0.5, "predicted": 1, "precision": 1.0},
               "a": {"n": 4, "hits": 2, "recall": 0.5, "predicted": 2, "precision": 1.0}}
        self.assertEqual(J.lowest_recall(tie)["label"], "b")   # a recall tie goes to the least support, not the name
        # a one-item class at full recall is the fragile one; it must headline over a large stable class
        stable = {"account": {"n": 30, "hits": 30, "recall": 1.0, "predicted": 30, "precision": 1.0},
                  "zzz_rare": {"n": 1, "hits": 1, "recall": 1.0, "predicted": 1, "precision": 1.0}}
        self.assertEqual(J.lowest_recall(stable), {"label": "zzz_rare", "hits": 1, "n": 1, "recall": 1.0})
        same = {"a": {"n": 2, "hits": 1, "recall": 0.5, "predicted": 1, "precision": 1.0},
                "b": {"n": 2, "hits": 1, "recall": 0.5, "predicted": 1, "precision": 1.0}}
        self.assertEqual(J.lowest_recall(same)["label"], "a")   # then the name, so the pick is stable

    def test_report_prints_counts_with_denominators(self):
        out = report_text(self.rep)
        self.assertIn("per-class recall    a 3/3  b 1/2  c 0/1", out)
        self.assertIn("per-class precision a 3/4  b 1/1  c 0/1", out)

    def test_report_prints_dash_for_options_without_support(self):
        out = report_text(J.score([row(0, "a", "a", True, 0.9)], QSET))
        self.assertIn("per-class recall    a 1/1  b -  c -", out)
        self.assertIn("per-class precision a 1/1  b -  c -", out)


class NoulScoring(SixRows):
    def test_rates(self):
        q = self.rep["questions"]["flag"]
        self.assertEqual((q["positives"], q["caught"], q["negatives"], q["false_positives"]), (3, 2, 3, 1))
        self.assertAlmostEqual(q["recall"], 2 / 3)
        self.assertAlmostEqual(q["precision"], 2 / 3)
        self.assertAlmostEqual(q["tnr"], 2 / 3)
        self.assertIn("TNR 0.67", report_text(self.rep))

    def test_zero_denominators(self):
        q = J.score([row(0, "a", "a", False, 0.1)], QSET)["questions"]["flag"]
        self.assertIsNone(q["recall"]); self.assertIsNone(q["precision"]); self.assertEqual(q["tnr"], 1.0)

    def test_string_and_numeric_labels_join_no_class(self):
        # "false" is truthy, so it was once scored as a positive the model missed; 0 and 1 are not booleans
        # either. None of them is a positive or a negative: they are listed, and the rates cover real labels
        res = [row(0, "a", "a", "false", 0.1), row(1, "a", "a", 0, 0.1), row(2, "a", "a", 1, 0.9), row(3, "a", "a", True, 0.9)]
        rep = J.score(res, QSET); q = rep["questions"]["flag"]
        self.assertEqual((q["n"], q["positives"], q["negatives"], q["caught"], q["false_positives"]), (4, 1, 0, 1, 0))
        self.assertEqual((q["recall"], q["tnr"]), (1.0, None))
        self.assertEqual(q["unlisted"], {'"false"': 1, "0": 1, "1": 1})
        self.assertEqual([m["expected"] for m in q["misses"]], ["false", 0, 1])
        self.assertIn('unlisted labels     "false" 1  0 1  1 1   (label-file values the question cannot take', report_text(rep))


class ScoreScoring(unittest.TestCase):
    QSET = {"questions": {"sev": {"type": "score", "instructions": "q", "criteria": ["low", "mid", "high"]}}}

    def srow(self, i, expected, got):
        return {"id": str(i), "state": "x", "expected": {"sev": expected}, "answers": {"sev": {"type": "score", "score": got}},
                "model": "jev-1.13.0", "tokens": 100, "ms": 10, "requests": 1, "questions_per_request": 1}

    def test_labels_off_the_rubric_do_not_traceback_after_the_run(self):
        # null, "high", a list and a level past the rubric used to raise inside float() once every item had been billed
        res = [self.srow(0, 1, 1.25), self.srow(1, None, 1.0), self.srow(2, "high", 2.0), self.srow(3, [1], 0.0),
               self.srow(4, 3, 2.0), self.srow(5, 2, 0.0)]
        rep = J.score(res, self.QSET); q = rep["questions"]["sev"]
        self.assertEqual((q["n"], q["within_half_level"]), (6, 1))
        self.assertAlmostEqual(q["mae"], (0.25 + 2.0) / 2)
        self.assertEqual(q["unlisted"], {'"high"': 1, "3": 1, "[1]": 1, "null": 1})
        self.assertEqual([m["id"] for m in q["misses"]], ["1", "2", "3", "4", "5"])
        out = report_text(rep, verbose=True)
        self.assertIn("MAE 1.12  within ½ level 1/6", out)
        self.assertIn("unlisted labels     ", out)

    def test_all_labels_invalid_reports_no_error_rather_than_zero(self):
        q = J.score([self.srow(0, None, 1.0)], self.QSET)["questions"]["sev"]
        self.assertIsNone(q["mae"]); self.assertEqual(q["within_half_level"], 0)


class LabelCheck(unittest.TestCase):
    """check_labels is the one place a label file is validated, and it runs before any request is sent."""
    QSET = {"questions": {"queue": {"type": "choice", "instructions": "q", "criteria": {"a": "A", "b": "B"}},
                          "flag": {"type": "noul", "instructions": "q"},
                          "sev": {"type": "score", "instructions": "q", "criteria": ["low", "mid", "high"]}}}

    def test_valid_label_per_type(self):
        qs = self.QSET["questions"]
        for value in ("a", "b"):
            self.assertTrue(J.valid_label(qs["queue"], value))
        for value in ("zzz", None, ["a"], 1, True):
            self.assertFalse(J.valid_label(qs["queue"], value), value)
        for value in (True, False):
            self.assertTrue(J.valid_label(qs["flag"], value))
        for value in ("false", "true", 0, 1, None, [True]):
            self.assertFalse(J.valid_label(qs["flag"], value), value)
        for value in (0, 1, 2, 1.5, 2.0):
            self.assertTrue(J.valid_label(qs["sev"], value), value)
        for value in (-1, 3, 2.5, "1", "high", None, True, float("nan")):
            self.assertFalse(J.valid_label(qs["sev"], value), value)

    def test_a_question_that_can_take_no_label_is_named_instead_of_the_file(self):
        # a misspelt type or a Choice/Score with no criteria rejects every label; blaming the label file sends
        # the reader to the wrong file, so the question set is named and only for questions the file labels
        items = [{"id": "t0", "state": "x", "expected": {"queue": "a", "sev": 1}}]
        for qset, fragment in (({"queue": {"type": "Choice", "criteria": {"a": "A"}}}, "queue: type 'Choice'"),
                               ({"queue": {"type": "choice"}}, "queue: a choice question needs 'criteria'"),
                               ({"sev": {"type": "score"}}, "sev: a score question needs 'criteria'"),
                               ({"queue": "oops"}, 'queue: "oops" is not a question object')):
            with self.subTest(qset=qset), self.assertRaises(J.JevError) as caught:
                J.check_labels(items, qset, "dev.jsonl")
            msg = str(caught.exception)
            self.assertIn("fix the questions, not the label file", msg)
            self.assertIn(fragment, msg)
            self.assertNotIn("dev.jsonl", msg)
        # a broken question the file never labels is not this file's problem
        J.check_labels(items, {"other": {"type": "wat"}})

    def test_worst_case_refuses_a_question_it_cannot_size(self):
        # ablate.py and sweep_batch.py size their columns from this before any row exists; a raw KeyError
        # there is a traceback instead of the scripts' `error: ...` exit
        for q in ({"criteria": {"a": "A"}}, {"type": "choice"}, {"type": "wat", "criteria": {"a": "A"}}):
            with self.subTest(q=q), self.assertRaises(J.JevError):
                J.worst_case(q, 10)

    def test_check_names_every_offending_item_and_sends_nothing(self):
        items = [{"id": "ok", "state": "x", "expected": {"queue": "a", "flag": False, "sev": 2}},
                 {"id": "partial", "state": "x", "expected": {"flag": True}},          # unlabelled questions are allowed
                 {"id": "extra", "state": "x", "expected": {"other_question": "whatever"}},   # keys outside the set are ignored
                 {"id": "bad1", "state": "x", "expected": {"queue": "zzz", "flag": "false"}},
                 {"id": "bad2", "state": "x", "expected": {"sev": None}},
                 {"id": "bad3", "state": "x", "expected": ["a"]}]
        with self.assertRaises(J.JevError) as caught:
            J.check_labels(items, self.QSET, "dev.jsonl")
        msg = str(caught.exception)
        self.assertNotIsInstance(caught.exception, J.JevNoJudgment)
        self.assertIn("dev.jsonl: 4 label(s)", msg)
        for fragment in ('bad1: queue="zzz"', 'bad1: flag="false"', "bad2: sev=null", 'bad3: expected=["a"] is not an object', "no request was sent"):
            self.assertIn(fragment, msg)
        for clean in ("ok", "partial", "extra"):
            self.assertNotIn(clean + ":", msg)
        J.check_labels(items[:3], self.QSET)   # a clean file passes silently
        J.check_labels(items[3:5], {"questions": {"unrelated": {"type": "noul"}}})   # only questions in the set are checked

    def test_load_items_rejects_a_non_object_expected(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dev.jsonl")
            with open(path, "w") as f:
                f.write('{"id": "t0", "state": "x", "expected": null}\n')
            with self.assertRaises(J.JevError) as caught:
                J.load_items(path)
            self.assertIn("line 1", str(caught.exception))


class TableCells(unittest.TestCase):
    """ablate.py and sweep_batch.py measure a column from cell(worst_case(q, n)); no real cell may be wider."""
    def test_worst_case_covers_the_longest_option_and_the_widest_mae(self):
        choice = {"type": "choice", "criteria": {"billing_refund": "r", "billing_dispute": "d", "ok": "o"}}
        real = {"type": "choice", "fine": 5, "n": 30, "coarse": None, "pass": 30, "pass_correct": 5, "mean_conf": 1.0,
                "lowest_recall": {"label": "billing_dispute", "hits": 0, "n": 30, "recall": 0.0}}
        self.assertIn("low billing_dispute 0/30", J.cell(real))
        self.assertGreaterEqual(len(J.cell(J.worst_case(choice, 30))), len(J.cell(real)))
        self.assertGreater(len(J.cell(J.worst_case(choice, 30, coarse=True))), len(J.cell(J.worst_case(choice, 30))))
        rubric = {"type": "score", "criteria": [f"level {i}" for i in range(12)]}
        self.assertIn("MAE 11.00", J.cell(J.worst_case(rubric, 30)))
        self.assertGreaterEqual(len(J.cell(J.worst_case(rubric, 30))),
                                len(J.cell({"type": "score", "mae": 10.0, "within_half_level": 0, "n": 30})))
        noul = J.cell(J.worst_case({"type": "noul"}, 30))
        self.assertGreaterEqual(len(noul), len(J.cell({"type": "noul", "caught": 30, "positives": 30, "false_positives": 30,
                                                       "negatives": 30, "lowest_true": None, "highest_false": None})))

    def test_cells_format_none_through_fmt(self):
        self.assertEqual(J.cell({"type": "score", "mae": None, "within_half_level": 0, "n": 2}), "MAE - within½ 0/2")
        self.assertIn("margin -/-", J.cell({"type": "noul", "caught": 0, "positives": 0, "false_positives": 0, "negatives": 0,
                                            "lowest_true": None, "highest_false": None}))


if __name__ == "__main__":
    unittest.main()
