"""Offline behavior tests for request handling, batching, and evaluation reporting."""
import ast
import copy
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from email.utils import formatdate
from unittest.mock import patch

try:
    from . import scripts_path  # noqa: F401  (package run: python3 -m unittest tests.test_jevlib)
except ImportError:
    import scripts_path  # noqa: F401  (discover -s tests, or python3 tests/test_jevlib.py)
import ablate
import eval as ev
import jevlib as J
import sweep_batch

KEY = "test-credential/with+padding="
QUESTIONS = {"flag": {"type": "noul", "instructions": "Does `message` need review?"}}


def write_fixture(directory):
    """Write a minimal support-triage style fixture (questions, batch template, 30 items); return its paths."""
    criteria = {
        "billing": {"what": "charges, invoices, refunds", "not_for": "pre-sales questions",
                    "examples": ["I was charged twice"]},
        "technical": {"what": "the product misbehaves", "not_for": "login problems",
                      "examples": ["export fails with a 500"]},
    }
    noul = {"type": "noul", "instructions": "Does `message` need attention today?",
            "criteria": {"true": "an outage or same-day deadline", "false": "a routine request"}}
    questions = {"model": J.DEFAULT_MODEL, "questions": {
        "queue": {"type": "choice",
                  "instructions": {"inspect": "message", "question": "`message` is a support ticket. Which queue?",
                                   "notes": ["judge from the message alone"]},
                  "criteria": criteria},
        "urgent": noul}}
    template = {"model": J.DEFAULT_MODEL, "array_field": "messages", "item_value": "message", "reference": "keyed",
                "shared_state": {"queues": {k: {"what": v["what"]} for k, v in criteria.items()}},
                "questions": {
                    "queue": {"type": "choice",
                              "instructions": {"inspect": "{ref}",
                                               "question": "Which queue handles the ticket at `{ref}`, per `queues`?"},
                              "criteria": {k: f"as defined in `queues.{k}`" for k in criteria}},
                    "urgent": {**noul, "instructions": "Does the ticket at `{ref}` need attention today?"}}}
    items = [{"id": f"t{i:02d}", "state": {"message": f"ticket {i}: " + ("charged twice" if i % 2 else "export fails")},
              "expected": {"queue": "billing" if i % 2 else "technical", "urgent": False}} for i in range(30)]
    paths = {}
    for name, data in (("questions.json", questions), ("batch-template.json", template)):
        paths[name] = os.path.join(directory, name)
        with open(paths[name], "w") as f:
            json.dump(data, f)
    paths["dev.jsonl"] = os.path.join(directory, "dev.jsonl")
    with open(paths["dev.jsonl"], "w") as f:
        f.write("".join(json.dumps(item) + "\n" for item in items))
    return paths


def response(answers=None, model=J.DEFAULT_MODEL):
    return {"model": model, "answers": answers or {"flag": {"type": "noul", "noul": 0.8}},
            "usage": {"input_tokens": 100, "output_tokens": 10}}


def wire(data=None):
    return io.BytesIO(json.dumps(response() if data is None else data).encode())


def http_error(code, retry_after=None, body=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(J.API_URL, code, "test error", headers,
                                  io.BytesIO((body or "synthetic error").encode()))


class ClientBehavior(unittest.TestCase):
    def test_request_contract_and_valid_response(self):
        with patch.object(J.urllib.request, "urlopen", return_value=wire()) as call:
            data, ms = J.post({"message": "hello"}, QUESTIONS, key=KEY)
        req = call.call_args.args[0]
        self.assertEqual(req.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(json.loads(req.data), {"model": J.DEFAULT_MODEL,
                         "state": {"message": "hello"}, "questions": QUESTIONS})
        self.assertEqual(req.get_header("Authorization"), "Bearer " + KEY)
        self.assertEqual(data["answers"]["flag"]["noul"], 0.8)
        self.assertGreaterEqual(ms, 0)

    def test_nontransient_errors_do_not_retry_and_sanitize_before_storage(self):
        for code in (400, 401, 403, 404, 409, 413, 422):
            with self.subTest(code=code), patch.object(J.urllib.request, "urlopen",
                    side_effect=http_error(code, body=f'{{"key":"{KEY}"}} bearer second/token+=')) as call, \
                    patch.object(J.time, "sleep") as sleep:
                with self.assertRaises(J.JevError) as caught:
                    J.post("item", QUESTIONS, key=KEY)
                self.assertNotIsInstance(caught.exception, J.JevNoJudgment)
                self.assertNotIn(KEY, str(caught.exception))
                self.assertNotIn("second/token", str(caught.exception))
                self.assertEqual(call.call_count, 1)
                sleep.assert_not_called()

    def test_transient_statuses_retry_and_honor_seconds(self):
        for code in (408, 429, 500, 503, 529):
            with self.subTest(code=code), patch.object(J.urllib.request, "urlopen",
                    side_effect=[http_error(code, "2"), wire()]) as call, \
                    patch.object(J.time, "sleep") as sleep:
                J.post("item", QUESTIONS, key=KEY)
                self.assertEqual(call.call_count, 2)
                sleep.assert_called_once_with(2.0)

    def test_retry_after_http_date(self):
        now = 1800000000
        with patch.object(J.time, "time", return_value=now), \
                patch.object(J.urllib.request, "urlopen", side_effect=[
                    http_error(429, formatdate(now + 12, usegmt=True)), wire()]), \
                patch.object(J.time, "sleep") as sleep:
            J.post("item", QUESTIONS, key=KEY)
        sleep.assert_called_once_with(12.0)

    def test_long_retry_after_returns_no_judgment_without_retrying_early(self):
        with patch.object(J.urllib.request, "urlopen", side_effect=http_error(429, "3600")) as call, \
                patch.object(J.time, "sleep") as sleep:
            with self.assertRaises(J.JevNoJudgment):
                J.post("item", QUESTIONS, key=KEY)
        self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()

    def test_invalid_retry_after_uses_backoff(self):
        for header in ("invalid", "NaN", "Infinity"):
            with self.subTest(header=header), patch.object(J.urllib.request, "urlopen",
                    side_effect=[http_error(429, header), wire()]), patch.object(J.time, "sleep") as sleep:
                J.post("item", QUESTIONS, key=KEY)
                sleep.assert_called_once_with(1.0)

    def test_exhausted_transport_errors_are_sanitized_and_bounded(self):
        with patch.object(J.urllib.request, "urlopen", side_effect=TimeoutError(KEY)) as call, \
                patch.object(J.time, "sleep") as sleep:
            with self.assertRaises(J.JevNoJudgment) as caught:
                J.post("item", QUESTIONS, key=KEY)
        self.assertEqual(call.call_count, 3)
        self.assertEqual([x.args[0] for x in sleep.call_args_list], [1.0, 2.0])
        self.assertNotIn(KEY, str(caught.exception))

    def test_exhausted_http_error_is_sanitized(self):
        with patch.object(J.urllib.request, "urlopen", side_effect=http_error(529, body=KEY)):
            with self.assertRaises(J.JevNoJudgment) as caught:
                J.post("item", QUESTIONS, key=KEY, max_retries=0)
        self.assertNotIn(KEY, str(caught.exception))

    def test_malformed_json_is_no_judgment(self):
        with patch.object(J.urllib.request, "urlopen", return_value=io.BytesIO(b"not JSON")) as call:
            with self.assertRaises(J.JevNoJudgment):
                J.post("item", QUESTIONS, key=KEY)
        self.assertEqual(call.call_count, 1)

    def test_missing_and_invalid_answers_are_no_judgment(self):
        bad = [[], {}, {**response(), "answers": {}}, {**response(), "usage": {}},
               response({"flag": {"type": "noul", "noul": float("nan")}}),
               response({"flag": {"type": "noul", "noul": 2}}),
               response({"flag": {"type": "choice", "noul": 0.5}})]
        for data in bad:
            with self.subTest(data=data), patch.object(J.urllib.request, "urlopen", return_value=wire(data)):
                with self.assertRaises(J.JevNoJudgment):
                    J.post("item", QUESTIONS, key=KEY)

    def test_choice_and_score_response_shapes(self):
        qs = {"route": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
              "level": {"type": "score", "criteria": ["low", "high"]}}
        answers = {"route": {"type": "choice", "choice": "a", "confidence": 0.5,
                             "probabilities": {"a": 0.75, "b": 0.25}},
                   "level": {"type": "score", "score": 0.25, "confidence": 0.5,
                             "probabilities": {"0": 0.75, "1": 0.25},
                             "legend": {"0": "low", "1": "high"}}}
        with patch.object(J.urllib.request, "urlopen", return_value=wire(response(answers))):
            data, _ = J.post("item", qs, key=KEY)
        self.assertEqual(data["answers"], answers)
        broken = copy.deepcopy(answers)
        broken["route"]["choice"] = "not-an-option"
        with patch.object(J.urllib.request, "urlopen", return_value=wire(response(broken))):
            with self.assertRaises(J.JevNoJudgment):
                J.post("item", qs, key=KEY)

    def test_raw_and_bearer_redaction(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": KEY}):
            result = J.redact(f'{KEY} BeArEr other/secret+=, normal')
        self.assertNotIn(KEY, result)
        self.assertNotIn("other/secret", result)
        self.assertIn("normal", result)


class BatchedBehavior(unittest.TestCase):
    def template(self, mode="keyed"):
        return {"array_field": "messages", "reference": mode, "shared_state": {"policy": "review"},
                "questions": {"flag": {"type": "noul", "instructions": "Review `{ref}`?"}}}

    def items(self):
        return [{"id": str(i), "state": value, "expected": {"flag": True}}
                for i, value in enumerate(['a "quoted"\nmessage', {"text": "object"}, ["array", "item"]])]

    def fake_post(self, state, questions, model, key):
        return response({qid: {"type": "noul", "noul": 0.8} for qid in questions}), 120

    def test_keyed_batch_mapping_and_partial_batch_usage(self):
        with patch.object(J, "post", side_effect=self.fake_post) as call:
            rows = J.run_batched(self.items(), self.template(), 2, key=KEY, sleep=0)
        state, qs = call.call_args_list[0].args[:2]
        self.assertEqual(state["messages"]["item001"], {"text": "object"})
        self.assertEqual(qs["flag__1"]["instructions"], "Review `messages.item001`?")
        self.assertEqual([r["id"] for r in rows], ["0", "1", "2"])
        self.assertEqual([r["batch_len"] for r in rows], [2, 2, 1])
        self.assertEqual(sum(r["requests"] for r in rows), 2)
        self.assertEqual(sum(r["tokens"] for r in rows), 200)

    def test_quoted_mode_embeds_every_json_value(self):
        items = self.items()
        with patch.object(J, "post", side_effect=self.fake_post) as call:
            J.run_batched(items, self.template("quoted"), 3, key=KEY, sleep=0)
        qs = call.call_args.args[1]
        for i, item in enumerate(items):
            text = qs[f"flag__{i}"]["instructions"]
            self.assertIn(json.dumps(item["state"], ensure_ascii=False), text)

    def test_index_mode_remains_supported(self):
        with patch.object(J, "post", side_effect=self.fake_post) as call:
            J.run_batched(self.items(), self.template("index"), 3, key=KEY, sleep=0)
        self.assertIsInstance(call.call_args.args[0]["messages"], list)
        self.assertEqual(call.call_args.args[1]["flag__2"]["instructions"], "Review `messages[2]`?")

    def test_invalid_size_collision_and_legacy_index_rejected_without_calls(self):
        bad = self.template()
        bad["shared_state"]["messages"] = "must survive"
        legacy = self.template()
        legacy["questions"]["flag"]["instructions"] = "Review messages[{j}]?"
        for tpl, n in [(self.template(), 0), (self.template(), -1), (bad, 1), (legacy, 1)]:
            with self.subTest(tpl=tpl, n=n), patch.object(J, "post") as call:
                with self.assertRaises(J.JevError):
                    J.run_batched(self.items(), tpl, n, key=KEY, sleep=0)
                call.assert_not_called()

    def test_failed_batch_keeps_every_item_unjudged_and_counted(self):
        with patch.object(J, "post", side_effect=J.JevNoJudgment("unavailable")):
            rows = J.run_batched(self.items(), self.template(), 2, key=KEY, sleep=0)
        self.assertTrue(all(r["answers"] is None for r in rows))
        report = J.score(rows, QUESTIONS)
        self.assertEqual(report["failed"], 3)
        self.assertEqual(report["requests"], 2)
        self.assertIsNone(report["usd_per_1000_items"])
        with redirect_stdout(io.StringIO()):
            J.print_report("all failed", report, True)


class ReportingBehavior(unittest.TestCase):
    def rows(self, model=J.DEFAULT_MODEL):
        return [{"id": 1, "state": "hello", "expected": {"flag": False},
                 "answers": response()["answers"], "model": model,
                 "requests": 1, "tokens": 100, "ms": 120}]

    def test_unknown_model_does_not_use_old_price(self):
        report = J.score(self.rows("future-model"), QUESTIONS)
        self.assertIsNone(report["usd_per_1000_items"])
        self.assertAlmostEqual(J.score(self.rows(), QUESTIONS)["usd_per_1000_items"], 0.0042)

    def test_mixed_models_and_failed_requests_are_visible(self):
        rows = self.rows() + self.rows("future-model") + [{"answers": None, "requests": 1}]
        report = J.score(rows, QUESTIONS)
        self.assertEqual(report["model"], "mixed")
        self.assertEqual(report["requests"], 3)
        out = io.StringIO()
        with redirect_stdout(out):
            J.print_report("mixed", report, True)
        self.assertIn("multiple resolved models", out.getvalue())
        self.assertIn("failed 1", out.getvalue())

    def test_perfect_confidence_is_in_closed_top_bucket(self):
        qs = {"route": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}
        rows = [{**self.rows()[0], "expected": {"route": "b"},
                 "answers": {"route": {"choice": "a", "confidence": 1.0}}}]
        report = J.score(rows, qs)
        self.assertEqual(report["questions"]["route"]["buckets"][0]["n_above"], 1)
        out = io.StringIO()
        with redirect_stdout(out):
            J.print_report("bucket", report, True)
        self.assertIn("[0.90,1.00]", out.getvalue())
        self.assertIn("miss 1", out.getvalue())

    def test_eval_reports_all_repeats(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(d)
            argv = ["eval.py", "--questions", fx["questions.json"], "--items", fx["dev.jsonl"], "--repeat", "3"]
            with patch("sys.argv", argv), patch.object(J, "run_single", return_value=self.rows()), redirect_stdout(out):
                ev.main()
        self.assertIn("run1 vs run2", out.getvalue())
        self.assertIn("run1 vs run3", out.getvalue())

    def test_eval_rejects_zero_repeats(self):
        with patch("sys.argv", ["eval.py", "--questions", "unused", "--items", "unused", "--repeat", "0"]), \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                ev.main()
        self.assertEqual(caught.exception.code, 2)

    def test_all_failed_sweep_completes_and_reports_failures(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(d)
            argv = ["sweep_batch.py", "--batch-template", fx["batch-template.json"],
                    "--items", fx["dev.jsonl"], "--sizes", "1,8", "--sleep", "0"]
            with patch("sys.argv", argv), patch.object(J, "load_key", return_value=KEY), \
                    patch.object(J, "post", side_effect=J.JevNoJudgment("unavailable")), redirect_stdout(out):
                sweep_batch.main()
        self.assertEqual(out.getvalue().count("failed 30/30"), 2)
        self.assertIn("no scored answers", out.getvalue())

    def sweep_rows(self, fake_post, sizes="1,8", billing_at=(3, 27)):
        """Run sweep_batch over a 30-ticket file whose billing tickets sit at billing_at (one in the first full batch
        of 8, one in the trailing batch of 6, so nothing depends on where a batch boundary falls) and return the
        printed size rows. fake_post(state, questions, model, key) stands in for the API."""
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(d)
            with open(fx["dev.jsonl"], "w") as f:
                for i in range(30):
                    text = f"billing ticket {i}" if i in billing_at else f"export fails {i}"
                    f.write(json.dumps({"id": f"t{i:02d}", "state": {"message": text},
                                        "expected": {"queue": "billing" if i in billing_at else "technical", "urgent": False}}) + "\n")
            argv = ["sweep_batch.py", "--batch-template", fx["batch-template.json"], "--items", fx["dev.jsonl"], "--sizes", sizes, "--sleep", "0"]
            with patch("sys.argv", argv), patch.object(J, "load_key", return_value=KEY), \
                    patch.object(J, "post", side_effect=fake_post), redirect_stdout(out):
                sweep_batch.main()
        return [l for l in out.getvalue().splitlines() if re.match(r"^\s*\d+\s+(\d+|-)\s", l)]   # size rows: N, q/req, cells

    @staticmethod
    def routing_fake(misroute):
        """Answers every ticket by its text (the fixture writes 'billing ticket N' for billing) and misroutes the tickets
        misroute(ticket, batch) says to, so a test controls the outcome by ticket and batch size, not by batch order."""
        def fake_post(state, questions, model, key):
            tickets = state["messages"]   # keyed mode: an object in item order; quoted/index: a list
            tickets = list(tickets.values()) if isinstance(tickets, dict) else tickets
            answers = {}
            for qid, q in questions.items():
                if q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": 0.2}; continue
                ticket = tickets[int(qid.rsplit("__", 1)[1])]   # run_batched asks each question once per item, as qid__j
                truth = "billing" if ticket.startswith("billing") else "technical"
                choice = "technical" if truth == "billing" and misroute(ticket, tickets) else truth
                answers[qid] = {"type": "choice", "choice": choice, "confidence": 0.9,
                                "probabilities": {k: (0.9 if k == choice else 0.1) for k in q["criteria"]}}
            return response(answers), 120
        return fake_post

    def test_sweep_flags_a_rare_class_collapse_that_fine_hides(self):
        # 28 technical + 2 billing. Once a request carries more than one ticket the fake misroutes billing tickets to
        # technical: both of them (billing 2/2 -> 0/2 while fine drops only 30 -> 28, below the fine rule) must flag;
        # one of them (2/2 -> 1/2) is a single-item swing and must not, or every rare class would flag on noise.
        for lost, fine, low, flagged in ((("billing ticket 3", "billing ticket 27"), "fine  28/30", "low billing 0/2", True),
                                         (("billing ticket 27",), "fine  29/30", "low billing 1/2", False)):
            with self.subTest(lost=lost):
                lines = self.sweep_rows(self.routing_fake(lambda t, batch: len(batch) > 1 and t in lost))
                self.assertEqual(len(lines), 2)
                # at full recall everywhere the recall tie goes to the larger class
                self.assertIn("fine  30/30", lines[0]); self.assertIn("low technical 28/28", lines[0]); self.assertNotIn("investigate", lines[0])
                self.assertIn(fine, lines[1]); self.assertIn(low, lines[1])
                (self.assertIn if flagged else self.assertNotIn)("investigate (heuristic): queue", lines[1])

    def test_sweep_does_not_read_a_failed_batch_as_an_accuracy_drop(self):
        # one batch of 8 times out at N=8: the scored rows are still all right (22/22), so nothing may flag even
        # though 22 is more than three hits below the 30/30 of N=1
        routed = self.routing_fake(lambda t, batch: False)

        def fake_post(state, questions, model, key):
            tickets = state["messages"]; tickets = list(tickets.values()) if isinstance(tickets, dict) else tickets
            if len(tickets) > 1 and "export fails 8" in tickets:
                raise J.JevNoJudgment("timed out")
            return routed(state, questions, model, key)
        lines = self.sweep_rows(fake_post)
        self.assertIn("fine  22/22", lines[1]); self.assertIn("failed 8/30", lines[1])
        self.assertNotIn("investigate", lines[1])

    def test_choice_drop_rules(self):
        def rowd(fine, n, conf=0.9, **classes):
            return {"type": "choice", "fine": fine, "n": n, "mean_conf": conf,
                    "per_class": {lab: {"n": c[1], "hits": c[0], "recall": c[0] / c[1]} for lab, c in classes.items()}}
        drop = sweep_batch.choice_drop
        # items lost are the recall drop over the smaller of the two labelled counts, so a best size with a
        # different count is never compared hit-for-hit: 4/8, then 3/3 after a failed batch, then 2/6 loses
        # two of the three the best size can vouch for and flags
        b = {}
        self.assertFalse(drop(rowd(24, 30, billing=(4, 8)), b))
        self.assertFalse(drop(rowd(22, 25, billing=(3, 3)), b))
        self.assertTrue(drop(rowd(24, 28, billing=(2, 6)), b))
        # a size that scored one item of a class at full recall cannot vouch for two items, so a later size that
        # merely repeats the baseline does not flag against it
        b = {}
        self.assertFalse(drop(rowd(24, 30, billing=(2, 8)), b))
        self.assertFalse(drop(rowd(18, 23, billing=(1, 1)), b))
        self.assertFalse(drop(rowd(24, 30, billing=(2, 8)), b))
        # nor can two items prove a collapse on eight, while 8/8 -> 6/8 is the documented rule exactly
        b = {}
        drop(rowd(28, 30, billing=(2, 2)), b)
        self.assertFalse(drop(rowd(26, 30, billing=(4, 8)), b))
        b = {}
        drop(rowd(30, 30, billing=(8, 8)), b)
        self.assertTrue(drop(rowd(28, 30, billing=(6, 8)), b))
        # a one-item class cannot lose two items, so 1/1 -> 0/1 is noise, and 2/2 -> 1/2 stays below the floor
        b = {}
        drop(rowd(30, 30, rare=(1, 1)), b)
        self.assertFalse(drop(rowd(29, 30, rare=(0, 1)), b))
        b = {}
        drop(rowd(30, 30, billing=(2, 2)), b)
        self.assertFalse(drop(rowd(29, 30, billing=(1, 2)), b))
        self.assertTrue(drop(rowd(28, 30, billing=(0, 2)), b))
        # fine: three items below the best accuracy over the smaller count; a shrunken n at the same accuracy is no drop
        b = {}
        drop(rowd(30, 30), b)
        self.assertFalse(drop(rowd(22, 22), b))
        self.assertFalse(drop(rowd(20, 22), b))
        self.assertTrue(drop(rowd(19, 22), b))
        self.assertTrue(drop(rowd(30, 30, conf=0.85), b))
        self.assertEqual(b["classes"], {})
        # the first row never flags, the best is kept in place rather than rebuilt, and a tie goes to the larger count
        b = {}
        self.assertFalse(drop(rowd(10, 30, conf=0.5, a=(1, 4)), b))
        first = b["classes"]
        drop(rowd(30, 30, conf=0.9, a=(4, 4)), b); drop(rowd(20, 20, conf=0.9, a=(2, 2)), b)
        self.assertIs(b["classes"], first); self.assertEqual(b, {"classes": {"a": [(1.0, 4)]}, "acc": [(1.0, 30)], "conf": 0.9})
        # a perfect score on a few items (most batches failed) does not hide a later fall from a large sample:
        # 2/2, then 90/100, then 50/100 flags on fine, and the same shape flags a class
        b = {}
        drop(rowd(2, 2), b); self.assertFalse(drop(rowd(90, 100), b))
        self.assertTrue(drop(rowd(50, 100), b))
        self.assertEqual(b["acc"], [(1.0, 2), (0.9, 100)])
        b = {}
        drop(rowd(30, 30, billing=(1, 1)), b); self.assertFalse(drop(rowd(30, 30, billing=(9, 10)), b))
        self.assertTrue(drop(rowd(30, 30, billing=(5, 10)), b))

    MAINS = {"eval.py": ev.main, "ablate.py": ablate.main, "sweep_batch.py": sweep_batch.main}

    def run_script(self, argv, fake_post=None):
        """Run a script's main() offline with argv; returns (exit code or None, stdout, stderr, the post mock)."""
        out, err = io.StringIO(), io.StringIO(); code = None
        with patch("sys.argv", argv), patch.object(J, "load_key", return_value=KEY), \
                patch.object(J, "post", side_effect=fake_post) as call, redirect_stdout(out), redirect_stderr(err):
            try:
                self.MAINS[argv[0]]()
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue(), call

    @staticmethod
    def edit_json(path, change):
        with open(path) as f:
            data = json.load(f)
        change(data)
        with open(path, "w") as f:
            json.dump(data, f)

    def test_scripts_refuse_a_bad_label_file_before_any_request(self):
        # a misspelt option, a string "false" and a null Score level are caught by check_labels in every script:
        # exit 2, the item, the label and what the question takes named on stderr, and post never called
        sev = {"type": "score", "instructions": "How severe is it?", "criteria": ["minor", "major", "outage"]}
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(d)
            for name in ("questions.json", "batch-template.json"):
                self.edit_json(fx[name], lambda data: data["questions"].update(sev=sev))
            with open(fx["dev.jsonl"], "a") as f:
                f.write(json.dumps({"id": "t30", "state": {"message": "x"},
                                    "expected": {"queue": "biling", "urgent": "false", "sev": None}}) + "\n")
            for argv in (["eval.py", "--questions", fx["questions.json"], "--items", fx["dev.jsonl"]],
                         ["eval.py", "--batch-template", fx["batch-template.json"], "--items", fx["dev.jsonl"]],
                         ["ablate.py", "--questions", fx["questions.json"], "--question", "queue", "--items", fx["dev.jsonl"]],
                         ["sweep_batch.py", "--batch-template", fx["batch-template.json"], "--items", fx["dev.jsonl"]]):
                with self.subTest(argv=argv[:3]):
                    code, _, err, call = self.run_script(argv)
                    self.assertEqual(code, 2)
                    call.assert_not_called()
                    self.assertIn('t30: queue="biling"', err)
                    self.assertIn('queue takes one of "billing", "technical"', err)
                    if "ablate.py" not in argv:   # ablate sends only the target question, so it checks only that label
                        self.assertIn('t30: urgent="false"', err)
                        self.assertIn("t30: sev=null", err)
                        self.assertIn("sev takes a number from 0 to 2", err)

    def test_a_bad_label_file_is_refused_before_the_earlier_files_are_billed(self):
        # a second --items file is checked with the first: the clean dev set must not be run and billed before
        # the bad val set is looked at
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(d)
            val = os.path.join(d, "val.jsonl")
            with open(val, "w") as f:
                f.write(json.dumps({"id": "v0", "state": {"message": "x"}, "expected": {"queue": "biling"}}) + "\n")
            for argv in (["eval.py", "--questions", fx["questions.json"], "--items", fx["dev.jsonl"], "--items", val],
                         ["ablate.py", "--questions", fx["questions.json"], "--question", "queue",
                          "--items", fx["dev.jsonl"], "--items", val],
                         ["sweep_batch.py", "--batch-template", fx["batch-template.json"],
                          "--items", fx["dev.jsonl"], "--items", val]):
                main = {"eval.py": ev.main, "ablate.py": ablate.main, "sweep_batch.py": sweep_batch.main}[argv[0]]
                err = io.StringIO()
                with self.subTest(script=argv[0]), patch("sys.argv", argv), patch.object(J, "load_key", return_value=KEY), \
                        patch.object(J, "post") as call, redirect_stdout(io.StringIO()), redirect_stderr(err):
                    with self.assertRaises(SystemExit) as caught:
                        main()
                    self.assertEqual(caught.exception.code, 2)
                    call.assert_not_called()
                    self.assertIn('v0: queue="biling"', err.getvalue())

    def test_a_malformed_question_set_is_named_instead_of_the_label_file(self):
        # a misspelt `type` or a Choice given an array of options makes every label invalid, and a malformed
        # question the file does not label would fail at the API; every script names the question before any
        # request, and never sends the reader to a label file that is correct
        faults = (("queue", {"type": "Choice"}, "queue: type 'Choice'"),
                  ("queue", {"criteria": ["billing", "technical"]},
                   "queue: a choice question needs 'criteria', an object of option name to description, not an array"),
                  ("extra", {"type": "nul", "instructions": "x"}, "extra: type 'nul'"))
        for qid, change, fragment in faults:
            with tempfile.TemporaryDirectory() as d:
                fx = write_fixture(d)
                for name in ("questions.json", "batch-template.json"):
                    self.edit_json(fx[name], lambda data: data["questions"].__setitem__(qid, {**data["questions"].get(qid, {}), **change}))
                for argv in (["eval.py", "--questions", fx["questions.json"], "--items", fx["dev.jsonl"]],
                             ["sweep_batch.py", "--batch-template", fx["batch-template.json"], "--items", fx["dev.jsonl"]],
                             ["ablate.py", "--questions", fx["questions.json"], "--question", "queue", "--items", fx["dev.jsonl"]]):
                    if qid == "extra" and argv[0] == "ablate.py":
                        continue   # ablate sends only its target question
                    with self.subTest(fault=fragment, script=argv[0]):
                        code, _, err, call = self.run_script(argv)
                        self.assertEqual(code, 2)
                        call.assert_not_called()
                        self.assertIn("fix the questions, not the label file", err)
                        self.assertIn(fragment, err)
                        self.assertNotIn("t00", err)

    def test_bad_input_files_and_arguments_exit_with_a_message(self):
        # each of these was a traceback: a missing second --items file, a question id ablate cannot find, a plain
        # question set passed as a batch template, and a --sizes that is not a number
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(d); missing = os.path.join(d, "val.jsonl")
            dev, qs, tpl = fx["dev.jsonl"], fx["questions.json"], fx["batch-template.json"]
            for argv, fragment in (
                    (["eval.py", "--questions", qs, "--items", dev, "--items", missing], "cannot read " + missing),
                    (["ablate.py", "--questions", qs, "--question", "queue", "--items", dev, "--items", missing], "cannot read " + missing),
                    (["sweep_batch.py", "--batch-template", tpl, "--items", dev, "--items", missing], "cannot read " + missing),
                    (["ablate.py", "--questions", qs, "--question", "queu", "--items", dev],
                     "has no question 'queu'; its questions are queue, urgent"),
                    (["sweep_batch.py", "--batch-template", qs, "--items", dev], "a batch template needs 'array_field'"),
                    (["eval.py", "--batch-template", qs, "--items", dev], "a batch template needs 'array_field'"),
                    (["sweep_batch.py", "--batch-template", tpl, "--items", dev, "--sizes", "1,x"], "--sizes must contain positive integers")):
                with self.subTest(argv=argv):
                    code, _, err, call = self.run_script(argv)
                    self.assertEqual(code, 2)
                    call.assert_not_called()
                    self.assertIn(fragment, err)
                    self.assertNotIn("Traceback", err)

    def test_ablate_reads_a_question_named_questions_as_a_question(self):
        # ablate wraps its target question itself; unwrapped, an id of "questions" was read as the wrapper, so its
        # labels went unchecked and the request carried the question's fields as questions
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(d)
            with open(fx["questions.json"]) as f:
                queue = json.load(f)["questions"]["queue"]
            qpath = os.path.join(d, "q.json"); items = os.path.join(d, "items.jsonl")
            with open(qpath, "w") as f:
                json.dump({"questions": {"questions": queue}}, f)
            with open(items, "w") as f:
                f.write(json.dumps({"id": "a", "state": {"message": "charged twice"}, "expected": {"questions": "billing"}}) + "\n")
            answer = lambda state, questions, model, key: (response({qid: {
                "type": "choice", "choice": "billing", "confidence": 0.9, "probabilities": {"billing": 0.9, "technical": 0.1}}
                for qid in questions}), 100)
            argv = ["ablate.py", "--questions", qpath, "--question", "questions", "--items", items, "--sleep", "0"]
            code, out, _, call = self.run_script(argv, answer)
            self.assertIsNone(code)
            self.assertEqual(list(call.call_args.args[1]), ["questions"])
            self.assertEqual(call.call_args.args[1]["questions"]["type"], "choice")
            self.assertIn("fine   1/1", out)
            with open(items, "a") as f:
                f.write(json.dumps({"id": "b", "state": {"message": "x"}, "expected": {"questions": "biling"}}) + "\n")
            code, _, err, call = self.run_script(argv, answer)
            self.assertEqual(code, 2)
            call.assert_not_called()
            self.assertIn('b: questions="biling"', err)

    def test_all_failed_ablation_does_not_claim_zero_error(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            fx = write_fixture(d)
            argv = ["ablate.py", "--questions", fx["questions.json"],
                    "--question", "queue", "--items", fx["dev.jsonl"], "--sleep", "0"]
            with patch("sys.argv", argv), patch.object(J, "load_key", return_value=KEY), \
                    patch.object(J, "post", side_effect=J.JevNoJudgment("unavailable")), redirect_stdout(out):
                ablate.main()
        self.assertIn("no scored answers (failed 30/30)", out.getvalue())
        self.assertNotIn("MAE 0.00", out.getvalue())


class ScriptHygiene(unittest.TestCase):
    def test_scripts_parse_as_python_3_7(self):
        # the docs promise Python 3; an assignment expression in jevlib once broke every script on 3.6 and 3.7
        for name in sorted(os.listdir(scripts_path.SCRIPTS)):
            if name.endswith(".py"):
                with self.subTest(script=name), open(os.path.join(scripts_path.SCRIPTS, name), encoding="utf-8") as f:
                    ast.parse(f.read(), name, feature_version=(3, 7))

    def test_jevlib_help_says_it_is_a_library(self):
        run = subprocess.run([sys.executable, os.path.join(scripts_path.SCRIPTS, "jevlib.py"), "--help"],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0)
        self.assertIn("it has no command line", run.stdout)


if __name__ == "__main__":
    unittest.main()
