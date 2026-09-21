"""Offline behavior tests for request handling, batching, and evaluation reporting."""
import copy
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from email.utils import formatdate
from unittest.mock import patch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "skills", "jev-frame-engineering", "scripts")))
import ablate  # noqa: E402  (scripts dir added above)
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


if __name__ == "__main__":
    unittest.main()
