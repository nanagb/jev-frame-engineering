---
name: jev-frame-engineering
description: >
  Frame Engineering for Jev: design, test, and refine the evidence, questions, and
  answer boundaries in an existing TypeSafe integration. Use for decision framing,
  question ablations, batch references, token costs, calibrated action thresholds,
  and supporting production behavior. Use TypeSafe's official typesafe-ai skill or
  live documentation for the programming model and initial integration.
metadata:
  version: 0.9.4
---

# Frame Engineering for Jev

*Designing and evaluating reliable decision frames.*

**Decision Frame Engineering**, shortened to **Frame Engineering**, is the design, testing,
and refinement of the evidence, questions, and answer boundaries that shape a model's judgments.
For Jev, a decision frame consists of the relevant `state`, the judgment and subject references
in `instructions`, and the options or rubric boundaries expressed in `criteria` where applicable.
Evaluate changes against labelled outcomes, including failures and sensitivity to irrelevant
details, to determine whether the frame captures the intended distinction.

The practices below cover frame design and evaluation, with supporting integration guidance for
retries, caching, and deployment. Treat the live TypeSafe docs as the API reference and proposed
optimizations as hypotheses to test on the application's data. The scripts are evaluation helpers,
not a production SDK or proof of model performance.

Where a rule below cites a measurement, it was taken on a synthetic support-triage fixture of
50 labelled tickets against `jev-1.13.0` on 2026-09-20 with three repeats per condition. The
fixture is published in this skill's repository under `tests/fixtures/support-triage/`, with the
commands that reproduce each number, but is not installed with the skill. It is too small to
establish production thresholds: the numbers illustrate the method, and claims beyond them are
hypotheses for your data. The
[authoritative references](#authoritative-references) at the end are the API contract.

## 1. Measure before changing the request

- Separate development/tuning data from an untouched final test set. If validation results guide
  wording or threshold selection, that set is tuning data too; report final performance elsewhere.
- Repeat identical requests when measuring run-to-run variation. `--repeat 3` is a useful starting
  check, not a statistical significance test. On the fixture, five repeats of 50 items gave the
  same labels every time while probabilities moved by up to 0.08 and Noul values by up to 0.09:
  a repeat can move an item across a threshold but did not change a label, so voting has nothing
  to vote on unless your own repeats show label variation.
- Compare request bodies before attributing an effect: state, question wording, options, item
  position, batch composition, and model version can all matter. Change one axis at a time.
- Inspect misses for missing evidence, ambiguous labels, model mistakes, and integration bugs;
  do not assume one cause in advance or discard hard examples to improve a score.
- Report errors and coverage for the action the application takes, plus failures with no answer.
  A typed response does not prove the decision is correct.
- Use `--permute-options` in single-item mode to test sensitivity to Choice option order. A label
  flip identifies sensitivity; it does not by itself explain its cause. Keep the production order
  fixed while selecting thresholds.

See [evaluation protocol](references/eval-protocol.md) for formats, metrics, and threshold selection.

## 2. Start simple; add the information the question needs

- Plain strings are a valid starting point. Structured `what`, `not_for`, `examples`, `notes`, and
  `inspect` fields are descriptive content, not API operators. Keep additions that improve the
  target behavior on representative data; examples and exclusions are not inherently harmful.
  Draw `examples` from a labelled pool disjoint from every evaluation set, and keep a paraphrase
  on the same side as its source: on the fixture a paraphrase pulled as hard as the literal string.
- Add `other`/`none` when a Choice's options may not cover the input. Noul criteria are optional;
  explicit `true` and `false` descriptions can clarify boundaries. Add a presence Noul only when
  that is a useful separate judgment.
- Put needed policies, rosters, relationships, and current facts in state. Keep exact counts,
  arithmetic, date comparisons, and lookups in code. Question IDs are not visible to the model;
  include the item's reference and the full question in `instructions`.
- A population frequency or policy-dependent outcome needs suitable evidence and outcome labels;
  do not reinterpret a text classification as an empirically calibrated forecast without testing it.
- Choice and Score express uncertainty through their distributions and confidence; Noul returns
  a probability of yes without a separate confidence field. Missing evidence usually does not
  lower confidence: on the fixture, nine of ten empty, irrelevant or wrong-field states scored at
  least 0.95 on an arbitrary option when the Choice had no no-match option, and only a bare
  greeting spread the distribution. With `other` present all ten went to `other` at 0.97 or
  above and the Nouls stayed below 0.2. The no-match option catches missing evidence; confidence
  alone does not. Test empty, irrelevant and out-of-scope inputs explicitly.

`ablate.py` removes one descriptive field at a time. See [question economics](references/question-economics.md).

## 3. Make batched item references explicit and test the shipped form

Name items or quote them. Numeric paths such as `messages[17]` are documented and work for short
arrays, but they are natural-language references resolved by the model, and resolution degrades
with array length. On the fixture, `queue` accuracy with index references went from 49/50 at 8
items to 46-47/50 at 16, 38-40/50 at 32 and 23-24/48 at 48, where the last two quarters scored
2-3/12 and the secret Noul 28/48; keyed references held at 47-50 and quoted at 46-49 at every
length, for 2% and 17% more input tokens. A wrong answer under index references carries a
neighbour's label.

- Compare single-item answers with a batch-size-one template before increasing batch size.
  Otherwise changed criteria or shared definitions can confound the size comparison.
- The template supports `"reference": "keyed" | "quoted" | "index"`. Keyed mode names items;
  quoted mode embeds a JSON rendering of each item in the question. Neither is a guarantee; test
  the form you ship at the length you ship.
- Sweep sizes, rotate positions where useful, and inspect per-position errors. Neighbouring labels
  can suggest reference confusion but do not prove it is the only failure mode.
- Questions are independent against one shared state. Question count is not an accuracy axis
  (16, 48 and 240 questions over the same 16-item state scored the same in every repeat), but
  adding items changes the state, and a growing state is not accuracy-neutral.
- Check both current context limits and actual token use. For `jev-1.13.0`, the docs currently
  specify 64k tokens overall and 32k for state plus the longest question; recheck before deployment.

See [batching](references/batching.md); use `sweep_batch.py` for a diagnostic sweep.

## 4. Test untrusted content and enforce action policy in code

TypeSafe documents adversarial steering as a limitation of Jev 1.13, so a named field, quoting
or a separate request is not an injection defense by itself. What the fixture showed: two
injection strings placed third in a 32-item array changed no neighbour's label under keyed or
quoted references (28-29/29 with and without them, three repeats each) and were themselves
labelled `other`; under index references the plain one cut the neighbours' `queue` accuracy from
17/29 to 10-12/29, so it deepened index drift rather than steering. Keep trusted policy separate
from user content, test steering of the item's own answer and of its neighbours on your data, and
isolate requests when the residual cross-item risk is unacceptable.

A model judgment can inform an action, but authorization, allowed operations, and required
confirmation belong in application code. A Noul that detects suspicious instructions is another
fallible signal, not a security boundary. Use separate questions for distinct signals only when
the application needs them.

See [production contract](references/production-contract.md).

## 5. Measure shared definitions and choose fan-out by cost and latency

- Shared state is sent once. Moving repeated definitions there and referring to them by name
  saves tokens only once several items share a request, and the indirection can cost accuracy:
  on the fixture at batch size one, pointer criteria cost 15% more tokens than inline and the
  `urgent` Noul fell from 47-48/50 to 43/50 in all three repeats, every miss a borderline
  negative pushed above 0.5, while adding the same definitions to state without pointing at them
  cost at most one item. Compare with inline criteria before adopting it.
- Independent and speculative questions can run together; code consumes the applicable answers.
  Additional questions still consume input tokens. Routing first saves unused branch tokens (on
  the fixture, 1,247 tokens per item for route-then-ask against 1,699 for one request carrying
  fifteen branch questions) but adds a sequential call and resends state. Measure the complete
  workflow cost and latency.
- Use a second call when the first answer is needed to fetch evidence or construct the next
  questions. Questions within one call cannot consume each other's answers.

`token_probe.py` measures usage on synthetic inputs. Its slopes and baseline are estimates for
those inputs, not universal billing constants. See [question economics](references/question-economics.md).

## 6. Validate probabilities and thresholds for each action

Choice returns relative probabilities over competing options; `choice` selects the maximum
without discarding the distribution. Noul estimates whether a statement is true. These are
different judgments: do not assume a Choice probability equals a separately worded Noul. On the
fixture, Choice put 0.99 or more on one option for 66% of items while one Noul per option found
two applicable options on 36%, and on deliberately mixed tickets Choice kept a split on 8 of 12.
The distribution carries some ambiguity, less than the Nouls; read "does X apply" off a Noul.
A Noul near 0.5 means comparable probability for yes and no, not medium intensity.

Choice/Score `confidence` summarizes distribution concentration. The public docs give no
contractual formula, only the demo's `(K·p_max − 1)/(K − 1)`. That expression reproduced every
Choice confidence on the fixture within 0.02 (522 answers, 2 to 8 options) and does not reproduce
Score confidence (errors to 0.2; a Score's value is the probability-weighted mean of its levels),
so consume the returned field rather than deriving it. It is neither a universal probability of
correctness nor permission to act. Re-evaluate calibration when questions, options, model, or
traffic change.

Under a simple cost model with perfect review, zero cost for a correct automatic action,
constant error cost `C_wrong > 0`, and review cost `C_review`, automatic action breaks even at
`p_correct >= 1 - C_review / C_wrong`. This is a condition on correctness probability, not a
confidence cutoff. Choose a cutoff using labelled tuning data, adequate sample support and
uncertainty estimates, then check it on untouched data. More complex costs need a loss matrix.
Below it, review, abstain, or use a separately validated coarse decision; a parent label is not
automatically correct.

Report per-class recall and precision rather than overall accuracy whenever the labels are
imbalanced, at the operating threshold when an action depends on it, and for rare positives add
prevalence. Consider precision-recall curves/average precision as well as ROC-AUC; the latter can
conceal a poor precision outcome at low prevalence. `eval.py` prints, for each Choice option,
recall and precision over all scored answers and again counting only answers at the policy
threshold (the rates an automatic action achieves), and for each Noul recall, precision and true
negative rate at its threshold, plus Choice buckets; it does not draw those curves or fit a
calibrator. Reassess any calibration layer under traffic or base-rate changes.

## 7. Keep failures distinct and cache the complete judgment input

A failed call has no model answer. Represent it explicitly instead of silently storing a default
label. The helper raises `JevError` for non-retryable HTTP failures and `JevNoJudgment` for
exhausted transient failures or invalid responses. Use bounded retries and an application deadline.

Pin a version when thresholds depend on it, log the response's resolved `model`, and version
questions and state construction. Cache by the actual input content (or its stable digest),
question ID/version, and resolved model version. Include relevant policy, roster, or other context
in the digest. An unchanged subject ID with changed content must not reuse an old answer.
Downstream effects need their own idempotency keys.

See [production contract](references/production-contract.md) and the [pre-ship checks](references/anti-patterns.md).

## Scripts

Python 3, standard library only. Credentials come from `TYPESAFE_API_KEY` or
`~/.config/typesafe/api_key`; set file permissions to 600 (the loader does not enforce the mode).
Run each script by its path from a directory that holds the question sets and labels, and keep
those files and any `--json` output out of this skill's directory;
`python3 <skill-dir>/scripts/eval.py --help` prints a script's flags. Every script calls the
live API and is billed; `jevlib.py` is the library they share. `eval.py`, `ablate.py` and
`sweep_batch.py` check the question set and every label file before the first request, so a
malformed question, a misspelt option or a string `"false"` is reported by item, with what the
question takes, rather than billed and misreported. The client sanitizes its errors for the
active credential. Raw evaluation output still contains input data and must be handled
accordingly.

| script | use |
|---|---|
| `eval.py` | Choice/Noul/Score metrics with per-class recall/precision (over all answers and at the threshold) and Noul recall/precision/TNR; repeats; single-item option reversal; batch evaluation; `--json` raw answers |
| `ablate.py` | remove descriptive fields; compare metrics (with the lowest per-class recall) and usage per item file, one target question at a time |
| `sweep_batch.py` | compare batch sizes and position quarters; compares each size with every earlier size on the items both scored, so a failed batch neither raises nor hides a flag, and names what fell: a Choice's accuracy, confidence or one option's recall, a Noul's false positives or its lowest-true or highest-false score, or last-quarter accuracy; flags are heuristic, not significance tests |
| `token_probe.py` | inspect usage for minimal questions and synthetic text |
| `jevlib.py` | HTTP client, runners, scoring, error redaction |

Input file formats are described in the [evaluation protocol](references/eval-protocol.md).
For new empirical claims, retain exact request bodies, labels, raw responses, failures, repeats,
model versions, sample counts, and date. Keep observations scoped to those runs.

## Authoritative references

- [Official typesafe-ai skill](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md): programming model and live-docs workflow.
- [HTTP API](https://docs.typesafe.ai/api): request/response shapes and documented error statuses.
- [Choice](https://docs.typesafe.ai/primitives/choice), [Noul](https://docs.typesafe.ai/primitives/noul), [Score](https://docs.typesafe.ai/primitives/score): primitive semantics and question design.
- [Confidence](https://docs.typesafe.ai/confidence): uncertainty signal and action-specific thresholds.
- [State](https://docs.typesafe.ai/concepts/state): structured context and separation of content from questions.
- [Models](https://docs.typesafe.ai/models): current IDs, aliases, limits, and pricing.
- [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out): independent questions evaluated together.
- [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13): documented model limitations, including adversarial steering.
