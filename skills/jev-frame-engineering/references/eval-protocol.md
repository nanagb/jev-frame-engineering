# Evaluation protocol

Use this procedure to test changes to a request and the decisions consuming its answers.
Repeated requests measure run-to-run variation; they do not establish population accuracy.

## 1. Build labelled sets

Items are JSONL:

```json
{"id": "ticket-1", "state": {"message": "..."}, "expected": {"queue": "billing", "urgent": true}}
```

Expected values are option names for Choice, JSON booleans for Noul, and zero-based rubric
positions for Score. Partial labels are allowed. Score predictions can fall between levels.
`eval.py`, `ablate.py` and `sweep_batch.py` check each label against the question set before
their first request and refuse the file, naming the items and what each question takes, when a
value is one the question cannot take (a misspelt option, the string `"false"`, a level off the
rubric), so a label-file mistake costs no API calls. Every `--items` file is checked before the
first of them is run, not as its turn comes. A question no request could carry — a misspelt
`type`, a Choice whose `criteria` is not an object of options, a Score whose `criteria` is not an
array of levels — is reported first as a question-set fault naming the question, whether or not
the file labels it, so a correct label file is not blamed for it.

Split before tuning, keeping related records and near duplicates in the same split. Use dev for
wording, validation for model/threshold selection, and an untouched test set for final reporting.
If validation misses influence wording, validation has become development data. A two-way split
is sufficient only if the final set stays untouched until decisions are fixed.

Criteria `examples` come from a fourth, labelled pool that is disjoint from dev, validation and
test. Keep a paraphrase on the same side of the line as its source: on the fixture a
paraphrased example pulled an ambiguous item as hard as the literal string, so an example that
resembles a test item leaks by meaning, not only by text.

Review label quality against the task definition and available evidence. Represent ambiguity
explicitly (adjudication, acceptable labels, or a separate slice); this basic scorer accepts one
label per item, so document that limitation. Do not automatically relabel disagreement as `other`
or remove difficult items. Choose sample size from the required error bound and class coverage,
not a universal item count.

## 2. Check repeatability

```sh
python3 scripts/eval.py --questions Q.json --items val.jsonl --repeat 3
```

The report compares each later run with run 1. Identical outputs show repeatability on those
inputs; a small count difference on a finite set can still reflect sampling uncertainty.
Use paired item comparisons and suitable uncertainty estimates when deciding whether a variant
improves accuracy. Repeating a systematic error need not fix it; evaluate any voting strategy
instead of assuming it helps or cannot help. On a synthetic fixture, five repeats of 50 items
never changed a label; probabilities moved by up to 0.08, enough to cross a threshold.

## 3. Isolate changes

Compare state, questions, criteria, model, reference form, batch size, position, and composition.
For batch tuning, compare the single request to the batch template at size one first. Hold
positions constant or rotate them. Inspect neighbouring labels as one possible clue to errors.

`--permute-options` reverses Choice options in single mode and reports label flips and threshold
crossings. It tests order sensitivity without asserting the cause. It does not permute Score
levels, since their order defines the scale. Use the production option order for calibration.

## 4. Read the metrics and their denominators

`eval.py` reports:

- **Choice:** fine and optional coarse accuracy; per-class recall and precision for every option,
  printed as hits/labelled and hits/predicted with `-` for a zero denominator (the headline on an
  imbalanced label set, where overall accuracy can hide a class that is never caught), over all
  scored answers and again counting only answers at or above the policy threshold, the rates an
  automatic action at that threshold achieves; accepted/correct counts at the policy threshold;
  mean confidence; with `--verbose`, per-bucket and cumulative accuracy/coverage, batch position
  quarters and the miss list. An answer outside the option list never reaches the scorer: the
  client rejects the whole response, so in batched mode every item in that request counts as a
  failure, and in single mode the item's answers to its other questions are lost with it.
  `ablate.py` and `sweep_batch.py` print the option with the lowest recall in each cell (a recall
  tie goes to the option with more labelled items, which has lost more of them).
- **Noul:** caught positives, false positives, class counts, and at the threshold recall
  (caught/positives), precision (caught/(caught + false positives)) and true negative rate
  ((negatives − false positives)/negatives), each printed as `-` when its denominator is zero;
  lowest true and highest false outputs. These are threshold metrics, not a calibration analysis.
- **Score:** mean absolute error and predictions within half a rubric level.
- **Failures:** items without answers. Accuracy and buckets condition on successful, labelled
  answers; include failure rate when reporting end-to-end coverage.
- **Unlisted labels:** rows whose label the question cannot take, should any reach the scorer
  (the scripts refuse them before the first request; `jevlib.score` is also called on rows from
  elsewhere). They are listed as JSON under `unlisted labels` for every question type, scored as
  misses, and counted in no class, so every rate above is computed over labels the question can
  take.

`sweep_batch.py` compares each batch size with every earlier size on the items both of them
scored, so a failed batch neither raises a flag (the answers it lost are compared with nothing)
nor hides one, and names what fell: `fine` when three more items are wrong, `confidence` when mean
confidence is 0.05 lower, `<option> recall` when an option loses at least two items and at least
a quarter of its items, for a Noul `false positives` when more negatives reach the threshold,
`lowest true` when the lowest positive score falls by 0.1 and `highest false` when the highest
negative score rises by 0.2, and `last-quarter` when the last position quarter's accuracy is 0.15
below the first's. These are fixed heuristics, not significance tests.

The scorer does not apply a production policy, compute calibration curves/confidence intervals,
or implement review. Coarse accuracy maps all labels to parents; it does not establish the
correctness of a fine-above-threshold/coarse-below-threshold policy.

Token estimates use reported usage, excluding unreported usage of failed/retried attempts.
Request counts include failed logical evaluations, not individual retry attempts.
`ms/item` is amortized elapsed time for successful evaluations, including their retries, not a batched item's response latency or
end-to-end pipeline latency. Prices are estimates for the known resolved model only.

## 5. Choose and validate thresholds

`policy.json` example (illustrative thresholds, not recommendations):

```json
{"queue": {"threshold": 0.75, "parents": {"billing": "money", "sales": "money"}}}
```

Under perfect review, zero cost for a correct action, constant error cost `C_wrong > 0`, and
review cost `C_review`, act when `(1-p_correct) * C_wrong <= C_review`. For a $3 review and a $40
error, the break-even correctness probability is 0.925. With asymmetric errors, imperfect
review, or different utilities, use the appropriate expected-loss calculation instead.

The bucket table helps propose a confidence cutoff; it does not convert confidence into
correctness probability. Check cumulative sample counts and uncertainty for the selected region,
including the action/class being automated. The built-in table aggregates Choice labels; a
single overall threshold is insufficient evidence for action-specific error costs. Select using
tuning data and validate once on untouched data. If evidence is insufficient, gather more or
retain review. No fixed minimum bucket count proves a target precision.

Consume returned `confidence`; the [confidence docs](https://docs.typesafe.ai/confidence) do not
publish an exact formula. Options and request changes can affect the output. Measure calibration
and operating precision on representative traffic, including class prevalence.

## 6. Re-evaluate when inputs change

Recheck after question, criteria, state construction, model, batching, population, or policy-cost
changes. Log resolved model versions and avoid combining versions into one accuracy claim.
Preserve raw results and dataset versions so a later reviewer can reproduce the comparison.

## Input files

- **Items** (`--items`): JSONL as in section 1. `id` defaults to the item's line number in the file (from 1, counting comments and blank lines) and `expected` to `{}`.
- **Questions** (`--questions`): `{"model": "jev-1.13.0", "questions": {<id>: {"type": "choice" | "noul" | "score", "instructions": ..., "criteria": ...}}}`. A bare question map is also accepted, and `model` defaults to `jevlib.DEFAULT_MODEL`. Question bodies follow the TypeSafe API: a Choice's `criteria` is an object of option name to description, a Score's an array of levels, a Noul's optional. `ablate.py` removes the descriptive fields `not_for`, `examples`, `notes`, and `inspect` where present, and adds a "minus all" variant only when more than one is.
- **Batch template** (`--batch-template`): `{"array_field", "item_value"?, "shared_state"?, "reference"?, "questions", "model"?}`. Each request places up to N item values (the whole `state`, or `state[item_value]`) under `array_field` on top of `shared_state`, and `{ref}` in each question expands per item. `reference` is `keyed` (default: an object keyed `item000`.., ref `<array_field>.item017`), `quoted` (plain array, ref `<array_field>[17]`, with the item's JSON value appended to the question), or `index` (plain array, no quoting).
- **Policy** (`--policy`): section 5. Thresholds default to 0.75 for Choice and 0.8 for Noul when omitted.
