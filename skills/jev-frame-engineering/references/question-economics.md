# Question economics

## Billing and measurement

The [Models page](https://docs.typesafe.ai/models) lists `jev-1.13.0` at $0.042 per million input
tokens, with free output tokens (reviewed 2026-09-20). Check current pricing before budgeting;
the helper's dollar estimate applies only to a known resolved model and excludes usage missing
from failed or retried calls.

`token_probe.py` compares synthetic requests to estimate a baseline, incremental question cost,
and text token density. Its fitted baseline includes the small test state and serialization;
it is not a measurement of pure protocol overhead. Its prose and criteria slopes use different
text, so they cannot show that identical content is billed differently in those fields. Character
ratios depend on language, text, and tokenizer; use API `usage.input_tokens` for real requests.

## Ablate descriptions without making brevity a goal by itself

Begin with a clear question and suitable options/levels. Add structured exclusions, examples,
and notes when they express the desired boundary. TypeSafe supports strings and structured
content; `what` and `not_for` are conventions, not required schema keys.

`ablate.py` removes each requested field from top-level instruction objects and option/level
description objects, plus a variant removing them all. It sends only the target question and
runs each variant once per supplied dataset. Repeat the script for a repeatability check. It
is not a recursive field-removal tool. An absent field is skipped rather than reported as a
measured null effect.

Select changes on tuning data, then report once on untouched data. Examples can clarify a
concept; their effect is task-dependent. Avoid copied or near-duplicate evaluation cases in
criteria, and do not use higher confidence alone as evidence of higher accuracy.

## Supply the missing evidence

Policies, private rosters, and live facts must be included when the answer depends on them.
Compute exact arithmetic and lookups in code. Separate item-only judgments from judgments about
a relationship when they serve distinct decisions; cache the latter with all relevant context.

## Share repeated definitions carefully

Move a repeated taxonomy into shared state and reference it by name to test a token reduction.
The model must still resolve that reference. Compare accuracy at batch size one against inline
criteria, then sweep larger batches. Savings and accuracy retention are measurements to make,
not fixed multipliers to inherit. See [batching](batching.md).

## Compare fan-out with staged routing

[Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out) asks independent branch questions
together, then code selects the relevant answers. State each speculative premise clearly.
Unread questions still consume input tokens. Routing first can avoid those questions, but it adds
a sequential call and may resend a large state. Measure total cost, error propagation, and
end-to-end latency across the entire pipeline. A second call is necessary when the first answer
changes the evidence or candidate set needed next.
