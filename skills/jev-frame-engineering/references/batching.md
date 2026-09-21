# Batching

All questions in a request see the same state independently. More questions over unchanged state
and more unrelated items in state are different experiments. TypeSafe documents both parallel
question evaluation and degradation from irrelevant state; neither establishes an optimal batch
size for your workload. See [fan-out](https://docs.typesafe.ai/patterns/fan-out) and
[Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

## Reference forms

| form | state shape | reference in instructions | tradeoff |
|---|---|---|---|
| `keyed` (helper default) | object `{"item000": …, "item017": …}` | `messages.item017` | explicit name; extra key text |
| `quoted` | array | `messages[17]` plus JSON-rendered item | repeats content in questions; cost grows with item size |
| `index` | array | `messages[17]` | compact; test resolution across positions |

These are natural-language references, not enforced pointers. Official examples use numeric
paths. On a synthetic 50-ticket fixture (three repeats, 2026-09-20) index references lost accuracy from 16
items and reached 50% at 48 while keyed and quoted held; the onset depends on item length and
domain, so it is a reason to test the shipped form, not a universal cutoff. Quoting and naming
are not an isolation guarantee for untrusted content, though on the fixture they removed the
cross-item effect of two injection strings (SKILL.md rule 4).

## Template shape

```json
{
  "model": "jev-1.13.0",
  "array_field": "messages",
  "item_value": "message",
  "reference": "keyed",
  "shared_state": {"queues": {"billing": "Payments", "other": "Anything else"}},
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which queue fits the ticket at `{ref}`, using `queues`?",
      "criteria": {"billing": "As defined in `queues.billing`", "other": "As defined in `queues.other`"}
    }
  }
}
```

`array_field`, `item_value`, `reference`, and `shared_state` are helper configuration, not API
fields. The runner constructs API `state` by combining shared fields with the item collection;
it repeats each question per item, substitutes `{ref}`, and sends only `state`, `model`, and
`questions`. Question IDs such as `queue__17` match responses but are not visible to the model.

`item_value` selects one field of each item's state; omit it to include the whole state. Do not
omit context a relational judgment needs. Keyed mode refuses a template containing `[{j}]`;
use `{ref}`. Quoted mode JSON-serializes strings, objects, and arrays into referenced question
text. Choose an `array_field` that does not collide with a shared-state field.

## Compare size without confounding question changes

1. Compare the original single-item request with the template at size one. If moving criteria
   into shared state changes accuracy, resolve that before attributing differences to batch size.
2. Sweep increasing sizes in the intended reference mode on tuning data. Rotate positions and
   batch composition as needed. Keep final evaluation data separate.
3. Inspect per-question errors, answer failures, position quarters, usage, and request latency.
   A tail drop suggests investigation; it does not identify the cause by itself.

`sweep_batch.py` concatenates all supplied item files into one sweep. Do not mix a final holdout
into that sweep while selecting size. Its flags use fixed heuristics and are not significance
checks. Compare the same successfully judged items if failure rates differ between conditions.

## Sharing definitions and budgets

Moving repeated criteria into shared state can reduce input tokens. It changes the question's
indirection and must be evaluated against the inline version. Do not assume bare `null` criteria
will retrieve a definition unless the instructions clearly connect the option to that context.

Check [Models](https://docs.typesafe.ai/models) for current budgets. As of the review,
`jev-1.13.0` allows 64k tokens for the complete request and 32k for state plus the longest
question. A batch may fit one limit and exceed the other. Check usage rather than estimating
exact token counts from character counts. Request latency divided by batch size is amortized
service time per item; each item still waits for the complete request.
