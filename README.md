# Frame Engineering for Jev

*Designing and evaluating reliable decision frames.*

**Decision Frame Engineering**, or **Frame Engineering**, is the design, testing, and refinement
of the evidence, questions, and answer boundaries that shape a model's judgments. This repository
applies that practice to existing integrations with TypeSafe's System One models (Jev).

[Understand the practice](#understanding-frame-engineering) · [The skill](#the-skill) · [Install](#install) · [Use it](#use-the-skill) · [Run the scripts](#run-the-scripts-directly)

## Understanding Frame Engineering

Here, Frame Engineering names the practice of designing the whole decision: what the model
sees, what it judges, and how the application acts on the result. Its central concern is whether
the model is judging the right thing from the right evidence. A confident answer to an ambiguous
question can still produce the wrong business outcome.

### A frame defines the decision

A decision frame connects four parts:

| Part | What it establishes | Example |
|---|---|---|
| Evidence and context | The facts and policies available to the model | A customer message and the definitions of the support queues |
| Question | The specific judgment and its subject | Which queue fits this customer's reported problem? |
| Answer boundaries | What answers mean and which distinctions matter | Billing, technical support, account access, or another issue |
| Application policy | How a judgment becomes an action | Route automatically only when the routing policy permits it; otherwise send for review |

In Jev, evidence and contextual policies belong in `state`; question `instructions` specify what
to judge. All questions in a request see the same state and are evaluated independently. If one
decision needs the result of another, the application must arrange that dependency explicitly.
See TypeSafe's [state model](https://docs.typesafe.ai/concepts/state).

The answer type is a design choice too. **Noul** represents a yes/no judgment as a probability;
**Choice** selects among alternatives; **Score** places a judgment on an ordered rubric. Use
separate questions for independent properties, and a Choice when the task requires selecting
one alternative. A Score needs meaningful levels, such as observable degrees of service impact,
so that different readers can understand what a higher value means.
See the [Noul](https://docs.typesafe.ai/primitives/noul),
[Choice](https://docs.typesafe.ai/primitives/choice), and
[Score](https://docs.typesafe.ai/primitives/score) definitions.

### One message can support different judgments

Consider this fictional support message:

> I was charged twice, but I only need a corrected invoice by Friday.

“Handle this ticket” leaves several decisions mixed together: identifying the problem,
interpreting the request, checking eligibility, and choosing an action. A clearer frame separates
those decisions:

| Question | Answer shape | Expected reading of this example |
|---|---|---|
| Which support queue fits the problem reported in this message? | Choice with defined queue options | Billing |
| Does the customer explicitly request a refund in this message? | Noul with criteria for an explicit refund request | No |

These are illustrative labels, not measured model outputs. They are compatible: a billing
problem does not imply a refund request. Likewise, “no refund requested” says nothing about
refund eligibility. That requires different evidence and a different decision.

This distinction also changes what belongs in the frame. Queue definitions help routing;
transaction records may help establish what happened. Unrelated account history may add cost
and distraction. The application should verify amounts, dates, permissions, and eligibility
rules in code wherever those checks are deterministic. A model's interpretation of the message
does not authorize moving money.

### Context and batching shape the evidence

More context is useful when it supplies missing evidence or resolves a relevant ambiguity.
More wording is useful when it clarifies a distinction. Neither is inherently an improvement:
extra examples can narrow a category unintentionally, and overlapping options can make the
intended answer unclear even to a human reviewer.

Batching introduces another design concern. A question must identify its own item explicitly
when several tickets share a state. Clear references, shared definitions, batch composition,
and item position all deserve evaluation because the model now sees neighbouring content too.
Independent questions are not a guarantee that unrelated content cannot influence an answer.

Keep customer content distinguishable from application policy, and test messages that try to
steer the judgment. Field names and delimiters help describe the boundary; they do not enforce
it. TypeSafe documents [adversarial content](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
as a model limitation. Authorization and allowed actions remain application responsibilities.

### Confidence, missing categories, and failures are different

Three situations require different handling:

| Situation | Meaning | Application consequence |
|---|---|---|
| No listed category fits | The available alternatives do not cover the item | An explicit no-match option can represent this legitimate outcome |
| The returned judgment is uncertain | The model does not strongly favour an answer | A validated policy may defer the item for review |
| The request fails or returns no usable answer | No judgment is available | Preserve a failure state and apply retry or recovery policy |

An `other` category is not automatically an uncertainty signal, and a failed request is not a
negative answer. For Noul, a probability near 0.5 expresses uncertainty between yes and no;
it does not mean “moderately true.” For Choice and Score, returned confidence describes the
concentration of the answer distribution. It is not proof of correctness. Thresholds need
validation against the errors and consequences of the actual task, as explained in TypeSafe's
[confidence guidance](https://docs.typesafe.ai/confidence).

### What makes this engineering

A frame is a testable design. A clear task definition and representative labelled examples make
its quality assessable, including difficult cases and cases where the right outcome is review.
Disagreement between human reviewers can reveal an unclear definition before any model tuning
begins.

Controlled comparisons reveal which changes help. Adding policy context might reduce one kind
of error while introducing another. Changing a decisive fact should change the answer when the
task demands it; irrelevant wording or a different batch position should preserve the intended
judgment. Repeating the same mistake consistently demonstrates repeatability, not correctness.

Improvement therefore depends on outcomes: correct decisions, errors by category, how much work
can be handled automatically, unanswered requests, cost, and latency. More confident outputs
alone do not establish improvement. An untouched final evaluation set helps distinguish gains
that generalize from adjustments that merely fit familiar examples. A change in model, input
population, or application policy can invalidate earlier evidence and calls for re-evaluation.

For practical work, use the [evaluation protocol](skills/jev-frame-engineering/references/eval-protocol.md),
[batching guide](skills/jev-frame-engineering/references/batching.md), and
[production contract](skills/jev-frame-engineering/references/production-contract.md).

## The skill

[`jev-frame-engineering`](skills/jev-frame-engineering/SKILL.md) covers state and question design,
batch evaluation, and action thresholds, with supporting guidance for production behavior.
Use [TypeSafe's official skill](https://github.com/typesafe-ai/skills/tree/main/skills/typesafe-ai)
or [live docs](https://docs.typesafe.ai/introduction) for the programming model and initial integration.

| practice | focus |
|---|---|
| 1 | labelled data, controlled comparisons, repeatability, untouched final evaluation |
| 2 | simple descriptions, necessary context, optional examples and no-match outcomes |
| 3 | explicit item references, batch-size and position tests, context budgets |
| 4 | adversarial-input tests and action policy enforced in code |
| 5 | measured shared definitions and fan-out/routing tradeoffs |
| 6 | action-specific thresholds and calibration on representative data |
| 7 | explicit failures, bounded retries, model/input versioning and caching |

`scripts/` uses Python 3 with no dependencies: `eval.py`, `ablate.py`, `sweep_batch.py`,
`token_probe.py`, and the shared `jevlib.py`.

### Install

In Claude Code, add this repository as a plugin marketplace and install the plugin:

```text
/plugin marketplace add nanagb/jev-frame-engineering
/plugin install jev-frame-engineering@blue-terra
```

Alternatively, from the repository root, link the skill into the directory used by your agent:

```sh
# Claude Code
mkdir -p ~/.claude/skills
ln -s "$PWD/skills/jev-frame-engineering" ~/.claude/skills/jev-frame-engineering

# Codex
mkdir -p ~/.codex/skills
ln -s "$PWD/skills/jev-frame-engineering" ~/.codex/skills/jev-frame-engineering
```

A copy works too; a symlink keeps the installed skill and repository in step.

### Use the skill

The skill is model-invoked. Claude Code reads every installed skill's name and description at
startup and loads the body when a request matches, so there is nothing to run; start a new
session after installing so the skill is discovered.

Requests that should load it:

- “Split this Jev request into separate questions and say what each one judges.”
- “Does adding `examples` to the `queue` criteria help? Ablate it against my labelled set.”
- “Routing is fine at 8 tickets per request and poor at 32 — what should I test?”
- “Choose an auto-routing threshold given a $3 review and a $40 wrong route.”

To load it deliberately, name it: in Claude Code type `/jev-frame-engineering:jev-frame-engineering`
after a plugin install (plugin skills are namespaced `plugin:skill`) or `/jev-frame-engineering`
after a symlink install, or write “use the jev-frame-engineering skill” in the request.

Have ready: a TypeSafe integration that already returns answers, the question set and state it
sends, labelled items for the decisions in question, and a credential for the scripts. The skill
works on an existing frame; for the programming model and a first integration, use
[TypeSafe's official skill](https://github.com/typesafe-ai/skills/tree/main/skills/typesafe-ai).

A session usually runs in this order: state the decision and what acting on it costs; separate
the judgments and choose an answer type for each; assemble labelled development and validation
sets and take a baseline with `eval.py`; change one axis at a time (`ablate.py` for criteria
fields, `sweep_batch.py` for batch size, position, and reference form); then select per-action
thresholds from the error and review costs and check them once on untouched data. The skill asks
for the evidence behind each step rather than treating a reported improvement as established.

### Run the scripts directly

The scripts are the skill's helpers and also work on their own. They live in
`skills/jev-frame-engineering/scripts/` in a clone,
`~/.claude/skills/jev-frame-engineering/scripts/` under a symlinked install, and
`~/.claude/plugins/cache/blue-terra/jev-frame-engineering/<version>/skills/jev-frame-engineering/scripts/`
under a plugin install. Each prints its usage with `--help` (`jevlib.py` is the library they
share), and the file formats are specified in the
[evaluation protocol](skills/jev-frame-engineering/references/eval-protocol.md).

Keep question sets, labels and raw output in a directory of your own and run the scripts by path
from there. Under a clone or a symlinked install the skill directory is this repository, so
labelled data or `--json` output written inside it would sit beside the skill as untracked files
that a broad `git add` would publish.

A run needs a question set and labelled items. `questions.json`:

```json
{
  "model": "jev-1.13.0",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "`message` is a support ticket. Which queue should handle it?",
      "criteria": {
        "billing": {"what": "charges, invoices, refunds", "not_for": "pre-sales questions"},
        "technical": {"what": "the product misbehaves"},
        "other": {"what": "anything the options above do not cover"}
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does `message` need attention today?",
      "criteria": {"true": "an outage or a same-day deadline", "false": "a routine request"}
    }
  }
}
```

`dev.jsonl`, one labelled item per line:

```json
{"id": "t01", "state": {"message": "I was charged twice this month."}, "expected": {"queue": "billing", "urgent": false}}
{"id": "t02", "state": {"message": "Export has failed with a 500 since this morning."}, "expected": {"queue": "technical", "urgent": true}}
```

```sh
export TYPESAFE_API_KEY=...                        # or ~/.config/typesafe/api_key, mode 600
S=~/.claude/skills/jev-frame-engineering/scripts   # or the clone or plugin path above
python3 "$S/eval.py" --questions questions.json --items dev.jsonl --repeat 3
```

`eval.py` checks the question set and every label before its first request, and refuses a file
whose values a question cannot take, saying what each question takes. It then reports, separately
for each `--items` file, per-class recall and precision over all answers and at the policy
threshold, failures, and token use; `--verbose` adds confidence buckets and the miss list, and
`--json out.json` keeps the raw answers. `ablate.py` makes the same checks and prints one column
per `--items` file, each cell ending with the Choice option with the lowest recall, since overall
accuracy can hold while one class collapses, and compares each variant's token cost with the full
question on the items both answered:

```sh
python3 "$S/ablate.py" --questions questions.json --question queue --items dev.jsonl
```

`sweep_batch.py` puts several items in each request, so it takes a batch template instead of a
question set: the same questions, pointed at each item by `{ref}` (see the
[batching guide](skills/jev-frame-engineering/references/batching.md)). `batch-template.json` for
the files above:

```json
{
  "model": "jev-1.13.0",
  "array_field": "messages",
  "item_value": "message",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "The ticket at `{ref}` is a support ticket. Which queue should handle it?",
      "criteria": {
        "billing": {"what": "charges, invoices, refunds", "not_for": "pre-sales questions"},
        "technical": {"what": "the product misbehaves"},
        "other": {"what": "anything the options above do not cover"}
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does the ticket at `{ref}` need attention today?",
      "criteria": {"true": "an outage or a same-day deadline", "false": "a routine request"}
    }
  }
}
```

```sh
python3 "$S/sweep_batch.py" --batch-template batch-template.json --items dev.jsonl --sizes 1,8,16,32
python3 "$S/token_probe.py"
```

The sweep runs its `--items` files as one set, and flags a size whose results fell against an
earlier size, compared on the items both of them scored.

Each of these calls the live API with your credential and is billed by TypeSafe; `--repeat` and
`--sizes` multiply that cost. The offline tests in `tests/` are the part of the repository that
runs without a key.

### Accuracy and evidence

Measurements cited in the skill were taken on a synthetic support-triage fixture of 50 labelled
tickets against `jev-1.13.0` on 2026-09-20 with three repeats per condition. The fixture is in
[`tests/fixtures/support-triage/`](tests/fixtures/support-triage/README.md) with the commands that
reproduce each number; it is published with the repository but not installed with the skill.
Performance claims beyond that fixture are hypotheses to evaluate on your own data. Offline
tests in `tests/` verify helper behavior, not live model quality; they are published with the
repository but are not part of the skill.

```sh
python3 -m unittest discover -s tests
```

### Credentials and output

Scripts read `TYPESAFE_API_KEY` or `~/.config/typesafe/api_key`. Set the file to mode 600; the
loader does not enforce its mode. Client error messages redact the active credential and bearer
tokens. Raw result files and verbose misses contain input data and are not generally redacted.

## Licence

MIT. The skill directory carries its own copy so it remains self-contained when copied.
