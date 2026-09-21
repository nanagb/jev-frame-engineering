# Frame Engineering for Jev

*Designing and evaluating reliable decision frames.*

**Decision Frame Engineering**, or **Frame Engineering**, is the design, testing, and refinement
of the evidence, questions, and answer boundaries that shape a model's judgments. This repository
applies that practice to existing integrations with TypeSafe's System One models (Jev).

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

### Accuracy and evidence

Measurements cited in the skill were taken on a synthetic support-triage fixture of 50 labelled
tickets, not included here, against `jev-1.13.0` on 2026-09-20 with three repeats per condition.
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
