# Example: support-ticket triage

A self-contained Frame Engineering for Jev fixture that exercises every script. Labels are hand-written
and the messages are invented; the set exists to show the workflow, not to benchmark the model. It
lives outside the skill directory, so it is published with the repository but not installed with the
skill; these are the exact files the measurements below and in `SKILL.md` were taken on.

| file | purpose |
|---|---|
| `questions.json` | single-item question set: `queue` (Choice, 5 options with `what`/`not_for`/`examples`/`notes`), `urgent` (Noul), `contains_secret` (Noul) |
| `batch-template.json` | the same judgments for N items per request: queue definitions once in `shared_state.queues`, items keyed by id under `messages` (`"reference": "keyed"`), `{ref}` in every per-item question |
| `policy.json` | act thresholds and the roll-up parents (`billing`/`sales` → `money`, `technical`/`account_access` → `product`) |
| `dev.jsonl` | 30 labelled tickets for designing questions |
| `val.jsonl` | 20 labelled tickets for demonstrating evaluation (not an audited holdout) |

```sh
# from the repository root; every command calls the live API and is billed, and none writes a file here
S=skills/jev-frame-engineering/scripts; E=tests/fixtures/support-triage
python3 $S/eval.py --questions $E/questions.json --items $E/dev.jsonl --items $E/val.jsonl --policy $E/policy.json --verbose
python3 $S/eval.py --questions $E/questions.json --items $E/val.jsonl --policy $E/policy.json --repeat 3      # repeatability, not a confidence interval
python3 $S/eval.py --questions $E/questions.json --items $E/val.jsonl --policy $E/policy.json --permute-options  # options reversed: flips, threshold crossings
python3 $S/ablate.py --questions $E/questions.json --question queue --items $E/dev.jsonl --items $E/val.jsonl --policy $E/policy.json
python3 $S/eval.py --batch-template $E/batch-template.json --batch-size 8 --items $E/val.jsonl --policy $E/policy.json
python3 $S/sweep_batch.py --batch-template $E/batch-template.json --items $E/dev.jsonl --policy $E/policy.json --sizes 1,5,10,17,25
python3 $S/token_probe.py
```

The criteria examples are distinct from the fixture messages to reduce evaluation leakage.
Some messages contain synthetic secret-like values; `--verbose` and raw result files can show
them. None is a real credential: the card numbers are standard test numbers, the passwords are
invented, and the key and token strings are shorter than any real key of their prefix. The
secret Noul illustrates classification, not a complete redaction system.

The single-item and batch templates have different queue descriptions: the batch version uses
shared `what` definitions and omits exclusions/examples/notes. Compare them at batch size one
before attributing any result to batching. The sweep concatenates supplied files; use dev data
while selecting batch size. The small fixture and its example thresholds are not sufficient for
production calibration.

## Measured on this fixture (2026-09-20, jev-1.13.0, three repeats per condition)

Dev and val concatenated (50 items), one fixed shuffled order, the batch template with only
`reference` changed. `queue` is fine/N; `urgent` and `contains_secret` are correct at 0.5.
Reproduce a row with `eval.py --batch-template $E/batch-template.json --batch-size N --items
$E/dev.jsonl --items $E/val.jsonl --repeat 3` after setting `"reference"` in the template.

| N per request | keyed (shipped) | index | quoted |
|---|---|---|---|
| 8 | 50, 43, 50 | 49, 43-44, 50 | 48, 44, 50 |
| 16 | 50, 43, 50 | 46-47, 44, 50 | 48-49, 43, 50 |
| 32 | 49, 43, 50 | 38-40, 41-42, 49 | 48-49, 43, 50 |
| 48 (of 48) | 47, 42-43, 48 | **23-24**, 32-34, **28-29** | 46, 42-43, 48 |

Index at 48 by position quarter: 11, 8-9, 2-3, 2 of 12, mean confidence falling to 0.35 in the
tail. Input tokens per item at 48: index 408, keyed 416, quoted 484. The same 16-item keyed
batches with 16, 48 or 240 questions per request scored 47-48/48 every time.

Other runs on the fixture the same day, with the raw request and response bodies kept outside
the distribution: two injection strings inserted third in a 32-item array left keyed and quoted
neighbours at 28-29/29 and pushed index neighbours from 17/29 to 10-12/29; five repeats of the
single-item question set never changed a label; pointer criteria at batch size one cost the
`urgent` Noul 4-5/50 in each repeat against inline `what` criteria. The single-item question set
scored `queue` 47/50, `urgent` 47-48/50, `contains_secret` 50/50.

The fixture is small and its labels are one author's; these numbers show the workflow and the
shape of each effect, not production accuracy or a threshold for another domain.
