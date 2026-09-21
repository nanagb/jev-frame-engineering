# Production contract

## Errors are not uncertainty

The [HTTP API](https://docs.typesafe.ai/api) documents 401 (credentials), 422 (body validation),
429 (rate limit), and 529 (overload). The helper also handles ordinary HTTP/transport failures.

| condition | helper behavior |
|---|---|
| non-transient HTTP status, including 400/401/403/404/422 | `JevError`, no retry; fix request/configuration |
| 408, 429, 5xx including 529 | bounded retries with exponential backoff and `Retry-After` |
| transport error or timeout | bounded retries, then `JevNoJudgment` |
| invalid JSON or response missing usable answers | `JevNoJudgment`; do not manufacture a label |

`JevNoJudgment` is a subclass of `JevError`; catch it first if handling them separately. The
runners preserve it as an unanswered item, while non-retryable errors stop the run. Transient
failures do occur: in about 1,200 requests on 2026-09-20, seven ended in a dropped connection or
a 200 body that failed validation, and every one succeeded when resent. The helper retries the
first kind and not the second, so plan a resend for unanswered items. A failure
has no uncertainty score. Use an explicit failure/review state distinguishable from a real
`false`, `other`, or low-confidence answer.

The helper defaults to two retries and a 20-second timeout per attempt. The default maximum permitted retry wait is 30 seconds: when a server asks for longer, return no judgment for later
rescheduling instead of retrying too early. Both delay-seconds and HTTP-date `Retry-After` are
accepted. This is not a complete end-to-end deadline; production callers must also budget total
elapsed time and concurrency. Prefer TypeSafe's SDK when its built-in behavior fits.

Error bodies can echo request data. The helper removes the active credential and bearer tokens
before constructing errors, but this is not general PII/secret detection. Avoid logging raw bodies
or evaluation state in production.

## Retries and side effects

Each retry can repeat evaluation; a timeout does not prove that the server did no work.
Make downstream effects idempotent using an item/operation ID and processing version. Enforce
allowed operations and authorization in code independently of model confidence.

## Versioning and caching

- Pin the model ID after validating thresholds; aliases track releases. Record the resolved
  response `model` and evaluate a release before moving a pinned integration.
- Version question text/criteria and state-building code. Retain raw judgments so changing only
  downstream weights or thresholds can reuse them when the evidence and question meaning agree.
- Cache by `(input-content digest, question_id, question_version, resolved_model)` or equivalent
  content versions. Include policies, rosters, related entities, and all other answer-relevant
  state. Subject identity alone does not protect against edited content or policy changes.
- For an alias, resolve/version the entry before reuse; a cache key containing only `jev-latest`
  cannot distinguish releases. Do not silently pool several resolved models into one evaluation.

## Untrusted input and privacy

Separate user content from trusted application facts. Named fields and separate requests can
make boundaries easier to test but do not guarantee resistance to injection. TypeSafe documents
[adversarial content](https://docs.typesafe.ai/model-jaggedness/jev-1.13) as a steering risk.
Test attempted influence on the target answer and other items sharing its state. A detector
Noul is a model prediction, not an authorization check or a complete secret-redaction system.

Send only relevant data consistent with the application's data policy. Keep credentials
server-side. These scripts read `TYPESAFE_API_KEY` or `~/.config/typesafe/api_key`; use mode 600
for the file (not enforced by the loader). `--json` and verbose miss reports can contain original
input, including synthetic or real secrets; credential error sanitization does not sanitize that
evaluation data.
