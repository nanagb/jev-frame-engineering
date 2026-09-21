# Pre-ship checks

Apply the checks relevant to the integration. These are review prompts, not extra API rules.

**Evaluation**
- [ ] Tuning and final evaluation data are separate, with representative classes and hard cases.
- [ ] Comparisons isolate intended changes; repeats and sampling uncertainty are distinguished.
- [ ] Metrics cover the actual action policy, failures, coverage, and class-specific errors.
- [ ] Option-order sensitivity was considered; a flip was investigated without assuming its cause.

**Questions**
- [ ] Choice covers expected inputs, with no-match where needed; multiple applicable labels use independent judgments when appropriate.
- [ ] Noul boundaries are clear; Score levels describe standalone situations on one dimension.
- [ ] Criteria examples do not duplicate evaluation cases; all required context is available.
- [ ] Exact calculations and authorization checks remain in code.
- [ ] Questions do not require generation or another answer from the same call.

**Batching and cost**
- [ ] Instructions identify each item; IDs used only as response keys do not supply meaning.
- [ ] Template changes were checked at size one before sizing; position and composition effects were tested.
- [ ] Both context budgets, rate limits, full-request latency, and failure handling were considered.
- [ ] Shared definitions and staged routing were measured for quality and whole-pipeline cost.

**Untrusted content**
- [ ] Adversarial inputs were tested; quoting, keys, and request isolation are not treated as guarantees.
- [ ] Code enforces allowed effects, authorization, and required confirmations.
- [ ] Model-based detection is not the only security or redaction boundary.

**Thresholds**
- [ ] Thresholds are validated for each action and population, with adequate uncertainty bounds.
- [ ] The cost model states its assumptions; confidence is not substituted for correctness probability.
- [ ] Fallback/coarse decisions are independently validated.
- [ ] Calibration is reassessed after question, option, state, model, or population changes.

**Operations**
- [ ] Non-retryable errors stop; transient retries are bounded; failed calls remain visibly unanswered.
- [ ] Downstream effects are idempotent; deployed models and questions are versioned.
- [ ] Cache keys change when relevant content/context changes and identify the resolved model.
- [ ] Credentials and sensitive evaluation data are excluded from inappropriate logs and artifacts.
