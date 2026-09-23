"""Shared helpers for the jev-frame-engineering scripts.

Client errors sanitize the active API key. Evaluation reports may contain raw input.
Errors are split into JevError (non-retryable request/configuration failure) and
JevNoJudgment (no usable response; never substitute a model answer).
"""
import json, math, os, re, statistics, sys, time, urllib.error, urllib.request
from collections import Counter
from email.utils import parsedate_to_datetime

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"          # pinned; aliases (jev-latest) move between releases
PRICE_PER_MTOK = 0.042                # jev-1.13.0, checked 2026-09-20; output free
PRICES_PER_MTOK = {DEFAULT_MODEL: PRICE_PER_MTOK}
BUCKETS = [0.9, 0.75, 0.6, 0.0]


class JevError(Exception):
    """The request or credentials are wrong. Surface it; do not retry."""


class JevNoJudgment(JevError):
    """Transient failure or invalid response. There is no usable answer."""


def load_key(path=None):
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        p = os.path.expanduser(path or "~/.config/typesafe/api_key")
        if os.path.exists(p):
            with open(p) as f:
                key = f.read().strip()
    if not key:
        raise JevError("no API key: set TYPESAFE_API_KEY or create ~/.config/typesafe/api_key (chmod 600)")
    return key


def _retry_wait(header, fallback):
    if header:
        try:
            seconds = float(header)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(header).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                return fallback
        if math.isfinite(seconds):
            return max(0.0, seconds)
    return fallback


def _number(value, low=0, high=1):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and low <= value <= high)


def _validate_response(data, questions):
    """Reject incomplete or unusable results before scoring or applying a judgment."""
    try:
        if not isinstance(data["model"], str) or not data["model"]:
            raise ValueError()
        for field in ("input_tokens", "output_tokens"):
            value = data["usage"][field]
            if type(value) is not int or value < 0:
                raise ValueError()
        for qid, q in questions.items():
            a = data["answers"][qid]
            if a["type"] != q["type"]:
                raise ValueError()
            if q["type"] == "noul":
                if not _number(a["noul"]):
                    raise ValueError()
                continue
            expected = (set(q["criteria"]) if q["type"] == "choice"
                        else {str(i) for i in range(len(q["criteria"]))})
            probs = a["probabilities"]
            if (set(probs) != expected or not all(_number(p) for p in probs.values())
                    or not math.isclose(sum(probs.values()), 1, abs_tol=0.01)
                    or not _number(a["confidence"])):
                raise ValueError()
            if q["type"] == "choice":
                if a["choice"] not in expected:
                    raise ValueError()
            elif q["type"] == "score":
                if not _number(a["score"], 0, len(expected) - 1) or set(a["legend"]) != expected:
                    raise ValueError()
            else:
                raise ValueError()
    except (KeyError, TypeError, ValueError, AttributeError):
        raise JevNoJudgment("invalid response: missing or malformed model, usage, or answers") from None


def post(state, questions, model=DEFAULT_MODEL, key=None, timeout=20.0, max_retries=2,
         max_retry_wait=30.0):
    """One evaluation. Returns (response_json, latency_ms). Retries 408/429/5xx with
    exponential backoff honouring Retry-After within max_retry_wait. Other HTTP
    statuses are not retried. Latency includes retries and waits on success."""
    if type(max_retries) is not int or max_retries < 0 or not _number(max_retry_wait, 0, float("inf")):
        raise JevError("max_retries must be a nonnegative integer and max_retry_wait finite and nonnegative")
    key = key or load_key()
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    delay = 1.0
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(API_URL, data=body, method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
                _validate_response(data, questions)
                return data, (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            text = redact(e.read().decode(errors="replace"), key)[:500]
            if e.code not in (408, 429) and not 500 <= e.code < 600:
                raise JevError(f"HTTP {e.code} (not retried; fix the request or key): {text}") from None
            if attempt == max_retries:
                raise JevNoJudgment(f"HTTP {e.code} after {max_retries} retries: {text}") from None
            ra = e.headers.get("retry-after") if e.headers else None
            wait = _retry_wait(ra, delay)
            if wait > max_retry_wait:
                raise JevNoJudgment(f"HTTP {e.code}: retry wait exceeds budget; reschedule later") from None
            time.sleep(wait); delay *= 2
        except (json.JSONDecodeError, UnicodeError):
            raise JevNoJudgment("invalid JSON response; no judgment") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == max_retries:
                raise JevNoJudgment(f"no response after {max_retries} retries: {redact(str(e), key)}") from None
            if delay > max_retry_wait:
                raise JevNoJudgment("retry wait exceeds budget; reschedule later") from None
            time.sleep(delay); delay *= 2


# ---------- files ----------

def load_json(path):
    with open(os.path.expanduser(path)) as f:
        return json.load(f)


def load_items(path):
    """JSONL: {"id": ..., "state": <string|object|array>, "expected": {question_id: value}}."""
    items = []
    with open(os.path.expanduser(path)) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            if "state" not in d:
                raise JevError(f"{path} line {i+1}: item has no 'state'")
            d.setdefault("id", str(i)); d.setdefault("expected", {})
            if not isinstance(d["expected"], dict):
                raise JevError(f"{path} line {i+1}: 'expected' must be an object of question id to label")
            items.append(d)
    return items


def questions_of(qset):
    """Accept either {"questions": {...}, "model": ...} or a bare question map."""
    return qset["questions"] if "questions" in qset else qset


# ---------- runners ----------

def run_single(items, qset, model=None, sleep=0.05, key=None):
    """One request per item, all questions fanned out on that item's state."""
    qs = questions_of(qset); model = model or (qset.get("model") if isinstance(qset, dict) else None) or DEFAULT_MODEL
    key = key or load_key(); out = []
    for it in items:
        try:
            d, ms = post(it["state"], qs, model, key)
        except JevNoJudgment as e:
            out.append({**it, "answers": None, "error": str(e), "tokens": 0, "ms": 0, "requests": 1}); continue
        out.append({**it, "answers": d["answers"], "model": d.get("model"),
                    "tokens": d["usage"]["input_tokens"], "ms": ms, "requests": 1, "questions_per_request": len(qs)})
        if sleep:
            time.sleep(sleep)
    return out


def _subst(obj, j, ref, quote=None):
    """Substitute {ref} (the item's path) and the legacy {j} (its index) in every string of a question; in
    quoted mode also embeds the item's JSON value in the question."""
    if isinstance(obj, str):
        out = obj.replace("{ref}", ref).replace("{j}", str(j))
        if quote is not None and ("{ref}" in obj or "{j}" in obj):
            out += f' The JSON value at that reference is: {quote}'
        return out
    if isinstance(obj, dict):
        if quote is not None and "question" in obj and isinstance(obj["question"], str):
            d = {k: _subst(v, j, ref) for k, v in obj.items()}
            d["question"] = d["question"] + f' The JSON value at that reference is: {quote}'
            return d
        return {k: _subst(v, j, ref, quote) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, j, ref, quote) for v in obj]
    return obj


def run_batched(items, template, n, model=None, sleep=0.05, key=None):
    """N items per request. template = {"array_field", "item_value"?, "shared_state"?, "reference"?, "questions"}.
    "reference" names the items: "keyed" (default) puts them in an object keyed item000.. and {ref} expands to
    `<array_field>.item017`; "quoted" keeps a plain array, {ref} expands to `<array_field>[17]`, and each
    question gets the item's JSON value appended; "index" is the plain-array form with no quoting.
    Reference accuracy must be measured on the target workload. Legacy {j} still expands to
    the index, but a template that writes `[{j}]` cannot be used in keyed mode and is refused with a message."""
    if type(n) is not int or n <= 0:
        raise JevError("batch size must be a positive integer")
    model = model or template.get("model", DEFAULT_MODEL)
    arr = template["array_field"]; pick = template.get("item_value"); shared = template.get("shared_state", {})
    if arr in shared:
        raise JevError("array_field would overwrite a shared_state field")
    mode = template.get("reference", "keyed")
    if mode not in ("keyed", "quoted", "index"):
        raise JevError(f'template "reference" must be keyed, quoted or index, not {mode!r}')
    tq = template["questions"]; key = key or load_key(); out = []
    if mode == "keyed" and "[{j}]" in json.dumps(tq):
        raise JevError('template uses `[{j}]` (numeric index) but "reference" is keyed: write `{ref}` instead, or set "reference": "index"')
    for i in range(0, len(items), n):
        chunk = items[i:i + n]
        values = [(it["state"][pick] if pick else it["state"]) for it in chunk]
        state = dict(shared)
        if mode == "keyed":
            state[arr] = {f"item{j:03d}": v for j, v in enumerate(values)}
            refs = [f"{arr}.item{j:03d}" for j in range(len(chunk))]
        else:
            state[arr] = values
            refs = [f"{arr}[{j}]" for j in range(len(chunk))]
        qs = {f"{qid}__{j}": _subst(q, j, refs[j], json.dumps(values[j], ensure_ascii=False) if mode == "quoted" else None)
              for j in range(len(chunk)) for qid, q in tq.items()}
        try:
            d, ms = post(state, qs, model, key)
        except JevNoJudgment as e:
            for it in chunk:
                out.append({**it, "answers": None, "error": str(e), "tokens": 0, "ms": 0, "requests": 1 / len(chunk)})
            continue
        for j, it in enumerate(chunk):
            ans = {qid: d["answers"][f"{qid}__{j}"] for qid in tq}
            out.append({**it, "answers": ans, "model": d.get("model"), "tokens": d["usage"]["input_tokens"] / len(chunk),
                        "ms": ms / len(chunk), "requests": 1 / len(chunk), "batch": i // n, "pos": j, "batch_len": len(chunk),
                        "questions_per_request": len(qs), "reference": mode})
        if sleep:
            time.sleep(sleep)
    return out


# ---------- scoring ----------

def subject_of(r, width=52):
    """A short readable form of an item's state, for miss lines. Printing only an id makes the
    reader write a lookup script before they can triage anything."""
    st = r.get("state")
    if isinstance(st, dict):
        vals = [str(v) for v in st.values() if isinstance(v, (str, int, float))]
        st = " | ".join(vals) if vals else json.dumps(st)
    elif not isinstance(st, str):
        st = json.dumps(st)
    st = " ".join(str(st).split())
    return st[:width - 1] + "…" if len(st) > width else st


def _mean(xs):
    xs = list(xs); return statistics.mean(xs) if xs else None


def _label(v):
    """A label-file value as JSON text, for an error message or the report's unlisted-labels row. Distinct
    values stay distinct: 1, 1.0 and true share one dict key but are three labels."""
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


LABEL_TYPES = ("choice", "noul", "score")


def question_defect(q):
    """Why a question cannot be label-checked at all, or None. A misspelt `type` or a Choice/Score with no
    `criteria` makes every label look wrong, so these are named as question-set faults rather than reported as
    a file full of bad labels."""
    if not isinstance(q, dict):
        return f"{_label(q) if isinstance(q, (str, int, float, bool, type(None))) else type(q).__name__} is not a question object"
    if q.get("type") not in LABEL_TYPES:
        return f"type {q.get('type')!r} is not one of " + ", ".join(LABEL_TYPES)
    if q["type"] in ("choice", "score") and not q.get("criteria"):
        return f"a {q['type']} question needs 'criteria'"
    return None


def valid_label(q, value):
    """Whether a label-file value is one the question can answer with (eval-protocol.md section 1): an option
    name for Choice, a JSON boolean for Noul, a rubric position for Score. Anything else (null, a list, the
    string "false", a misspelt option, a level off the rubric) is a label-file error, never a model miss.
    A question question_defect() rejects can take no label at all; check_labels names that fault separately."""
    typ = q.get("type"); crit = q.get("criteria") or {}
    if typ == "choice":
        return isinstance(value, str) and value in crit
    if typ == "noul":
        return isinstance(value, bool)
    if typ == "score":
        return _number(value, 0, len(crit) - 1)
    return False


def check_labels(items, qset, path=None):
    """Refuse a label file before any request is sent: every 'expected' value for a question in the set must
    satisfy valid_label. Keys for questions outside the set are ignored and an item may leave a question
    unlabelled. Raises JevError naming the offending items, so a misspelt option or a string "false" costs no
    API calls instead of a report annotation, a silently wrong rate, or a traceback after the whole run. A
    question that can take no label at all is raised against the question set instead (question_defect)."""
    qs = questions_of(qset); bad = []
    # a question that can take no label at all would reject every item and read as a ruined label file; name the
    # question instead, and only for questions the file actually labels
    labelled = {qid for it in items if isinstance(it.get("expected"), dict) for qid in it["expected"]}
    broken = [(qid, why) for qid in sorted(labelled & set(qs)) if (why := question_defect(qs[qid]))]
    if broken:
        raise JevError("the question set cannot check these labels; fix the questions, not the label file: "
                       + "; ".join(f"{qid}: {why}" for qid, why in broken))
    for it in items:
        exp = it.get("expected", {})
        if not isinstance(exp, dict):
            bad.append(f"{it.get('id')}: expected={_label(exp)} is not an object"); continue
        bad.extend(f"{it.get('id')}: {qid}={_label(exp[qid])}" for qid, q in qs.items()
                   if qid in exp and not valid_label(q, exp[qid]))
    if bad:
        raise JevError((f"{path}: " if path else "") + f"{len(bad)} label(s) the question set cannot take; fix the "
                       "label file (no request was sent): " + "; ".join(bad[:10]) + (" ..." if len(bad) > 10 else ""))


def lowest_recall(per_class):
    """The Choice option with the lowest recall among those with labelled support, or None.
    ablate.py and sweep_batch.py headline it because overall accuracy can hold while one class collapses.
    A recall tie goes to the option with the least support, the class most likely to collapse next."""
    supported = {lab: c for lab, c in (per_class or {}).items() if c["n"]}
    if not supported:
        return None
    lab = min(supported, key=lambda k: (supported[k]["recall"], supported[k]["n"], k))   # then the name, so the pick is stable
    c = supported[lab]
    return {"label": lab, "hits": c["hits"], "n": c["n"], "recall": c["recall"]}


def fmt_low(low):
    """The ' low <option> hits/n' fragment ablate.py and sweep_batch.py append to a Choice cell; '' without
    support. The option name is printed whole; worst_case() measures the column from the longest one."""
    return f" low {low['label']} {low['hits']}/{low['n']}" if low else ""


def score(results, qset, policy=None):
    """Score answered items against their 'expected' values. policy = {qid: {"threshold": x, "parents": {...}}}."""
    qs = questions_of(qset); policy = policy or {}
    ok = [r for r in results if r.get("answers")]
    models = sorted({r["model"] for r in ok if r.get("model")})
    rep = {"items": len(results), "failed": len(results) - len(ok),
           "requests": round(sum(r["requests"] for r in results), 2),
           "tokens_per_item": _mean(r["tokens"] for r in ok), "ms_per_item": _mean(r["ms"] for r in ok),
           "questions_per_request": ok[0].get("questions_per_request") if ok else None,
           "model": models[0] if len(models) == 1 else ("mixed" if models else None),
           "models": models, "questions": {}}
    rep["usd_per_1000_items"] = (
        _mean(r["tokens"] * PRICES_PER_MTOK[r["model"]] for r in ok) * 1000 / 1e6
        if ok and all(r.get("model") in PRICES_PER_MTOK for r in ok) else None)
    for qid, q in qs.items():
        rows = [r for r in ok if qid in r["expected"] and qid in r["answers"]]
        if not rows:
            continue
        pol = policy.get(qid, {}); typ = q["type"]
        # a label the question cannot take (check_labels refuses one before the spend; score() also sees rows
        # from elsewhere) is never a hit: it is reported under 'unlisted', listed as a miss, and joins no class,
        # so every rate below is computed over labels the question can take
        good = lambda r: valid_label(q, r["expected"][qid])
        unlisted = dict(sorted(Counter(_label(r["expected"][qid]) for r in rows if not good(r)).items()))
        if typ == "choice":
            thr = pol.get("threshold", 0.75); parents = pol.get("parents", {})
            options = list(q["criteria"])
            pairs = [(r, r["answers"][qid]) for r in rows]
            hit = lambda r, a: a["choice"] == r["expected"][qid]   # an invalid label never equals an option name
            fine = sum(hit(r, a) for r, a in pairs)
            roll = lambda lab: parents.get(lab, lab)
            # a label the question cannot take is a coarse miss, like a fine one, and `good` short-circuits before
            # roll() sees it: without the guard a label that happens to name a parent rolls onto every child of
            # that parent and scores as correct, and an unhashable one raises TypeError
            coarse = sum(good(r) and roll(a["choice"]) == roll(r["expected"][qid]) for r, a in pairs) if parents else None
            passed = [(r, a) for r, a in pairs if a["confidence"] >= thr]
            buckets, prev = [], 1.01
            for lo in BUCKETS:
                b = [(r, a) for r, a in pairs if lo <= a["confidence"] < prev]
                above = [(r, a) for r, a in pairs if a["confidence"] >= lo]
                buckets.append({"lo": lo, "hi": min(prev, 1.0), "n": len(b),
                                "acc": (sum(hit(r, a) for r, a in b) / len(b)) if b else None,
                                "acc_above": (sum(hit(r, a) for r, a in above) / len(above)) if above else None,
                                "n_above": len(above), "coverage_above": len(above) / len(pairs)})
                prev = lo
            by_pos = None
            if any("pos" in r and r.get("batch_len", 1) >= 4 for r, _ in pairs):
                by_pos = []
                for quarter in range(4):   # not `q`: that is the question, read again below
                    qp = [(r, a) for r, a in pairs if "pos" in r and r.get("batch_len", 1) >= 4
                          and quarter * r["batch_len"] / 4 <= r["pos"] < (quarter + 1) * r["batch_len"] / 4]
                    by_pos.append({"quarter": quarter + 1, "n": len(qp), "acc": (sum(hit(r, a) for r, a in qp) / len(qp)) if qp else None,
                                   "mean_conf": _mean(a["confidence"] for _, a in qp)})
            n_exp, n_pred, hits = Counter(), Counter(), Counter()   # one pass over the pairs
            for r, a in pairs:
                n_pred[a["choice"]] += 1
                if good(r):
                    e = r["expected"][qid]; n_exp[e] += 1
                    if hit(r, a):
                        hits[e] += 1
            # one row per option, in production order; the client rejects an answer outside the option
            # list before scoring, so only the label file can put a value outside it
            per_class = {lab: {"n": n_exp[lab], "hits": hits[lab], "recall": (hits[lab] / n_exp[lab]) if n_exp[lab] else None,
                               "predicted": n_pred[lab], "precision": (hits[lab] / n_pred[lab]) if n_pred[lab] else None}
                         for lab in options}
            rep["questions"][qid] = {
                "type": "choice", "n": len(pairs), "fine": fine, "coarse": coarse, "threshold": thr, "by_position": by_pos,
                "per_class": per_class, "lowest_recall": lowest_recall(per_class), "unlisted": unlisted,
                "pass": len(passed), "pass_correct": sum(hit(r, a) for r, a in passed),
                "mean_conf": _mean(a["confidence"] for _, a in pairs), "buckets": buckets,
                "misses": [{"id": r["id"], "subject": subject_of(r), "expected": r["expected"][qid], "got": a["choice"], "conf": round(a["confidence"], 2)}
                           for r, a in pairs if not hit(r, a)]}
        elif typ == "noul":
            thr = pol.get("threshold", 0.8)
            pairs = [(r, r["answers"][qid]["noul"]) for r in rows]
            pos = [(r, p) for r, p in pairs if r["expected"][qid] is True]; neg = [(r, p) for r, p in pairs if r["expected"][qid] is False]
            caught = sum(p >= thr for _, p in pos); fp = sum(p >= thr for _, p in neg)
            rep["questions"][qid] = {
                "type": "noul", "n": len(pairs), "threshold": thr,
                "positives": len(pos), "caught": caught,
                "negatives": len(neg), "false_positives": fp,
                "recall": (caught / len(pos)) if pos else None,
                "precision": (caught / (caught + fp)) if (caught + fp) else None,
                "tnr": ((len(neg) - fp) / len(neg)) if neg else None,
                "lowest_true": min((p for _, p in pos), default=None), "highest_false": max((p for _, p in neg), default=None),
                "unlisted": unlisted,
                "misses": [{"id": r["id"], "subject": subject_of(r), "expected": r["expected"][qid], "p": round(p, 2)}
                           for r, p in pairs if not good(r) or (p >= thr) != r["expected"][qid]]}
        elif typ == "score":
            pairs = [(r, s, abs(s - r["expected"][qid]) if good(r) else None) for r, s in ((r, r["answers"][qid]["score"]) for r in rows)]
            err = [e for _, _, e in pairs if e is not None]
            rep["questions"][qid] = {"type": "score", "n": len(pairs), "mae": _mean(err),
                                     "within_half_level": sum(e <= 0.5 for e in err), "unlisted": unlisted,
                                     "misses": [{"id": r["id"], "subject": subject_of(r), "expected": r["expected"][qid], "got": round(s, 2)}
                                                for r, s, e in pairs if e is None or e > 0.5]}
    return rep


def agreement(a, b, qset):
    """Determinism between two runs over the same items: label agreement and mean |Δ|."""
    qs = questions_of(qset); out = {}
    for qid, q in qs.items():
        pairs = [(x["answers"][qid], y["answers"][qid]) for x, y in zip(a, b)
                 if x.get("answers") and y.get("answers") and qid in x["answers"] and qid in y["answers"]]
        if not pairs:
            continue
        if q["type"] == "choice":
            out[qid] = {"same_label": sum(x["choice"] == y["choice"] for x, y in pairs), "n": len(pairs),
                        "mean_abs_dconf": _mean(abs(x["confidence"] - y["confidence"]) for x, y in pairs)}
        elif q["type"] == "noul":
            out[qid] = {"n": len(pairs), "mean_abs_dp": _mean(abs(x["noul"] - y["noul"]) for x, y in pairs)}
        else:
            out[qid] = {"n": len(pairs), "mean_abs_dscore": _mean(abs(x["score"] - y["score"]) for x, y in pairs)}
    return out


# ---------- printing ----------

def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def cell(r):
    """One scored question as a cell for the ablate.py and sweep_batch.py tables. Measure the column with
    cell(worst_case(q, n)) before the first row prints; counts are right-aligned to three digits so rows line
    up while n holds, and the scripts ljust the cell to the measured width when it does not."""
    if r["type"] == "choice":
        return (f"fine {r['fine']:>3}/{r['n']} " + (f"coarse {r['coarse']:>3} " if r["coarse"] is not None else "")
                + f"≥thr {r['pass']:>3}({r['pass_correct']:>3}) conf {fmt(r['mean_conf'])}" + fmt_low(r.get("lowest_recall")))
    if r["type"] == "noul":
        return (f"caught {r['caught']}/{r['positives']} fp {r['false_positives']}/{r['negatives']} "
                f"margin {fmt(r['lowest_true'])}/{fmt(r['highest_false'])}")
    return f"MAE {fmt(r['mae'])} within½ {r['within_half_level']}/{r['n']}"


def worst_case(q, n, coarse=False):
    """A report row for question q with every count at its maximum over n items, its longest option name and
    the largest MAE its rubric allows, so a script can measure a column before the first row prints rather
    than guess a width. coarse: whether the policy rolls this question up, which adds a count to a Choice cell.
    A question no report row can come from raises JevError, so the script exits with a message naming it."""
    why = question_defect(q)
    if why:
        raise JevError(f"cannot size a report column for this question: {why}")
    crit = q.get("criteria") or {}
    if q["type"] == "choice":
        longest = max((str(k) for k in crit), key=len, default="")
        return {"type": "choice", "fine": n, "n": n, "coarse": n if coarse else None, "pass": n, "pass_correct": n,
                "mean_conf": 1.0, "lowest_recall": {"label": longest, "hits": n, "n": n, "recall": 1.0}}
    if q["type"] == "noul":
        return {"type": "noul", "caught": n, "positives": n, "false_positives": n, "negatives": n,
                "lowest_true": 1.0, "highest_false": 1.0}
    return {"type": "score", "n": n, "mae": float(max(len(crit) - 1, 0)), "within_half_level": n}


def _print_unlisted(q):
    if q.get("unlisted"):
        print("      unlisted labels     " + "  ".join(f"{lab} {n}" for lab, n in q["unlisted"].items())
              + "   (label-file values the question cannot take; scored as misses, counted in no class)")


def print_report(title, rep, verbose=False):
    print(f"\n=== {title} ===  items {rep['items']}  failed {rep['failed']}  requests {rep['requests']}  "
          f"q/req {rep['questions_per_request']}  tokens/item {fmt(rep['tokens_per_item'], 0)}  ms/item {fmt(rep['ms_per_item'], 0)}  "
          f"estimated USD/1000 items {fmt(rep['usd_per_1000_items'], 3)}  model {rep['model']}")
    if rep["model"] == "mixed":
        print(f"  WARNING: multiple resolved models: {', '.join(rep['models'])}; evaluate versions separately")
    for qid, q in rep["questions"].items():
        if q["type"] == "choice":
            line = (f"  {qid:<28} choice  fine {q['fine']:>3}/{q['n']}"
                    + (f"  coarse {q['coarse']:>3}/{q['n']}" if q["coarse"] is not None else "")
                    + f"  ≥{q['threshold']}: {q['pass']} ({q['pass_correct']} right)  mean conf {fmt(q['mean_conf'])}")
            print(line)
            pc = q.get("per_class") or {}
            if pc:
                print("      per-class recall    " + "  ".join(f"{lab} {c['hits']}/{c['n']}" if c["n"] else f"{lab} -" for lab, c in pc.items()))
                print("      per-class precision " + "  ".join(f"{lab} {c['hits']}/{c['predicted']}" if c["predicted"] else f"{lab} -" for lab, c in pc.items()))
            _print_unlisted(q)
            if verbose:
                if q.get("by_position"):
                    print("      position quarter   n   acc   conf   (investigate position effects; these do not identify the cause)")
                    for b in q["by_position"]:
                        print(f"      {b['quarter']:>7}          {b['n']:>3}  {fmt(b['acc'])}  {fmt(b['mean_conf'])}")
                print("      bucket        n   acc   n≥  acc≥  cov≥")
                for b in q["buckets"]:
                    closing = "]" if b["hi"] == 1 else ")"
                    print(f"      [{b['lo']:.2f},{b['hi']:.2f}{closing} {b['n']:>3}  {fmt(b['acc'])}  {b['n_above']:>3}  {fmt(b['acc_above'])}  {fmt(b['coverage_above'])}")
                for m in q["misses"]:
                    print(f"      miss {str(m['id'])[:14]:<14} {m.get('subject',''):<52} expected {str(m['expected']):<22} got {str(m['got']):<22} {m['conf']}")
        elif q["type"] == "noul":
            print(f"  {qid:<28} noul    caught {q['caught']}/{q['positives']}  false positives {q['false_positives']}/{q['negatives']}"
                  f"  recall {fmt(q['recall'])}  precision {fmt(q['precision'])}  TNR {fmt(q['tnr'])}"
                  f"  lowest true {fmt(q['lowest_true'])}  highest false {fmt(q['highest_false'])}  (threshold {q['threshold']})")
            _print_unlisted(q)
            if verbose:
                for m in q["misses"]:
                    print(f"      miss {str(m['id'])[:14]:<14} {m.get('subject',''):<52} expected {m['expected']}  p {m['p']}")
        else:
            print(f"  {qid:<28} score   MAE {fmt(q['mae'])}  within ½ level {q['within_half_level']}/{q['n']}")
            _print_unlisted(q)
            if verbose:
                for m in q["misses"]:
                    print(f"      miss {str(m['id'])[:14]:<14} {m.get('subject',''):<52} expected {m['expected']}  got {m['got']}")


def redact(text, key=None):
    """Remove an active credential and bearer tokens; not a general secret detector."""
    secrets = {key, os.environ.get("TYPESAFE_API_KEY", "").strip()} - {None, ""}
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    return re.sub(r"(?i)(Bearer\s+)[^\s\"',<>}\]]+", r"\1<redacted>", text)
