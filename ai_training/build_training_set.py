"""
Usage:
    python ai_training/build_training_set.py data/ai_training/sessions_with_synthetic.csv data/ai_training/training

Input: a sessions CSV with the 8 fields of doc section 3.2 (+ optional is_synthetic, scenario).
For every session (the `target`) this script:
  1. builds its `comparisons` the way the backend will live (doc section 3.1):
       same tutor_id OR same student_id, |lesson_date diff| <= 30 days,
       submitted_at strictly earlier than the target, at most 100 sessions
  2. computes the 19 features with features.py (shared with FastAPI)
  3. labels it with the business rules (valid / needs_revision / rejected)

Outputs (in the output folder):
    training_dataset.csv   one row per session: features + label (ready for model.fit)
    sample_requests.json   example {target, comparisons, config} payloads + expected
                           label, for testing the FastAPI endpoint
"""
import csv
import datetime as dt
import json
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import DEFAULT_CONFIG, FEATURE_NAMES, compute_features, label_from_features  # noqa: E402

WINDOW_DAYS = 30
MAX_COMPARISONS = 100
SESSION_FIELDS = ["id", "tutor_id", "student_id", "subject_ids", "lesson_date", "start_time",
                  "duration_minutes", "submitted_at"]


def load_sessions(path):
    return prepare_sessions(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))


def prepare_sessions(records):
    sessions = []
    for r in records:
        s = {k: r[k] for k in SESSION_FIELDS}
        s["subject_ids"] = json.loads(s["subject_ids"]) if isinstance(s["subject_ids"], str) else list(s["subject_ids"])
        s["duration_minutes"] = int(s["duration_minutes"])
        s["_date"] = dt.date.fromisoformat(s["lesson_date"])
        s["_synthetic"] = str(r.get("is_synthetic", "0")) in ("1", "true", "True")
        s["_scenario"] = r.get("scenario") or "real"
        sessions.append(s)
    return sessions


def public(s):
    return {k: s[k] for k in SESSION_FIELDS}


def build_comparisons(target, by_person):
    """Sessions of the same tutor OR student, +-30 days, submitted earlier, max 100."""
    lo = target["_date"] - dt.timedelta(days=WINDOW_DAYS)
    hi = target["_date"] + dt.timedelta(days=WINDOW_DAYS)
    seen, pool = set(), []
    for key in (("t", target["tutor_id"]), ("s", target["student_id"])):
        dates, rows = by_person[key]
        for s in rows[bisect_left(dates, lo):bisect_right(dates, hi)]:
            if s["id"] != target["id"] and s["id"] not in seen and s["submitted_at"] < target["submitted_at"]:
                seen.add(s["id"])
                pool.append(s)
    # If more than 100, keep the sessions closest in lesson_date, newest submissions first.
    # NOTE for the backend: the live query should pick the same 100 (rarely matters; see report).
    pool.sort(key=lambda s: (abs((s["_date"] - target["_date"]).days), _neg(s["submitted_at"])))
    return pool[:MAX_COMPARISONS], len(pool)


def _neg(ts):
    return [-ord(ch) for ch in ts]


def label_sessions(sessions):
    """Build comparisons, features and labels for every session. Returns (rows, samples, capped)."""
    by_person = defaultdict(lambda: ([], []))
    for s in sorted(sessions, key=lambda s: s["_date"]):
        for key in (("t", s["tutor_id"]), ("s", s["student_id"])):
            by_person[key][0].append(s["_date"])
            by_person[key][1].append(s)

    rows, samples, capped = [], {}, 0
    for s in sessions:
        comps, pool_size = build_comparisons(s, by_person)
        capped += pool_size > MAX_COMPARISONS
        feats, related = compute_features(public(s), [public(c) for c in comps], DEFAULT_CONFIG)
        validity, findings = label_from_features(feats)
        rel = next((related[f] for f in ("suspected_duplicate", "fuzzy_duplicate", "tutor_time_overlap", "student_time_overlap", "self_time_overlap")
                    if f in related), "")
        rows.append({"session_id": s["id"], **feats, "related_session_id": rel,
                     "findings": ";".join(findings), "validity": validity,
                     "scenario": s["_scenario"], "is_synthetic": int(s["_synthetic"])})
        # keep one small example payload per (validity, first finding), preferring few comparisons
        sample_key = f"{validity}:{findings[0] if findings else 'none'}"
        if sample_key not in samples or len(comps) < len(samples[sample_key]["request"]["comparisons"]):
            samples[sample_key] = {"expected_validity": validity, "expected_findings": findings,
                                   "request": {"target": public(s), "comparisons": [public(c) for c in comps],
                                               "config": DEFAULT_CONFIG}}

    return rows, samples, capped


def main(src, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, samples, capped = label_sessions(load_sessions(src))
    cols = ["session_id", *FEATURE_NAMES, "related_session_id", "findings", "validity", "scenario", "is_synthetic"]
    with open(out / "training_dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out / "sample_requests.json").write_text(json.dumps(dict(sorted(samples.items())), indent=2, ensure_ascii=False))

    n = len(rows)
    print(f"sessions: {n}  (real {n - sum(r['is_synthetic'] for r in rows)}, synthetic {sum(r['is_synthetic'] for r in rows)})")
    print(f"targets with more than {MAX_COMPARISONS} candidate comparisons: {capped}")
    print("comparisons per target: max", max(r["comparisons_count"] for r in rows),
          " mean", round(sum(r["comparisons_count"] for r in rows) / n, 1))
    for label, c in Counter(r["validity"] for r in rows).most_common():
        print(f"  {label:15s} {c:5d}  ({c / n:.1%})")
    print("findings:", dict(Counter(f for r in rows for f in r["findings"].split(";") if f).most_common()))
    print("label by source:", dict(Counter((r["is_synthetic"], r["validity"]) for r in rows)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
