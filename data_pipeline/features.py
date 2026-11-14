"""
Shared feature + label logic for the Mumtaz attendance-validity model.

Used by BOTH:
  - the training script (build_training_set.py), and
  - the FastAPI service at inference time,
so features are computed identically in training and live (doc section 4.5).
Pure Python, no dependencies.

Input shape (doc section 4.1):
  target      = {id, tutor_id, student_id, subject_ids, lesson_date, start_time,
                 duration_minutes, submitted_at}
  comparisons = list of sessions in the same shape (already filtered by the
                backend: same tutor OR student, +-30 days, submitted earlier, max 100)
  config      = see DEFAULT_CONFIG

Business rules (doc section 1.1 + AE corrections 2026-11-14 + overlap rules):
  RED  rejected        : exact duplicate - tutor, student, subjects, date,
                         start_time AND duration identical to an earlier session
  BLUE needs_revision  : duration > 180 or < 30 minutes
                         start > 21:00 or < 07:00
                         "duplikat samar": same tutor/student/subjects/date/duration,
                           different start_time
                         tutor has an overlapping session with a DIFFERENT student
                         student has an overlapping session with a DIFFERENT tutor
                         same tutor AND same student have overlapping sessions that are
                           not an exact duplicate (e.g. same start/different duration,
                           same time/different subjects, start shifted a few minutes)
  GREEN valid          : none of the above
  Priority: rejected > needs_revision > valid. All findings are still reported.
  Boundaries are allowed: exactly 180 / 30 minutes, exactly 21:00 / 07:00 = valid.
"""

DEFAULT_CONFIG = {
    "durationOutlierMinutes": 180,   # BLUE when duration > this
    "durationMinMinutes": 30,        # BLUE when duration < this   (AE rule, new)
    "lateStartAfter": "21:00",       # BLUE when start > this
    "earlyStartBefore": "07:00",     # BLUE when start < this      (AE rule, new)
}

# Column order of the model input. Changing it means retraining the model.
FEATURE_NAMES = [
    # A. from target + config (doc 3.3, 1-5)
    "duration_minutes",
    "duration_over_limit",
    "start_minute",
    "start_over_limit",
    "subject_count",
    # B. target vs comparisons (doc 3.3, 6-13)
    "comparisons_count",
    "exact_dup_count",
    "same_tutor_student_date_count",
    "same_student_date_diff_tutor_count",
    "time_overlap_count",
    "tutor_sessions_30d",
    "student_sessions_30d",
    "duration_diff_from_tutor_mean_30d",
    # C. new for the AE's 2026-11-14 rules (14-17)
    "duration_under_limit",
    "start_under_limit",
    "fuzzy_dup_count",
    "tutor_overlap_diff_student_count",
    # D. overlap rules (18-19)
    "student_overlap_diff_tutor_count",
    "self_overlap_count",
]


def hhmm_to_minutes(value):
    if value is None:
        return None
    h, m = str(value).split(":")
    return int(h) * 60 + int(m)


def _subjects(session):
    return tuple(sorted(session.get("subject_ids") or []))


def _overlaps(a_start, a_dur, b_start, b_dur):
    if a_start is None or b_start is None:
        return False
    return a_start < b_start + b_dur and b_start < a_start + a_dur


def compute_features(target, comparisons, config=None):
    """Return (features: dict in FEATURE_NAMES order, related: dict of finding -> comparison id)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    dur = int(target["duration_minutes"])
    start = hhmm_to_minutes(target.get("start_time"))
    subj = _subjects(target)
    t_id, s_id, date = target["tutor_id"], target["student_id"], target["lesson_date"]

    exact = fuzzy = same_tsd = same_sd_diff_t = overlap = tutor_overlap_diff_s = student_overlap_diff_t = self_overlap = 0
    tutor_n = student_n = 0
    tutor_durations = []
    related = {}

    for c in comparisons:
        c_start = hhmm_to_minutes(c.get("start_time"))
        c_dur = int(c["duration_minutes"])
        c_subj = _subjects(c)
        same_t, same_s, same_d = c["tutor_id"] == t_id, c["student_id"] == s_id, c["lesson_date"] == date

        if same_t:
            tutor_n += 1
            tutor_durations.append(c_dur)
        if same_s:
            student_n += 1
        if not same_d:
            continue

        if same_t and same_s:
            same_tsd += 1
            if c_subj == subj and c_dur == dur:
                if c_start == start:
                    exact += 1
                    related.setdefault("suspected_duplicate", c["id"])
                else:
                    fuzzy += 1
                    related.setdefault("fuzzy_duplicate", c["id"])
        if same_s and not same_t and set(c_subj) & set(subj):
            same_sd_diff_t += 1
        if (same_t or same_s) and _overlaps(start, dur, c_start, c_dur):
            overlap += 1
            if same_t and not same_s:
                tutor_overlap_diff_s += 1
                related.setdefault("tutor_time_overlap", c["id"])
            if same_s and not same_t:
                student_overlap_diff_t += 1
                related.setdefault("student_time_overlap", c["id"])
            is_exact = c_subj == subj and c_dur == dur and c_start == start
            if same_t and same_s and not is_exact:
                self_overlap += 1
                related.setdefault("self_time_overlap", c["id"])

    tutor_mean = sum(tutor_durations) / len(tutor_durations) if tutor_durations else dur
    start_or_zero = start if start is not None else 0
    features = {
        "duration_minutes": dur,
        "duration_over_limit": dur - cfg["durationOutlierMinutes"],
        "start_minute": start_or_zero,
        "start_over_limit": start_or_zero - hhmm_to_minutes(cfg["lateStartAfter"]),
        "subject_count": len(subj),
        "comparisons_count": len(comparisons),
        "exact_dup_count": exact,
        "same_tutor_student_date_count": same_tsd,
        "same_student_date_diff_tutor_count": same_sd_diff_t,
        "time_overlap_count": overlap,
        "tutor_sessions_30d": tutor_n,
        "student_sessions_30d": student_n,
        "duration_diff_from_tutor_mean_30d": round(dur - tutor_mean, 2),
        # Positive = rule broken, 0 or negative = safe (same idea as doc section 3.4)
        "duration_under_limit": cfg["durationMinMinutes"] - dur,
        "start_under_limit": (hhmm_to_minutes(cfg["earlyStartBefore"]) - start) if start is not None else -1,
        "fuzzy_dup_count": fuzzy,
        "tutor_overlap_diff_student_count": tutor_overlap_diff_s,
        "student_overlap_diff_tutor_count": student_overlap_diff_t,
        "self_overlap_count": self_overlap,
    }
    return features, related


def label_from_features(f):
    """Rule-based label (the 'guardrail'). Returns (validity, list of finding types)."""
    findings = []
    if f["exact_dup_count"] >= 1:
        findings.append("suspected_duplicate")
    if f["duration_over_limit"] > 0:
        findings.append("duration_outlier")
    if f["duration_under_limit"] > 0:
        findings.append("duration_too_short")
    if f["start_over_limit"] > 0:
        findings.append("late_start")
    if f["start_under_limit"] > 0:
        findings.append("early_start")
    if f["fuzzy_dup_count"] >= 1:
        findings.append("fuzzy_duplicate")
    if f["tutor_overlap_diff_student_count"] >= 1:
        findings.append("tutor_time_overlap")
    if f["student_overlap_diff_tutor_count"] >= 1:
        findings.append("student_time_overlap")
    if f["self_overlap_count"] >= 1:
        findings.append("self_time_overlap")

    if "suspected_duplicate" in findings:
        validity = "rejected"
    elif findings:
        validity = "needs_revision"
    else:
        validity = "valid"
    return validity, findings
