"""
Usage:
    python ai_training/inject_anomalies.py data/ai_training/data_ai_clean.csv data/ai_training/sessions_with_synthetic.csv

Real data is ~94% valid and some rules have almost no examples, so the model
cannot learn them. This script adds SYNTHETIC rows for every rule (doc section 5.5,
extended with the AE's 2026-11-14 rules) until the final dataset matches
TARGET_SHARE (default 60% valid / 20% needs_revision / 20% rejected).
The counts per scenario are calculated from the real data's labels, so the shares
stay right when the real data changes. Every synthetic row is built from a real,
valid session, marked is_synthetic=1, and given a `scenario` name.

Safety: synthetic rows are placed so they do NOT change the label of any real row
(free days/slots, submit times after the session they copy). build_training_set.py
re-checks this.

Same seed -> same output every run. Change TARGET_SHARE / FIXED_COUNTS to rebalance.
"""
import csv
import datetime as dt
import json
import random
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import DEFAULT_CONFIG, hhmm_to_minutes  # noqa: E402

SEED = 20261114
FIELDS = ["id", "tutor_id", "student_id", "subject_ids", "lesson_date", "start_time",
          "duration_minutes", "submitted_at"]
DAY_START, DAY_END = 7 * 60, 21 * 60   # "normal" teaching hours used for safe slots
LAST_MINUTE = 23 * 60 + 59            # form rejects lessons ending after 23:59 (doc 5.4)

# Final share of each label in the training data (real + synthetic).
TARGET_SHARE = {"valid": 0.60, "needs_revision": 0.20, "rejected": 0.20}

# Scenarios with a fixed count. A "kombinasi" pair adds 2 rows per count:
# an unusual session (needs_revision) + its exact copy (rejected, both findings).
FIXED_COUNTS = {
    "batas_valid": 100,           # exactly 180/30 min, exactly 21:00/07:00 -> valid (teaches the edges)
    "kombinasi_dup_panjang": 50,  # long session + copy
    "kombinasi_dup_pendek": 50,   # short session + copy
    "kombinasi_dup_malam": 50,    # late session + copy
    "kombinasi_dup_pagi": 50,     # early session + copy
}
# The rest is calculated in plan_counts():
#   needs_revision, split evenly over these 7 scenarios
NEEDS_REVISION_SCENARIOS = [
    "durasi_panjang",        # 181-300 min
    "durasi_pendek",         # 10-29 min
    "mulai_malam",           # start 21:01-22:30
    "mulai_pagi",            # start 05:00-06:59
    "duplikat_samar",        # copy of a real session, different start_time
    "tumpang_tindih",        # same tutor, other student, overlapping time
    "tumpang_tindih_murid",  # same student, other tutor, overlapping time
    "tumpang_tindih_sendiri",  # same tutor + student, overlapping, not exact copy
]
#   rejected: "duplikat" = exact copy of a real session, submitted later


def plan_counts(real_labels):
    """How many synthetic rows per scenario so the final data matches TARGET_SHARE."""
    pairs = sum(v for k, v in FIXED_COUNTS.items() if k.startswith("kombinasi"))
    valid_total = real_labels["valid"] + FIXED_COUNTS["batas_valid"]
    total = round(valid_total / TARGET_SHARE["valid"])
    want_nr = round(total * TARGET_SHARE["needs_revision"])
    want_rej = total - valid_total - want_nr
    nr_singles = want_nr - real_labels["needs_revision"] - pairs
    dup = want_rej - real_labels["rejected"] - pairs
    if nr_singles < 0 or dup < 0:
        sys.exit("TARGET_SHARE cannot be reached by adding rows; lower FIXED_COUNTS or change the shares")
    counts = dict(FIXED_COUNTS, duplikat=dup)
    per, extra = divmod(nr_singles, len(NEEDS_REVISION_SCENARIOS))
    for i, name in enumerate(NEEDS_REVISION_SCENARIOS):
        counts[name] = per + (1 if i < extra else 0)
    return counts


def hm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def ts(value):
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def ts_str(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


class Injector:
    def __init__(self, real_rows, rnd, counts):
        self.rnd = rnd
        self.counts = counts
        self.self_overlap_variant = 0
        self.rows = real_rows
        self.synthetic = []
        self.day = defaultdict(list)          # (kind, person, date) -> [(start, end)]
        self.students_of_tutor = defaultdict(set)
        self.tutors_of_student = defaultdict(set)
        self.all_tutors = sorted({r["tutor_id"] for r in real_rows})
        self.all_subjects = sorted({sid for r in real_rows for sid in json.loads(r["subject_ids"])})
        self.max_submit = defaultdict(lambda: dt.datetime.min)   # (kind, person) -> latest submitted_at
        for r in real_rows:
            self._occupy(r)
            self.students_of_tutor[r["tutor_id"]].add(r["student_id"])
            self.tutors_of_student[r["student_id"]].add(r["tutor_id"])
        # base pool: real sessions that are clean & unambiguous
        self.base_pool = self._clean_bases()

    # ---------------------------------------------------------------- bookkeeping
    def _slot(self, r):
        s = hhmm_to_minutes(r["start_time"])
        return s, s + int(r["duration_minutes"])

    def _occupy(self, r):
        for key in (("t", r["tutor_id"]), ("s", r["student_id"])):
            self.day[(*key, r["lesson_date"])].append(self._slot(r))
            self.max_submit[key] = max(self.max_submit[key], ts(r["submitted_at"]))

    def _free(self, kind, person, date, start, end, ignore=None):
        for a, b in self.day[(kind, person, date)]:
            if (a, b) == ignore:
                ignore = None          # skip exactly one matching slot (the base session)
                continue
            if start < b and a < end:
                return False
        return True

    def _clean_bases(self):
        count = Counter((r["tutor_id"], r["student_id"], r["lesson_date"]) for r in self.rows)
        pool = []
        for r in self.rows:
            s, e = self._slot(r)
            if (count[(r["tutor_id"], r["student_id"], r["lesson_date"])] == 1
                    and DAY_START <= s and e <= DAY_END and 30 <= int(r["duration_minutes"]) <= 180
                    and self._free("t", r["tutor_id"], r["lesson_date"], s, e, ignore=(s, e))
                    and self._free("s", r["student_id"], r["lesson_date"], s, e, ignore=(s, e))):
                pool.append(r)
        return pool

    def _new_id(self):
        return str(uuid.UUID(int=self.rnd.getrandbits(128), version=4))

    def _add(self, row, scenario):
        row = {**row, "id": self._new_id(), "is_synthetic": 1, "scenario": scenario}
        self.synthetic.append(row)
        self._occupy(row)
        return row

    def _submit_after_lesson(self, row, latest_than=None):
        """Lesson end (WIB) + 5..180 min, and later than `latest_than` if given."""
        end_local = dt.datetime.fromisoformat(row["lesson_date"]) + dt.timedelta(minutes=self._slot(row)[1])
        t = end_local - dt.timedelta(hours=7) + dt.timedelta(minutes=self.rnd.randint(5, 180),
                                                               seconds=self.rnd.randint(0, 59))
        if latest_than and t <= latest_than:
            t = latest_than + dt.timedelta(minutes=self.rnd.randint(1, 600), seconds=self.rnd.randint(0, 59))
        return ts_str(t)

    def _free_day_for(self, base, start, end, max_tries=60):
        """A date near the base where neither tutor nor student has any session."""
        d0 = dt.date.fromisoformat(base["lesson_date"])
        for _ in range(max_tries):
            d = (d0 + dt.timedelta(days=self.rnd.choice([-1, 1]) * self.rnd.randint(1, 20))).isoformat()
            if not self.day[("t", base["tutor_id"], d)] and not self.day[("s", base["student_id"], d)]:
                return d
        return None

    # ---------------------------------------------------------------- scenarios
    def standalone(self, scenario, start, duration):
        """One unusual session on a day the tutor and student are otherwise free.
        Submitted after every real session of that tutor/student, so no real row
        ever has it in its comparisons."""
        for _ in range(200):
            base = self.rnd.choice(self.base_pool)
            date = self._free_day_for(base, start, start + duration)
            if not date:
                continue
            row = {**base, "lesson_date": date, "start_time": hm(start), "duration_minutes": duration}
            latest = max(self.max_submit[("t", base["tutor_id"])], self.max_submit[("s", base["student_id"])])
            row["submitted_at"] = self._submit_after_lesson(row, latest_than=latest)
            return self._add(row, scenario)
        raise RuntimeError(f"no free day found for {scenario}")

    def exact_copy(self, base, scenario):
        row = {k: base[k] for k in FIELDS}
        row["submitted_at"] = ts_str(ts(base["submitted_at"]) + dt.timedelta(
            minutes=self.rnd.randint(1, 4320), seconds=self.rnd.randint(0, 59)))
        return self._add(row, scenario)

    def fuzzy_copy(self, base):
        s, e = self._slot(base)
        dur = e - s
        slots = [m for m in range(DAY_START, DAY_END - dur + 1, 15)
                 if m != s and self._free("t", base["tutor_id"], base["lesson_date"], m, m + dur)
                 and self._free("s", base["student_id"], base["lesson_date"], m, m + dur)]
        if not slots:
            return None
        row = {k: base[k] for k in FIELDS}
        row["start_time"] = hm(self.rnd.choice(slots))
        row["submitted_at"] = ts_str(ts(base["submitted_at"]) + dt.timedelta(
            minutes=self.rnd.randint(1, 4320), seconds=self.rnd.randint(0, 59)))
        return self._add(row, "duplikat_samar")

    def overlap_other_student(self, base):
        s, e = self._slot(base)
        others = [st for st in self.students_of_tutor[base["tutor_id"]]
                  if st != base["student_id"] and not self.day[("s", st, base["lesson_date"])]]
        if not others:
            return None
        dur = self.rnd.choice([60, 60, 60, 90, 45])
        start = self.rnd.randint(max(DAY_START, s - dur + 15), min(e - 15, DAY_END - dur))
        start -= start % 5
        if not (start < e and s < start + dur):
            return None
        # must overlap ONLY the base session of this tutor
        if not self._free("t", base["tutor_id"], base["lesson_date"], start, start + dur, ignore=(s, e)):
            return None
        row = {k: base[k] for k in FIELDS}
        row.update(student_id=self.rnd.choice(sorted(others)), start_time=hm(start), duration_minutes=dur)
        row["submitted_at"] = ts_str(ts(base["submitted_at"]) + dt.timedelta(
            minutes=self.rnd.randint(1, 4320), seconds=self.rnd.randint(0, 59)))
        return self._add(row, "tumpang_tindih")

    def overlap_other_tutor(self, base):
        s, e = self._slot(base)
        date = base["lesson_date"]
        known = [t for t in self.tutors_of_student[base["student_id"]]
                 if t != base["tutor_id"] and not self.day[("t", t, date)]]
        others = known or [t for t in self.all_tutors if t != base["tutor_id"] and not self.day[("t", t, date)]]
        if not others:
            return None
        dur = self.rnd.choice([60, 60, 60, 90, 45])
        start = self.rnd.randint(max(DAY_START, s - dur + 15), min(e - 15, DAY_END - dur))
        start -= start % 5
        if not (start < e and s < start + dur):
            return None
        # must overlap ONLY the base session of this student
        if not self._free("s", base["student_id"], date, start, start + dur, ignore=(s, e)):
            return None
        row = {k: base[k] for k in FIELDS}
        row.update(tutor_id=self.rnd.choice(sorted(others)), start_time=hm(start), duration_minutes=dur)
        row["submitted_at"] = ts_str(ts(base["submitted_at"]) + dt.timedelta(
            minutes=self.rnd.randint(1, 4320), seconds=self.rnd.randint(0, 59)))
        return self._add(row, "tumpang_tindih_murid")

    def self_overlap(self, base):
        """Same tutor + same student + same date, overlapping the base, but NOT an exact
        copy and NOT a fuzzy duplicate (so the only finding is self_time_overlap).
        Three variants, in rotation:
          0 same start, different duration       (e.g. 14:00 60 min vs 14:00 90 min)
          1 same start + duration, different subjects
          2 start shifted 5-30 min AND different duration"""
        s, e = self._slot(base)
        dur = e - s
        variant = self.self_overlap_variant % 3
        row = {k: base[k] for k in FIELDS}
        if variant == 0:
            new_dur = self.rnd.choice([d for d in range(30, 181, 5) if d != dur])
            start = s
        elif variant == 1:
            subs = json.loads(base["subject_ids"])
            others = [x for x in self.all_subjects if x not in subs]
            if len(subs) > 1 and self.rnd.random() < 0.5:
                new_subs = subs[:-1]                                  # one subject fewer
            else:
                new_subs = subs[:-1] + [self.rnd.choice(others)]      # one subject swapped
            row["subject_ids"] = json.dumps(sorted(new_subs))
            start, new_dur = s, dur
        else:
            start = s + self.rnd.choice([-1, 1]) * self.rnd.randint(1, 6) * 5
            new_dur = self.rnd.choice([d for d in range(30, 181, 5) if d != dur])
        end = start + new_dur
        if not (DAY_START <= start and end <= DAY_END and start < e and s < end):
            return None
        # overlap ONLY the base, for both tutor and student
        if not (self._free("t", base["tutor_id"], base["lesson_date"], start, end, ignore=(s, e))
                and self._free("s", base["student_id"], base["lesson_date"], start, end, ignore=(s, e))):
            return None
        row.update(start_time=hm(start), duration_minutes=new_dur)
        row["submitted_at"] = ts_str(ts(base["submitted_at"]) + dt.timedelta(
            minutes=self.rnd.randint(1, 4320), seconds=self.rnd.randint(0, 59)))
        self.self_overlap_variant += 1
        return self._add(row, f"tumpang_tindih_sendiri")

    def run(self):
        rnd = self.rnd
        cfg = DEFAULT_CONFIG
        max_dur, min_dur = cfg["durationOutlierMinutes"], cfg["durationMinMinutes"]
        late, early = hhmm_to_minutes(cfg["lateStartAfter"]), hhmm_to_minutes(cfg["earlyStartBefore"])

        def long_():
            dur = rnd.randint(max_dur + 1, 300)
            return rnd.choice(range(DAY_START, LAST_MINUTE - dur, 15)), dur

        def short():
            return rnd.choice(range(13 * 60, 19 * 60, 15)), rnd.randint(10, min_dur - 1)

        def late_():
            start = rnd.randint(late + 1, 22 * 60 + 30)
            return start, rnd.randint(min_dur, min(120, LAST_MINUTE - start))

        def early_():
            return rnd.randint(5 * 60, early - 1), rnd.choice([60, 60, 90, 45, 120])

        shapes = {"panjang": long_, "pendek": short, "malam": late_, "pagi": early_}
        for name, shape in (("durasi_panjang", long_), ("durasi_pendek", short),
                            ("mulai_malam", late_), ("mulai_pagi", early_)):
            for _ in range(self.counts[name]):
                self.standalone(name, *shape())
        for kind, shape in shapes.items():
            name = f"kombinasi_dup_{kind}"
            for _ in range(self.counts[name]):
                self.exact_copy(self.standalone(name, *shape()), name)

        edges = [(13 * 60, max_dur), (14 * 60, min_dur), (late, 60), (early, 60)]
        for i in range(self.counts["batas_valid"]):
            start, dur = edges[i % 4]
            self.standalone("batas_valid", start, dur)

        # copies of real sessions: each base used once
        bases = self.base_pool[:]
        rnd.shuffle(bases)
        used = 0
        for scenario, fn in (("duplikat", lambda b: self.exact_copy(b, "duplikat")),
                             ("duplikat_samar", self.fuzzy_copy),
                             ("tumpang_tindih", self.overlap_other_student),
                             ("tumpang_tindih_murid", self.overlap_other_tutor),
                             ("tumpang_tindih_sendiri", self.self_overlap)):
            made = 0
            while made < self.counts[scenario] and used < len(bases):
                base = bases[used]
                used += 1
                if fn(base):
                    made += 1
            if made < self.counts[scenario]:
                sys.exit(f"Not enough clean real sessions: only {made}/{self.counts[scenario]} rows for "
                         f"{scenario}. Lower TARGET_SHARE for that label.")
        return self.synthetic


def main(src, dst):
    rnd = random.Random(SEED)
    real = []
    for r in csv.DictReader(open(src, encoding="utf-8-sig", newline="")):
        real.append({**{k: r[k] for k in FIELDS}, "duration_minutes": int(r["duration_minutes"]),
                     "is_synthetic": 0, "scenario": "real"})
    from build_training_set import label_sessions, prepare_sessions
    real_rows, _, _ = label_sessions(prepare_sessions(real))
    real_labels = Counter(r["validity"] for r in real_rows)
    counts = plan_counts(real_labels)
    print("real labels:", dict(real_labels))
    print("planned synthetic rows:", counts)
    synthetic = Injector(real, rnd, counts).run()
    rows = real + synthetic
    for r in rows:
        r["subject_ids"] = r["subject_ids"] if isinstance(r["subject_ids"], str) else json.dumps(r["subject_ids"])
    Path(dst).parent.mkdir(parents=True, exist_ok=True)   # create output folder if missing
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["is_synthetic", "scenario"])
        w.writeheader()
        w.writerows(rows)
    print(f"real {len(real)} + synthetic {len(synthetic)} = {len(rows)} rows")
    print("synthetic per scenario:", dict(Counter(r["scenario"] for r in synthetic)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
