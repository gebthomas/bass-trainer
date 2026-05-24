# Session Retention Policy

This document defines how the bass-trainer system manages accumulated session
logs over time: what to keep, what to delete, and when and why.

No code implementing this policy exists yet.  This document is the design
record that precedes implementation.

---

## 1. Why Oldest-First Deletion Is Not Appropriate

The naive approach to bounded storage — delete the oldest file when a limit
is reached — is appropriate for temporary caches.  Practice logs are not a
cache.  They are a longitudinal record of skill development, and the oldest
entries often carry the highest long-term value.

Specific reasons not to delete oldest-first blindly:

**Baseline value.**  An old session log from when a player first learned a
piece is the only way to measure how far they have come.  Deleting it removes
the reference point that makes progress visible.  The system's design principle
9 ("optimize for long-term repertoire maintenance") depends on keeping enough
history to detect skill decay or improvement over months.

**Milestone asymmetry.**  Rare good events — the first time a difficult
passage was nailed cleanly, a personal-best timing score — are not evenly
distributed across time.  They may be concentrated in old logs.  An oldest-
first deletion policy is blind to this.

**Redundancy is not correlated with age.**  A player who does the same
walking-bass warmup every morning accumulates thirty nearly-identical logs in a
month.  Any one of those logs is informative; twenty-nine of them are redundant.
The redundant sessions are recent, not old.  Oldest-first deletion discards the
wrong end.

**Sparsely visited exercises.**  A player may return to a piece they have not
practised in six months.  If the policy has already discarded older logs for
that exercise, there is no baseline to compare against.  This is precisely the
scenario where longitudinal history is most valuable.

**Practical observation.**  Session JSON files are small (see section 3).  The
pressure to delete arises primarily from accumulation of routine repetitions,
not from age.

---

## 2. Desired Retention Behavior

The goal is a retention shape that reflects how practice history is actually
used: recent history consulted in detail, older history consulted for trend
and context, certain sessions kept indefinitely regardless of age.

### Dense recent history

All sessions within a recent window (e.g., the last 14–30 days) should be
kept in full without pruning.  This is the working memory of practice: a player
reviewing last week's sessions expects to see all of them.

### Sparse older history

Beyond the recent window, history becomes a sample rather than a complete
record.  One representative session per week, then one per month, is enough to
support trend analysis and regression detection without accumulating thousands
of files over years.  Redundant sessions within a time bucket — same exercise,
similar metrics, no milestone flag — are the natural candidates for deletion.

### Keep milestones and best sessions

Some sessions have irreplaceable historical significance independent of their
age:

- First-time achievement on a difficult passage
- Personal-best timing or hit-rate on a given exercise
- Sessions the user has explicitly marked

These should be protected from deletion until the user explicitly chooses to
remove them.

### Delete redundant routine logs first

When space must be reclaimed, the correct order is:

1. Unprotected sessions in the oldest time bucket with no distinguishing
   metrics (same exercise, no milestone, not a best performance)
2. Unprotected sessions in the next bucket
3. Continue forward in time until the limit is satisfied

Protected sessions are never candidates for automatic deletion regardless of
age or budget.

---

## 3. Session JSON vs. Future Audio Recordings

These two artifact types have fundamentally different storage characteristics
and require separate retention treatment.

### SessionLog JSON files

A SessionLog JSON file records the events, timestamps, and aggregate metrics
for a single practice run.  For a typical session of 60–120 targets, the file
size is on the order of 10–50 KB.  Thousands of sessions would occupy tens to
low hundreds of megabytes — negligible on any modern device.

The practical pressure to manage JSON accumulation is primarily organizational
(too many files to navigate) rather than storage (not enough space).  The
initial retention policy therefore enforces limits that are generous by storage
standards but keep the file count navigable.

### Audio recordings (future)

A mono 44.1 kHz / 16-bit WAV file runs approximately 5 MB per minute.  A
10-minute practice recording is 50 MB.  A week of daily recordings is 350 MB
or more.  These numbers compound quickly and represent a qualitatively different
storage problem.

Audio retention must be governed by separate, stricter rules that are not part
of the initial implementation.  Decisions about whether to keep audio at all,
which format to use, and whether to allow audio deletion automatically are
deferred until the audio layer exists and there is real usage data.

**The initial retention policy explicitly does not touch audio files.**

---

## 4. Proposed Initial Implementation

### Scope

- Operates on SessionLog JSON files only (`.session.json` extension).
- Does not delete audio recordings, exercise files, alignment files, or any
  other file type.
- Operates within a single designated root directory.

### Limits

Two configurable limits govern when cleanup is triggered:

| Parameter | Description | Suggested default |
|---|---|---|
| `max_logs` | Maximum number of `.session.json` files in the root | 500 |
| `max_total_bytes` | Maximum total size of all `.session.json` files | 100 MB |

Either limit being exceeded triggers a cleanup pass.  The cleanup pass removes
the minimum number of unprotected logs needed to bring both metrics back under
their limits.

### Protection flags

A session log is protected from automatic deletion if its `metadata` dict
contains any of the following keys with the value `"true"`:

| Key | Meaning |
|---|---|
| `keep` | General-purpose "do not delete" flag |
| `milestone` | Marks a first-achievement or notable event |
| `best` | Marks a personal-best performance on the associated exercise |
| `starred` | User-starred this session explicitly |

String comparison is case-insensitive (`"True"`, `"TRUE"` are equivalent to
`"true"`).  Protected logs are never candidates for automatic deletion,
regardless of age or whether limits are exceeded.

### Deletion order within the unprotected set

Among unprotected logs, delete in ascending order of `started_at` timestamp
(oldest unprotected first).  This is a safe approximation given that the
initial implementation does not yet implement time-bucket sampling (see
section 5).  It is less harmful than naive global oldest-first because
protection flags allow the user to pin sessions that should survive.

### What is never deleted

- Any log whose `metadata` contains a protection flag set to `"true"`.
- Any file that does not match the `.session.json` suffix.
- Any file outside the designated root directory.
- Any file reached via a symbolic link.

---

## 5. Proposed Future Implementation

The initial implementation is intentionally simple.  These capabilities are the
natural next tier, to be added when real usage patterns are known.

### Time-bucket sampling

Rather than deleting oldest-unprotected-first globally, the future
implementation should thin the archive by time bucket:

```
last 30 days    →  keep all
last 3 months   →  keep at most 1 per week per exercise
older           →  keep at most 1 per month per exercise
```

Within a bucket, prefer keeping the session with the highest hit rate or lowest
mean absolute timing error (whichever metric is most meaningful for that
exercise type).  Discard the rest unless they are protected.

### Per-exercise retention

The time-bucket approach should be applied per exercise, not globally.  A
player who practises one piece daily and another once a month should not see
the infrequently-practised piece's history thinned at the same rate as the
daily piece.

Grouping by `exercise_path` in the session log is sufficient for the initial
per-exercise pass.

### Keep best recent performance per exercise

Beyond protection flags, the system should automatically retain the single
best-metrics session per exercise per time bucket even if the user has not
starred it.  "Best" should be determined at cleanup time from the `metrics`
field (`hit_rate`, `mean_abs_error_s` from `LogMetrics`, or equivalent).

### Separate retention rules for audio

When audio recording is introduced, it will need its own retention policy with:

- Much stricter total-byte limits
- A separate `max_audio_bytes` parameter independent of `max_total_bytes`
- Explicit user confirmation before any audio file is deleted
- No automatic audio deletion without a deliberate opt-in setting
- Protection flags honoured independently per file type

Audio and JSON limits must be evaluated and enforced independently.  A player
hitting their JSON limit should never trigger audio deletion, and vice versa.

---

## 6. Safety Rules

These rules are non-negotiable and must be enforced unconditionally by any
implementation of this policy, initial or future.

1. **Never delete outside the root.**  All file operations must resolve to an
   absolute path and verify that the resolved path is a descendant of the
   designated root directory before any deletion is attempted.  Path traversal
   sequences (`..`) are not permitted.

2. **Never follow symlinks.**  All candidate files must be identified using a
   non-symlink-following stat (e.g. `os.lstat` or `Path.lstat()`).  If a path
   is a symlink, skip it silently.

3. **Only delete `.session.json` files.**  The filename must end with
   `.session.json` (case-sensitive).  No other extension is touched, including
   `.json` without the `.session.` infix.

4. **Ignore non-session files.**  Files that do not match the suffix pattern
   are invisible to the retention system.  They are not counted against any
   limit and are never selected for deletion.

5. **Verify root before operating.**  The designated root must exist and be a
   real directory (not a symlink to a directory) before any scan or deletion
   begins.  If the root does not exist or is not a directory, the retention
   function raises rather than creating or guessing.

6. **Protected logs are never deleted.**  A log with any protection flag set
   is removed from the candidate set before any deletion logic runs.  There is
   no override, no "force" option, and no situation in which the system deletes
   a protected log automatically.

7. **Deletion is logged before execution.**  Any file selected for deletion
   must be recorded (path, size, reason) in a return value or log structure
   before `unlink` is called.  The caller can inspect what was deleted.  This
   supports auditability and potential future undo.
