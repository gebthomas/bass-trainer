def get_match_window(target_index, targets):
    prev_gap = None
    next_gap = None

    if target_index > 0:
        prev_gap = targets[target_index]["time"] - targets[target_index - 1]["time"]
    if target_index + 1 < len(targets):
        next_gap = targets[target_index + 1]["time"] - targets[target_index]["time"]

    if prev_gap is not None and next_gap is not None:
        local_gap = min(prev_gap, next_gap)
    elif prev_gap is not None:
        local_gap = prev_gap
    elif next_gap is not None:
        local_gap = next_gap
    else:
        local_gap = 0.0

    window = local_gap * 0.40
    return max(0.12, min(0.40, window))


class TargetMatcher:
    def __init__(self, targets, timing_offset_ms, results_logger):
        self.targets = list(targets)
        self.timing_offset_ms = timing_offset_ms
        self.results_logger = results_logger
        self.current_target_index = 0
        self.target_candidates = []

    def add_onset_candidate(self, onset_time, note, raw_error, corrected_error, pitch_ok, timing_label):
        self.target_candidates.append({
            "onset_time": onset_time,
            "note": note,
            "raw_error": raw_error,
            "corrected_error": corrected_error,
            "pitch_ok": pitch_ok,
            "timing_label": timing_label,
        })

    def _finalize_target(self, target):
        if not self.target_candidates:
            self.results_logger.append_miss(target)
            return

        correct_candidates = [c for c in self.target_candidates if c["pitch_ok"]]
        if correct_candidates:
            chosen = min(correct_candidates, key=lambda c: abs(c["corrected_error"]))
        else:
            chosen = min(self.target_candidates, key=lambda c: abs(c["raw_error"]))

        for candidate in self.target_candidates:
            if candidate is chosen:
                self.results_logger.append_hit(
                    target,
                    candidate["onset_time"],
                    candidate["raw_error"],
                    candidate["corrected_error"],
                    candidate["pitch_ok"],
                    candidate["timing_label"],
                    candidate["note"],
                )
            else:
                self.results_logger.append_extra(candidate["note"], candidate["onset_time"])

        self.target_candidates = []

    def process_onset_against_targets(self, onset_time, note, timing_error_fn, compare_note_fn):
        while self.current_target_index < len(self.targets):
            target = self.targets[self.current_target_index]
            window = get_match_window(self.current_target_index, self.targets)
            if onset_time > target["time"] + window:
                print(f"    Missed target {target['note']} @ {target['time']:.3f}s")
                self._finalize_target(target)
                self.current_target_index += 1
                continue
            break

        if self.current_target_index >= len(self.targets):
            print(f"    Extra note at {onset_time:.3f}s: no remaining target")
            self.results_logger.append_extra(note, onset_time)
            return True

        target = self.targets[self.current_target_index]
        window = get_match_window(self.current_target_index, self.targets)

        if onset_time < target["time"] - window:
            print(f"    Extra note at {onset_time:.3f}s: before target window")
            self.results_logger.append_extra(note, onset_time)
            return True

        raw_error = timing_error_fn(onset_time, target["time"])
        corrected_error = raw_error + self.timing_offset_ms
        pitch_ok = compare_note_fn(note, target["note"])

        if abs(corrected_error) < 30:
            timing_label = "tight"
        elif abs(corrected_error) < 80:
            timing_label = "ok"
        else:
            timing_label = "off"

        self.add_onset_candidate(
            onset_time,
            note,
            raw_error,
            corrected_error,
            pitch_ok,
            timing_label,
        )
        return True

    def finalize_remaining_targets(self):
        while self.current_target_index < len(self.targets):
            self._finalize_target(self.targets[self.current_target_index])
            self.current_target_index += 1
