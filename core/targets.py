import json

def load_targets(filename="targets.json"):
    with open(filename, "r") as f:
        return json.load(f)

def nearest_target(onset_time, targets):
    return min(targets, key=lambda t: abs(t["time"] - onset_time))

def timing_error_ms(onset_time, target_time):
    return round((onset_time - target_time) * 1000)

def compare_note(detected_note, target_note):
    return detected_note == target_note