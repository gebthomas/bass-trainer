# Fast-reference target files

Small stress-test corpus for evaluating pitch detection at higher tempos and
across instrument types.

## Files

| File | Instrument | String | Notes | Tempo | Note value | Spacing |
|------|------------|--------|-------|-------|------------|---------|
| fretted_G_string_fast_high.json    | Fretted electric | G | G3 A3 B3 C4 D4                        | 120 BPM | eighth | 0.250 s |
| fretless_G_string_fast_high.json   | Fretless electric | G | G3 A3 B3 C4 D4                       | 100 BPM | eighth | 0.300 s |
| upright_pizz_D_string_fast_mid.json | Upright (pizz) | D | D2 E2 F#2 G2 A2 B2 C#3 D3            |  90 BPM | eighth | 0.333 s |
| upright_bow_A_string_medium_low.json | Upright (bow)  | A | A1 B1 C#2 D2 E2                       |  70 BPM | quarter | 0.857 s |

All files: first note at 1.0 s, position labels fast_1, fast_2, …

## Recording protocol

- Leave one second of silence before the first note
- Play each note cleanly at the listed time
- Aim for even dynamics and consistent attack
- Keep notes short: do not let them ring into the next target window
- Use a metronome at the listed BPM; count in 4 beats before playing
- No effects or compression
- Record fretted and fretless electrically (DI or clean amp mic)
- Record upright_pizz with pizzicato; upright_bow with arco
- Use consistent input gain within each instrument group

## Purpose

These files are used to stress-test onset detection and pitch estimation at
tempos where note spacing approaches the onset detector's refractory period.
The high-register G-string files target octave confusion risk in pyin.
