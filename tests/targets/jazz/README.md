# Jazz Targets — ii–V–I in C

## Progression context

Uses `tests/progressions/ii_v_i_C.json`:

```
| Dm7 (0–2s) | G7 (2–4s) | Cmaj7 (4–6s) |
```

## ii_v_i_musical_line.json

**100 BPM, eighth notes, first downbeat at 1.0s.**

Line: D2 F2 A2 B2 | C#2 D2 E2 G2

```
Time    Note  Chord  Function          Class
------  ----  -----  ----------------  -------
1.000s  D2    Dm7    root              chord
1.300s  F2    Dm7    minor 3rd         chord
1.600s  A2    Dm7    5th               chord
1.900s  B2    Dm7    Dorian 6th (13th) scale
2.200s  C#2   G7     chromatic (♯4)    out
2.500s  D2    G7     5th               chord
2.800s  E2    G7     Mixolydian 6th    scale
3.100s  G2    G7     root              chord
```

### Harmonic intent

**ii bar (Dm7):** The first three notes outline the Dm7 arpeggio (root–m3–5th). B2 is the characteristic Dorian 6th — the note that distinguishes D Dorian from D natural minor and colours the ii chord with its jazz flavour.

**V bar (G7):** C#2 is a half-step chromatic approach from below into the 5th (D). It is outside G Mixolydian (♯4 relative to G) and serves as the single "outside" note in the line. The resolution to D2 on the next eighth makes the tension explicit. E2 is the Mixolydian 6th/13th, a smooth scale tone that walks up to the root G2.

### Expected harmonic classification output

```
chord tones   5  (62%)
scale tones   2  (25%)
outside       1  (12%)
```

### Suggested run command

```bash
python tools/analyze_fast_reference.py <recording.wav> \
  tests/targets/jazz/ii_v_i_musical_line.json \
  --apply-calibration \
  --progression tests/progressions/ii_v_i_C.json
```

### Bass fingering

The line is fingered in two shapes:

**Dm7 bar — D string:**
- D2: open
- F2: 3rd fret
- A2: 7th fret
- B2: 9th fret

**G7 bar — A string (shift at bar line), return to D string:**
- C#2: 4th fret A string
- D2: 5th fret A string
- E2: 7th fret A string
- G2: 5th fret D string
