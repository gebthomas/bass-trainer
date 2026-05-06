# Reference target files

## Overview

- **Fretted electric** targets are fret/position-based (chromatic landmarks).
- **Fretless electric** targets are also fret-position-based but stop at fret 19
  (the upper register is less commonly used on fretless without physical landmarks).
- **Upright** targets (pizz and bow) are scale-based because there are no frets.
  Note sequences follow a major scale up approximately 1.5 octaves per string.
  Record pizzicato and arco separately.

## Recording protocol

- Leave one second of silence before the first note
- Play each target note at the time listed in the JSON file
- Sustain each note for about 2 seconds
- Leave about 1 second of silence between notes
- Use clean fingerstyle or plucked tone for electric and upright_pizz
- Use a clean bow stroke for upright_bow
- No effects or compression
- Use consistent input gain across all takes for the same instrument

## File naming

`{instrument}_{string}_string_reference.json`

Instruments: fretted, fretless, upright_pizz, upright_bow
Strings: E, A, D, G

## Fretted electric — positions: 0, 3, 5, 7, 9, 12, 15, 17, 19, 24

| String | open | fret 3 | fret 5 | fret 7 | fret 9 | fret 12 | fret 15 | fret 17 | fret 19 | fret 24 |
|--------|-------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|----------|
| E | E1     | G1     | A1     | B1     | C#2    | E2     | G2     | A2     | B2     | E3     |
| A | A1     | C2     | D2     | E2     | F#2    | A2     | C3     | D3     | E3     | A3     |
| D | D2     | F2     | G2     | A2     | B2     | D3     | F3     | G3     | A3     | D4     |
| G | G2     | Bb2    | C3     | D3     | E3     | G3     | Bb3    | C4     | D4     | G4     |

## Fretless electric — positions: 0, 3, 5, 7, 9, 12, 15, 17, 19

| String | open | fret 3 | fret 5 | fret 7 | fret 9 | fret 12 | fret 15 | fret 17 | fret 19 |
|--------|-------|----------|----------|----------|----------|-----------|-----------|-----------|----------|
| E | E1     | G1     | A1     | B1     | C#2    | E2     | G2     | A2     | B2     |
| A | A1     | C2     | D2     | E2     | F#2    | A2     | C3     | D3     | E3     |
| D | D2     | F2     | G2     | A2     | B2     | D3     | F3     | G3     | A3     |
| G | G2     | Bb2    | C3     | D3     | E3     | G3     | Bb3    | C4     | D4     |

## Upright (pizz and bow) — scale-based sequences

Position labels are scale_1, scale_2, … (no fret numbers).

  E: E1, F#1, G#1, A1, B1, C#2, D#2, E2, F#2, G#2, A2
  A: A1, B1, C#2, D2, E2, F#2, G#2, A2, B2, C#3, D3
  D: D2, E2, F#2, G2, A2, B2, C#3, D3, E3, F#3, G3
  G: G2, A2, B2, C3, D3, E3, F#3, G3, A3, B3, C4
