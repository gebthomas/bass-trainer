# Pocket Lab: A Laboratory for Groove, Timing, and Bass Performance

## Vision

Pocket Lab is a collection of analytical and training tools designed to help musicians understand, visualize, and improve groove, timing, note choice, and musical interaction.

Unlike traditional timing trainers that simply report whether a note was early or late, Pocket Lab seeks to help musicians understand why a performance feels good, how expert musicians create groove, and how timing, harmony, articulation, and structure interact.

The philosophy is investigative rather than judgmental:

> Don't simply score the groove. Study the groove.

---

# Core Concepts

Pocket Lab treats music as several interacting layers:

1. Timing and rhythmic placement
2. Bass-drum interaction
3. Note selection and harmonic function
4. Song structure
5. Musical feel and pocket
6. Performer consistency

The goal is to provide tools that allow these layers to be examined independently and together.

---

# Proposed Tools

## Beat Microscope

A synchronized visual and auditory inspection tool.

Displays:

* Waveform
* Onset envelope
* Beat grid
* Shuffle grid
* Detected note attacks
* Timing offsets

Supports:

* Zooming into individual notes
* Listening to short loops
* Examining onset-detection behavior
* Distinguishing string noise from note attacks

Questions answered:

* What is the true onset of the note?
* Why did the detector trigger here?
* Was the note early or late?
* Was this a downbeat or passing note?

---

## Pocket Meter

Provides summary timing statistics.

Metrics may include:

* Mean timing offset
* Mean absolute error
* Timing variability
* Groove spread
* Timing histogram
* Early/late tendency

Questions answered:

* Am I rushing?
* Am I dragging?
* How consistent am I?

---

## Groove Comparator

Compares:

* Multiple takes
* Student versus recording
* Student versus expert

Displays:

* Timing differences
* Groove profiles
* Consistency metrics

Questions answered:

* Which take grooves better?
* What changed?
* Am I improving?

---

## Groove Mapper

Tracks groove characteristics across an entire song.

Displays:

* Timing versus song position
* Groove consistency versus song position
* Section-by-section summaries

Questions answered:

* Where do I lose the groove?
* Where do I settle into the groove?
* Which sections are strongest?

---

## Feel Explorer

Experimental environment for studying pocket.

Allows controlled timing shifts:

* -60 ms
* -30 ms
* 0 ms
* +30 ms
* +60 ms

Supports blind listening comparisons.

Questions answered:

* What does "behind the beat" sound like?
* What timing do I prefer?
* How much timing variation is perceptible?

---

## Beat Trainer

Real-time timing trainer.

Conceptually similar to Guitar Hero.

Features:

* Moving note targets
* Real-time timing feedback
* Visual indication of early/late performance

Questions answered:

* Can I execute intended timing?
* Can I maintain a groove under pressure?

---

## Rhythm Relationship Analyzer

Studies bass-drum interaction.

Extracts:

* Kick events
* Snare events
* Bass attacks

Measures:

* Bass relative to kick
* Bass relative to snare
* Changes throughout the song

Questions answered:

* Does the bassist lock to the kick?
* Does the bassist sit behind the snare?
* How does the relationship change?

---

## Line Decoder

Analyzes bass note content.

Identifies:

* Notes played
* Note durations
* Articulations
* Repeated patterns

Questions answered:

* What did the bassist actually play?
* How can I learn this line?

---

## Harmony Explorer

Examines harmonic function.

Identifies:

* Chord progression
* Scale tones
* Passing tones
* Approach notes
* Walking patterns

Questions answered:

* Why were these notes chosen?
* How does the bassist support the harmony?

---

## Structure Mapper

Analyzes song organization.

Identifies:

* Verse
* Chorus
* Bridge
* Solo
* Outro

Measures:

* Groove changes
* Density changes
* Complexity changes

Questions answered:

* How does the bass evolve through the song?
* How does the bassist support song structure?

---

## Groove Archeology

Studies expert recordings.

Potential analyses:

* James Jamerson
* Carol Kaye
* Donald "Duck" Dunn
* Pino Palladino
* Nathan East
* John Paul Jones

Questions answered:

* What makes these players sound unique?
* How consistent are they?
* Where do they sit relative to the beat?

---

# Extraction and Analysis Tools

## Bass Extractor

Separates or emphasizes bass content from full recordings.

Possible approaches:

1. Frequency-domain filtering
2. Event extraction
3. Source separation models
4. Manual correction

Outputs support:

* Line Decoder
* Groove Comparator
* Harmony Explorer

---

## Rhythm Extractor

Separates or emphasizes rhythmic content.

Potential outputs:

* Kick events
* Snare events
* Hi-hat patterns
* Ride patterns

Supports:

* Beat Microscope
* Rhythm Relationship Analyzer
* Groove Mapper

---

## Spectrum Analyzer

Pitch-focused analysis tool.

Displays:

* Spectrograms
* Harmonic content
* Fundamental pitch
* Note tracking

Supports:

* Learning bass lines
* Identifying articulations
* Mapping note choices
* Comparing student and expert performances

Questions answered:

* What notes are being played?
* Which note is sounding at a given time?
* What harmonic role does the note serve?

---

# Guiding Philosophy

Pocket Lab is not intended to answer:

> Was the note correct?

Pocket Lab is intended to answer:

> What happened?

The system should help musicians investigate timing, groove, harmony, note choice, articulation, and musical interaction in order to build deeper listening skills and more informed practice habits.

