# Design Principles

See `design_principles.md` for the detailed principles governing feature, UX, and architectural decisions.

---

# 1. Musical Usefulness Over Technical Sophistication

A simpler system that meaningfully improves practice is preferable to a technically impressive system with limited practical value.

The project should prioritize:
- effective practice workflows
- rehearsal readiness
- retention
- fluency
- consistency
- adaptability

over:
- unnecessary complexity
- algorithmic novelty
- excessive analysis depth
- engineering for its own sake

Technical sophistication is valuable only when it improves musical outcomes.

---

# 2. Support Internalization, Not Dependency

The system should help players gradually strengthen:
- internal pulse
- harmonic understanding
- form awareness
- listening
- memory
- recovery ability

The long-term goal is not dependence on the tool.

The tool should help musicians become increasingly independent from:
- charts
- tabs
- dense click tracks
- recordings
- visual prompts
- external timing references

---

# 3. Progressive Removal of Scaffolding

Early learning may require strong support:
- visible notation
- tablature
- count-ins
- dense metronome cues
- highlighted passages
- recorded bass parts

As fluency develops, those supports should gradually diminish.

The system should encourage transition toward:
- internal pulse
- memory-based performance
- harmonic prediction
- cue-independent playing
- ensemble adaptability

---

# 4. Groove and Stability Matter More Than Quantized Precision

Musical timing is not equivalent to robotic alignment with a click.

The system should value:
- consistency
- stable pulse
- groove continuity
- recovery after disruption
- ensemble reliability

Metrics emphasizing:
- timing variance
- stability
- recovery
- continuity

are often more musically meaningful than minimizing average timing error alone.

Current metrics focus primarily on timing accuracy as a measurable and operationally useful starting point. They are not intended to fully capture groove continuity, recovery ability, or broader musicianship — those remain the longer-term direction.

---

# 5. Realtime Feedback Should Have Low Cognitive Load

During performance-oriented practice, the player’s attention should remain focused on:
- listening
- groove
- musical interaction
- anticipation
- form awareness

Realtime feedback should therefore be:
- sparse
- calm
- supportive
- minimally distracting

Avoid:
- constant numerical displays
- dense visual clutter
- excessive alerts
- punitive feedback loops

Detailed analytics are often more useful after a phrase or session than during active playing.

---

# 6. Train Recovery, Not Just Error Avoidance

Real ensemble performance includes:
- wrong turns
- tempo drift
- missed entrances
- unstable accompaniment
- interruptions
- imperfect communication

The system should eventually help train:
- recovery
- re-entry
- pulse retention
- form recovery
- adaptive listening
- ensemble stabilization

Musicianship includes the ability to continue gracefully under imperfect conditions.

Implementing recovery-oriented practice will likely require architectural extensions beyond the current target-matching and session-engine model — see `future_directions.md` for context.

---

# 7. Preserve Musical Context Whenever Possible

Practice should remain connected to real musical situations.

Ear training, timing drills, improvisation, and harmonic work should ideally occur within:
- grooves
- phrases
- forms
- chord progressions
- ensemble-like contexts

Contextual fluency is generally more transferable than isolated abstract drills.

---

# 8. Practice Guidance Is More Valuable Than Practice Grading

The project should emphasize:
- productive next steps
- continuity
- prioritization
- reinforcement
- adaptive progression

rather than:
- harsh evaluation
- arbitrary scoring
- competitive gamification

The goal is to support sustained, effective practice habits over time.

Grading as feedback — scoring notes as good, warn, or miss — is useful and appropriate. What the system should avoid is punitive evaluation, competitive ranking, or scoring designed to discourage rather than redirect.

---

# 9. Optimize for Long-Term Repertoire Maintenance

Many working musicians manage large and evolving repertoires.

The system should eventually support:
- memory reinforcement
- repertoire tracking
- decay detection
- targeted refresh practice
- gig preparation
- difficult-section isolation

The project should help musicians maintain fluency over months and years, not merely complete isolated exercises.

---

# 10. Respect Different Modes of Musicianship

Different musical contexts require different skills.

Examples:
- exact reproduction in pit orchestra work
- structural fluency in country charts
- improvisation in jazz
- groove consistency in rhythm sections
- recovery and adaptation in live ensembles

The system should avoid assuming there is only one “correct” form of musical competence.

This principle applies to how feedback is framed and interpreted, not as a mandate to build separate mode-specific workflows for each context.

---

# 11. Prefer Adaptation Over Static Difficulty

Difficulty should eventually adapt based on:
- consistency
- retention
- confidence
- fatigue
- repeated weak points
- upcoming performance needs

Adaptive support is generally more useful than fixed practice structures.

---

# 12. Avoid Premature Complexity

Do not introduce major complexity until simpler approaches demonstrate real musical value.

Examples:
- large ML systems
- advanced DSP pipelines
- cloud-first architecture
- elaborate social features
- highly polished UI frameworks

The project should remain grounded in practical practice outcomes.

---

# 13. The System Should Help Musicians Become More Self-Sufficient

The long-term direction of the project is not dependency creation.

Success means the player gradually becomes:
- more confident
- more internally stable
- more adaptable
- more musically fluent
- less reliant on external support

The best outcome is stronger real-world musicianship outside the software itself.