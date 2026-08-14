# EXPERIMENT_LOG.md — Dated Record of Findings

Every real, evidence-backed finding from training and hardware work, in the order it happened.
Purpose: so the interview narrative doesn't rely on memory — every claim here traces to a
specific run, config, or measurement. Add a new entry any time an experiment changes what we
believe, not for routine progress updates.

-----

## 2026-08-13 — Hardware corrected: BiBoard V1.0 / Bittle X, not NyBoard / base Bittle

Physical hardware photos showed the assembled robot's control board is silkscreened
**"BiBoard V1.0"**, not the NyBoard originally assumed in the project brief. Verified from
Petoi's own documentation and product pages:

- BiBoard V1.0 = ESP32-MINI-1 module, dual-core Xtensa LX6, 4MB flash, built-in 2.4GHz WiFi +
  Bluetooth LE, up to 12 PWM servos, runs OpenCat firmware (same command family as NyBoard:
  `i`/`m` set joint angles, `j` queries them, `v`/`V` read IMU/gyro).
- BiBoard V1.0 specifically pairs with **Bittle X V2** per Petoi's docs — needs visual
  confirmation against the physical unit, since the robot body alone doesn't show a model
  number in the photos taken.
- Practical implication: comms options now include wired serial, WiFi, and Bluetooth (NyBoard
  was serial-only, BLE required a separate dongle). Recommended wired serial for Phase 3 —
  more deterministic latency, which matters for the control-latency system-ID work — but this
  is a recommendation, not yet a tested decision.
- `CLAUDE.md` §4 and §6 updated to match.

## 2026-08-13 — Reward rebalance: forward_velocity_coeff 5.0→3.5, alive_bonus 0.5→1.0

**Problem found:** benchmarking the difficulty-0.80 terrain policy (`notebooks/16_terrain_benchmark.py`)
showed the RL policy beat the fixed scripted gait on distance at every rough difficulty, but fell
*more often* than the fixed gait at every one of those same difficulties (e.g. fall_rate 0.60 vs 0.30
at difficulty 0.25). Root cause: at `forward_velocity_coeff=5.0`, `alive_bonus=0.5`, a modest 0.1 m/s
speed already scored as much reward per step as the entire alive bonus — so risking a fall for extra
speed cost the policy almost nothing.

**Fix tested:** short 200k-step A/B at terrain difficulty 0.4, same seed, random init, comparing
`forward=5.0/alive=0.5` (current) vs `forward=3.5/alive=1.0` (proposed). Old balance oscillated up to
fall_rate 1.00 by step 120k; new balance never fell once across all 5 checkpoints, and covered more
distance (0.43m vs 0.336m at 200k steps). Change committed to `src/sim/config.py`.

**Caveat:** this was a single seed, single difficulty, short-horizon test — decisive at that scale,
but not proof it holds over a full multi-million-step run (see next entry).

## 2026-08-13 — Entropy runaway discovered in the full-scale rebalanced retrain

A full fresh retrain under the new reward balance reached curriculum difficulty 0.90 faster than the
old run reached 0.70 (3.5M steps vs 4.3M) — but the policy's action-noise parameter (`std` in the PPO
logs) climbed continuously and without bound across the entire visible training window: 2.59 → 4.57
over ~1.5M steps, never stabilizing. Verified mechanistically from the logged loss components: with
`ent_coef=0.01` and `entropy_loss≈-21` near the end, the entropy term alone (`0.01 × -21 = -0.21`)
accounted for nearly the entire total loss (`≈-0.2` to `-0.25`) — the policy-gradient term
(`≈-0.04`) was almost irrelevant by comparison. The optimizer was maximizing randomness, not
improving the gait.

**Confirmed by benchmark:** the resulting policy walked measurably *less* far than the pre-rebalance
model at every terrain difficulty (e.g. 1.054m → ~0.83m at difficulty 0.0, 0.405m → ~0.26m at
difficulty 1.0). The old backed-up model (`models/bittle_terrain_v1_backup/`, difficulty 0.70,
1.190m, real benchmarked numbers in `data/terrain_benchmark.csv`) remains the best result to date.

**Diagnosis:** this is the opposite failure mode from the earlier entropy-*collapse* problem this
project already fixed once (std stuck at ~0.137 in an old 16M-step checkpoint). Both are the same
underlying gap: nothing bounds `std` in either direction over a long enough run. Short validation
tests (200k steps) don't catch this — it's a slow-burn effect that only appears over millions of
steps.

**Fix proposed, not yet applied:** hard-clamp `std` to a sane range (e.g. roughly [0.1, 1.5]) via a
callback, checked periodically during training, so it can neither collapse nor explode regardless of
how the entropy-bonus math behaves at any given reward scale.

## 2026-08-13 — Competition-strategy research: commercial hardware is not a disqualification risk

**Worry:** using a purchased Petoi Bittle X (not self-built hardware) might read as "not our own work" to
judges, prompting a push toward adding rescue/human-detection framing to compensate.

**Researched (5-agent pass, sourced via WebSearch — WebFetch was blocked for every domain in this
sandbox, so treat as high-confidence secondary sourcing, not primary-source-verified; verify directly
before quoting in competition paperwork):**

- **ISEF's own rules** ban "kit building" (doing nothing but assembling a purchased kit), not the use of
  commercial hardware as a base for real research. No rule anywhere prohibits purchased platforms.
- **Sabahın Alimləri directly mirrors ISEF's rules** (matching form numbers 1/1A/1B/7, near-identical
  "kit building" prohibition wording) — it is a genuine ISEF-affiliated national fair, not independently
  written rules.
- **Real precedent:** Michelle Wang's ISEF 2023 project (ROBO047) used unmodified commercial DJI Tello
  drones with custom control/coordination software as the novel contribution — won 4th place Grand Award
  in Robotics AND an Air Force Research Labs special award simultaneously.
- **The judging rubric (100 pts, Engineering track, governs ROBO category):** Research Problem 10, Design/
  Methodology 15, Execution 20, Creativity 20, Presentation 35 (poster 10 + interview 25). Only **4 of 100
  points** are tied to "real-world impact" (one interview sub-item). The other 96 are rigor, methodology,
  execution, originality, and whether the student can defend the work independently.
- **Found the actual citation for our "not first to do sim-to-real on Bittle" novelty claim:** Neuman et
  al., RSS 2022 Sim2Real Workshop, did sim-to-real specifically on this robot — added to BIBLIOGRAPHY.md
  as high-priority reading, since the novelty framing in CLAUDE.md §3 needs to point at this, not just
  assert it.

**Decision:** do not add "helping humanity" / rescue / human-detection framing — it's explicitly excluded
in CLAUDE.md §3 and §10 already, and the rubric confirms it isn't what's actually being scored. The
existing CV plan (pretrained person-detector as an honestly-labeled demo on the rubble terrain, not a
rescue claim) remains available as a visual demo element, but the project's strength is the ablation
study's rigor, not a product pitch.

**Open item:** have mentor verify the Sabahın Alimləri rules page directly and/or contact the fair's
review committee about the hardware setup before formal experiments — the specific "Robotics & AI
category hardware disclosure clause" sub-question could not be confirmed either way from this sandbox.

## 2026-08-14 — Five measurement/plumbing bugs found; benchmarks had been measuring the wrong model

A codebase audit before further training found that most of the poor rough-terrain results
were caused by **measurement and plumbing bugs, not by the learning algorithm.** All five are
verified against source or measured directly, and all are fixed in this commit.

**1. `best_model.zip` had been saving an early, undertrained snapshot every run.**
`TerrainCurriculumCallback` compared `_best_dist` as a single global scalar *across all
curriculum difficulties*. Distance naturally falls as terrain hardens, so a best set at
difficulty 0.20 (0.865 m, step 300k) could never be beaten at difficulty 1.0 (~0.3 m).
Evidence: in the most recent 5M-step run, `best` was set at step 300,000 and never updated
across the remaining 4.7M steps — even though the policy went on to reach difficulty 1.0.
**Consequence: every benchmark in `data/terrain_benchmark.csv` measured an early-curriculum
snapshot, not the trained policy.** Fixed by ranking checkpoints lexicographically on
(difficulty, distance) — harder terrain wins outright, ties broken by distance — plus a
per-rung `best_by_difficulty` archive and a `best_model_meta.json` that
`16_terrain_benchmark.py` now prints, so this can never go unnoticed again.

**2. Resumed runs silently ignored the configured learning rate.** SB3 builds
`model.lr_schedule` inside `PPO.load()`, *before* `apply_resume_overrides` assigned
`model.learning_rate`; training reads the schedule, never the attribute. Measured directly:
assigning `learning_rate = 1e-9` to a loaded model left `lr_schedule(1.0)` at the
checkpoint's baked-in `0.0003`; calling `model._setup_lr_schedule()` afterwards correctly
changed it to `1e-9`. Every resumed run had therefore been training at the checkpoint's
original rate. (`ent_coef`/`target_kl` are read directly each update, so those did work.)

**3. Checkpoints saved 4× less often than intended.** `CheckpointCallback` counts
*vectorised* steps, so `save_freq=100_000` with `n_envs=4` saved every 400k real steps —
and would degrade further as `n_envs` grows. Fixed by dividing by `n_envs`; verified that
checkpoints now land at the requested interval.

**4. Debris collisions could be silently missed.** `_place_debris` resizes `geom_size` at
runtime, but MuJoCo's broadphase bounding sphere `geom_rbound` is computed at model-compile
time and does not update on resize (measured: stays 0.0469 when it should be 0.0768 for a
grown chunk). Contacts near a chunk's corners could be filtered out — meaning the "hard"
terrain was potentially easier than believed. Now recomputed on every placement.

**5. Nothing was seeded.** No `seed=` on PPO, no env seeding, and the curriculum's own
evaluation env was unseeded — so consecutive evaluations faced *different terrain*, making
eval-to-eval changes partly a measurement of the map rather than the policy. Added
`BITTLE_SEED`, and the curriculum now evaluates on 10 fixed seeds.

**Also changed:** `n_eval` 3 → 10 (with 3 episodes, `fall_rate` could only be 0/0.33/0.67/1.0,
so every promotion decision was near a coin flip); added curriculum **demotion** after 2
consecutive bad evaluations with a `min_steps_at_difficulty` anti-thrash guard (promotion was
previously one-way, so a single lucky evaluation could strand the robot on ground it could
not hold); added `BITTLE_FIXED_DIFFICULTY` for controlled A/B runs and
`BITTLE_MODEL_PATH`/`BITTLE_BENCH_EPISODES` overrides for the benchmark.

**Caveat / not yet done:** none of this has been re-benchmarked yet. The immediate next step
is to re-run `16_terrain_benchmark.py` against the *final* model of the last run — which has
never actually been measured — before drawing any conclusion about how good the current
policy really is.

-----

*Format for new entries: date — one-line headline, then Problem/Fix/Evidence/Caveat as needed.
Keep every claim traceable to a specific script, config diff, or log line.*
