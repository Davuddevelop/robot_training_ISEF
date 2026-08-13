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

-----

*Format for new entries: date — one-line headline, then Problem/Fix/Evidence/Caveat as needed.
Keep every claim traceable to a specific script, config diff, or log line.*
