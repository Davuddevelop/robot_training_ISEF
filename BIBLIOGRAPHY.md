# Bibliography — sources we actually use

Competition rule: **≥5 sources**, tracked as we use them (not invented at the end).
Each entry says WHAT we took from it, so the "borrowed vs. original" line (Rule 3.12)
is always defensible.

---

## Prior work we build on (Rule 3.12 — cite what's borrowed)

1. **OpenCat Gym** — ger01d. RL gait training for Bittle/Nybble in simulation
   (PyBullet + Stable-Baselines3 PPO).
   https://github.com/ger01d/opencat-gym
   *Used as:* reference for the RL environment design (8-joint offset action
   space, PPO/SB3 choice, reward-term structure). We borrow the flat-ground
   RL-locomotion approach as our baseline. We do NOT use their PyBullet
   simulator, their 246-dim joint-history observation, or their code.

2. **opencat-gym-sim2real** — ger01d. Sim-to-real transfer of a trained Bittle
   gait to real hardware (switched NyBoard → BiBoard; custom ESP32 firmware).
   https://github.com/ger01d/opencat-gym-sim2real
   *Used as:* reference for the sim-to-real deployment path and a documented
   hardware finding (NyBoard may be too slow for real-time policy serial I/O —
   flagged for our mentor). We do NOT copy their firmware or serial code.

3. **MuJoCo Bittle model** — AIWintermuteAI (URDF) → adapted for MuJoCo by the
   Petoi community (gravesreid/mujoco_mpc_bittle, fork of google-deepmind/
   mujoco_mpc, Apache-2.0).
   https://github.com/gravesreid/mujoco_mpc_bittle
   *Used as:* our starting robot BODY model (src/sim/bittle_model/bittle_body.xml).
   Our scene, servos, sensors, reward, terrain, and training are our own.

## Tools / frameworks

4. **Stable-Baselines3** — Raffin et al. (2021), JMLR. Reliable PPO implementation.
   https://jmlr.org/papers/v22/20-1364.html
   *Used as:* our RL algorithm library (PPO).

5. **MuJoCo** — Todorov, Erez, Tassa (2012), IROS. Physics simulator.
   (Todorov et al., "MuJoCo: A physics engine for model-based control".)
   *Used as:* our simulation engine (contacts, heightfield terrain).

6. **Gymnasium** — Farama Foundation. RL environment API.
   https://gymnasium.farama.org/
   *Used as:* the env interface our BittleEnv implements.

## Method references (to add as we read them)

- PPO — Schulman et al. (2017), "Proximal Policy Optimization Algorithms".
  https://arxiv.org/abs/1707.06347
  *Why:* the algorithm we use; needed to explain clipping to a judge.
- Terrain curriculum / rough-terrain RL locomotion — (ADD the specific ANYmal/
  Unitree paper we actually read before citing; do not cite unread.)
- Domain randomization — Tobin et al. (2017), "Domain Randomization for
  Transferring Deep Neural Networks from Simulation to the Real World", IROS
  2017 / Peng et al. (2018), "Sim-to-Real Transfer of Robotic Control with
  Dynamics Randomization", ICRA 2018. https://arxiv.org/abs/1710.06537
  *Why:* justifies our domain-randomization ablation factors — Peng et al.
  ran exactly this kind of one-factor-at-a-time ablation (latency,
  obs-noise, mass, friction) and found latency/noise mattered far more than
  mass/friction for their robot. Add once read.

## Found via competition-strategy research (2026-08-13) — READ BEFORE CITING

These turned up while researching whether commercial hardware + novel
software is a legitimate ISEF pattern (it is — see EXPERIMENT_LOG.md). Real,
verified-relevant papers, but found via a research pass, not yet read
firsthand by the student. Do not cite until actually read, per this file's
own rule above.

- **HIGH PRIORITY — this is the actual citation for our "not first to do
  sim-to-real on Bittle" novelty claim (CLAUDE.md §3):** Neuman et al.,
  "Closing the Sim-to-Real Gap for Ultra-Low-Cost, Resource-Constrained
  Quadruped Robot Platforms", RSS 2022 Sim2Real Workshop.
  https://a2r-lab.org/publication/bittlesim2real/
  *Why it matters:* done on this exact robot (8-DoF, ~$300 Bittle);
  explicitly frames "ultra-low-cost, resource-constrained" platforms as a
  distinct research regime from lab-grade quadrupeds — this is the paper our
  novelty claim needs to point at, not just assert.
- Tan et al., "Sim-to-Real: Learning Agile Locomotion for Quadruped Robots",
  RSS 2018. https://www.roboticsproceedings.org/rss14/p10.pdf
  *Why:* closest prior "which mismatch factor matters most" study on a
  quadruped (Minitaur) — found actuator/latency modeling was the critical
  factor, not mass/friction.
- Boney et al., "RealAnt: An Open-Source Low-Cost Quadruped for Education
  and Research in Real-World Reinforcement Learning".
  https://arxiv.org/abs/2011.03085
  *Why:* the clearest published argument that cheap/reproducible hardware is
  itself a legitimate research contribution, not just a budget compromise —
  directly backs our "why does cheap hardware matter" framing.

---

NOTE: Only cite sources we have actually read/used. Placeholders above marked
"ADD" must be filled with a real, read source before the final bibliography.
