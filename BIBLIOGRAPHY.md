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
- Domain randomization — Tobin et al. (2017) / Peng et al. (2018).
  *Why:* justifies our domain-randomization ablation factors. Add once read.

---

NOTE: Only cite sources we have actually read/used. Placeholders above marked
"ADD" must be filled with a real, read source before the final bibliography.
