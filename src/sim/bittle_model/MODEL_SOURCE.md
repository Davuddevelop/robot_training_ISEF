# Bittle model — source & attribution

This folder contains the MuJoCo model of the Petoi Bittle used for simulation.

## Files

| File | What it is | Whose work |
|---|---|---|
| `bittle_body.xml` | The robot body: links, 8 leg joints, inertias, foot contacts, meshes. | Community model (see below), one path edit by us. |
| `assets/*.obj` | The 3D meshes the body references. | Community model. |
| `scene.xml` | Floor, light, **8 servo actuators**, **IMU + joint sensors**, home pose. | **Ours.** |

## Source of the body model

`bittle_body.xml` and the meshes come from the community repository
**[gravesreid/mujoco_mpc_bittle](https://github.com/gravesreid/mujoco_mpc_bittle)**
(file `mjpc/tasks/bittle/bittle_scaled.xml` and its `assets/`), which is a fork of
**[google-deepmind/mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc)**
(licensed Apache-2.0). That model was itself built from a Bittle URDF.

We made **one change** to the body file: `meshdir="assets"` → `meshdir=""`, so the
mesh paths resolve relative to this folder. Everything in `scene.xml` (actuators,
sensors, floor, keyframe) is ours.

## How to describe this honestly (for the science fair)

> "I used a community-made body model of the Bittle (geometry and joint layout
> derived from the manufacturer's URDF) so my simulated robot matches the real
> hardware. The control interface — the servo actuators, the IMU and joint
> sensors, the reward, and the whole learning environment — is my own work."

This matches your brief: *"I did not build the hardware... my contribution is the
learned control and the transfer method."* Using a community body model is the
sanctioned plan in CLAUDE.md §5 ("start from a community Bittle model, refine it").

## Verified (in MuJoCo 3.9.0)

- Loads cleanly: 15 position coords, 14 velocity coords, **8 actuators**.
- Sensors present: orientation (quat), angular velocity (gyro), torso pos/vel,
  8 joint angles.
- Stands stably at the home pose (settles to ~0.094 m torso height) when the
  servos hold the neutral angle.
