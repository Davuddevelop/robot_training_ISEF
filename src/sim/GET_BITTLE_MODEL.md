# The Bittle MuJoCo Model — already in the repo ✅

**You do not need to download anything.** The Bittle model is now committed to
this project (and verified to load and stand in MuJoCo 3.9.0). After a
`git pull`, you have it.

## Where it lives

```
src/sim/bittle_model/
    scene.xml         ← OUR scene: floor + 8 servos + IMU/joint sensors + home pose
    bittle_body.xml   ← the community body model (links, joints, inertias)
    assets/*.obj      ← the 3D meshes the body uses
    MODEL_SOURCE.md   ← where the body came from + how to describe it honestly
```

See `bittle_model/MODEL_SOURCE.md` for the source and attribution.

## See it in 3D (do this first)

```
D:\robot_venv\Scripts\python.exe notebooks\10_view_bittle.py
```

A 3D window opens showing the Bittle standing on a checkered floor — four legs
pointing **down** under the body (the dog shape, not the spider Ant). Drag with
the mouse to orbit and inspect it.

## What a MuJoCo XML model is

It is a text file describing the robot in full detail:
- Every body part with its mass and size, and the meshes that draw it
- Every joint connecting them, with angle limits
- Every actuator (servo) and its torque limit
- The ground plane, light, gravity, and sensors

Open `scene.xml` in your editor — it reads like HTML with robot tags. You can
already explain every line of it (we wrote the comments for exactly that).

## What's next

We wire `scene.xml` into `bittle_env.py` (currently stubbed) to make a Gymnasium
environment, then train it with the same PPO + VecNormalize pipeline we proved on
Ant. We will build `bittle_env.py` together, in small pieces — see CLAUDE.md §2.
