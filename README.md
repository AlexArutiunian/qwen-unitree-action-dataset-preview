# Real-robot deployment

This branch is reserved for the real Unitree G1 deployment layer of the Voice-to-JSON pipeline. The static dataset preview inherited from `main` is intentionally retained.

The first recovered component is [`deploy/voice-server`](deploy/voice-server): a small speech-recognition server copied from `/home/arutiunyan_ag/g1_voice_best` on 2026-09-04. Its virtual environment, caches and ASR model files were not copied.

Planned additions include:

- robot-side SDK transport and lifecycle handling;
- JSON schema and joint-limit validation;
- IK feasibility checks and rejection/refinement routing;
- timing and rapid-arm-motion safety gates;
- simulation/real configuration separation;
- execution logging and visual post-action verification.

The recovered voice service should be reviewed and normalized before production use. In particular, deployment addresses and model paths currently come from its historical server configuration.

Only the server half existed in the recovered `g1_voice_best` directory. The robot-side client described by its historical README was not present on that server and is therefore not included in this branch.
