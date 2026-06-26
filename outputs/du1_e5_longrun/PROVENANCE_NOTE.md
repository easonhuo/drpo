# Provenance note

The formal scientific run is bound to `22c5823d66169eb90c256de342e27c5391e464c3`. The historical `run_categorical.py` and raw categorical artifacts were not committed, so this experiment is explicitly a protocol reconstruction from the locked handoff rather than an exact legacy-code replay.

Two earlier attempts were terminated by the execution tool's external call-duration limit after 66/120 runs. Their verified failed-run packages are retained. The third attempt started from a new empty output root, used the same frozen protocol and commit, completed 120/120, exited with code 0, and produced the verified raw-complete artifact.
