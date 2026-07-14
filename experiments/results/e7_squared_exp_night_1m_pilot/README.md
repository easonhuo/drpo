# EXT-H-E7-SQUARED-EXP-NIGHT-01 compact result evidence

This directory archives the compact, reviewable evidence for the completed two-development-seed screening pilot. The full result package is identified by `PACKAGE_SHA256.txt`; checkpoints and large raw logs are intentionally not committed to the code repository.

## Provenance limitation

The run reports local launch commit `fbafb44c2e562ed93f6f63b43e8c5439b881d6a8`. That commit object was not present on GitHub at archive time. Its substantive delta from the reviewed dev implementation was the required trainer-plumbing correction from the unsupported variant label `iqlv_squared_exp_night` to the canonical supported label `iqlv_exp_rank`; the repository integration reproduces that exact source change. This limitation prevents upgrading the run to formal evidence, but the package remains usable as a development screening result.

## Interpretation boundary

- 126/126 branches completed at 1M updates; terminal audit PASS; zero NaN/Inf failures.
- The KL-refresh PPO path beat fixed-K4 PPO in 16/21 paired cells, with mean late-window difference `+3.79`; this is a positive finite-horizon mean-performance signal.
- Mean late seed standard deviation was not lower for KL-refresh (`7.03` versus `5.87`), so no general stability claim is allowed.
- Frequent KL-triggered refreshes produced an average realized reference lifespan of about four updates; the result supports adaptive refresh timing, not a claim that a fixed longer K is superior.
- Stage C GAE started zero branches and remains blocked pending a verified trajectory/terminal/truncation contract.
- Hopper/Walker are external-validity tasks and do not replace C-U1 or D-U1 controlled mechanism evidence.
