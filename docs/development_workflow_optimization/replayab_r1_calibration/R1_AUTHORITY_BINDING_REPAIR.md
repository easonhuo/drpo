# ReplayAB R1 Authority-Binding Repair

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Calibration: `REPLAYAB-R1-C1-CALIBRATION-01`

Base: `main@dd46727c1efefd2e6d4cdf6f3b204ec1fc58fca3`

Scientific impact: none

## Authorization and diagnosis

The user authorized this narrow repair on 2026-07-20 after formal calibration reported 10/10 behavioral verdict agreement, zero covered unsafe passes, zero covered false rejections, and a passing runtime guardrail, while blocking closure because the frozen fixtures contained an orphaned evidence-schema digest.

The incorrect digest was `5f02130bc08513540e694fc1b1abf1d8437f9fa697f92535c49a795dcd9de179`. Historical provenance diagnostic run `29719811214` found no repository object matching that digest. The implementation contract has the stable SHA-256 `ae7f23134285b5314647bd1068bc3fa1f3935deccdd24536eb8183cb57e11494`.

## Repair boundary

- Replace only the orphaned evidence-schema digest in the two R1 manifests, four R1 run artifacts, and two Candidate 01 C1 contracts.
- Recompute only the two derived R1 case-contract digests and synchronize the four run artifacts that bind them.
- Do not change the calibration inventory, expected verdicts, test behavior, thresholds, production code, Candidate 01 implementation, V1, Stage 5 authority, handoff, registry, scientific code, or default route.
- Preserve the original failed calibration and diagnostic artifacts.

## Validation gate

The repaired fixtures must pass the complete frozen R1 calibration module, focused ReplayAB tests, full repository PR gates, and governance checks before merge. Ruff is required for changed Python files; this repair changes no Python file. Passing authorizes R1 closure review only and does not start R2.
