# PAPER-CODE-REFERENCE-01 Size and Completeness Policy

**Status:** authoritative correction for this task  
**Branch:** `dev/paper-code-reference-01`  
**Base:** `4544005bd7df69c53bad70a9dcac846af01285e4`

## Correction

Earlier conversational estimates of a 6,000-line target, a 7,500-line review threshold, or a 9,000-line failure threshold are withdrawn. They were not derived from a completed source inventory and must not be used as implementation or acceptance gates.

The final line count is an observed consequence of the complete paper-reproduction implementation. It is not a target to optimize.

## Governing order

The implementation priority is:

1. preserve the exact scientific protocol used by the manuscript;
2. preserve all functionality required to regenerate the registered results;
3. pass legacy-to-reference function, gradient, update, trajectory, terminal-audit, and full-result reproduction gates;
4. remove only code that is demonstrably outside the paper-reproduction path;
5. share genuinely common logic when the shared implementation remains readable and behaviorally equivalent;
6. report the resulting line count after implementation and validation.

No code may be deleted, merged, shortened, or generalized merely to meet a size estimate.

## Required functionality

The paper code must retain every component needed by the included claims, including where applicable:

- exact data construction or dataset loading and validation;
- frozen configuration, seeds, optimizer settings, budgets, and checkpoint rules;
- model and policy definitions;
- critic or advantage preparation;
- positive and negative objectives;
- remoteness coordinates, controls, and gradient-budget matching;
- training and resume behavior required for reproduction;
- selected-checkpoint and terminal evaluation;
- task-performance, support or variance/probability-boundary, and NaN/Inf separation;
- terminal-state audit;
- aggregation and paper-facing result/figure-data generation;
- dependency and clean-checkout reproduction commands.

A shorter implementation that omits any required item fails. A longer implementation that is necessary for correctness and reproducibility is acceptable.

## Permitted simplification

Code may be excluded only when the source-to-target inventory proves that it is not required for any included paper claim or reproduction command. Typical exclusions include governance machinery, historical superseded protocols, internal registration transactions, packaging/upload infrastructure, and unrelated development experiments.

Common components may be extracted only after differential tests demonstrate equivalence. Abstraction is not accepted merely because it reduces line count.

## Size reporting

Line counts are descriptive diagnostics only. At each phase, the review records:

- production code lines by module;
- test code lines;
- duplicated scientific formulas or training logic found;
- exclusions and their paper-scope justification;
- functionality and correctness gates completed.

There is no maximum line-count acceptance threshold for this task. Acceptance is determined by scope completeness, readability, absence of unjustified duplication, and successful reproduction.
