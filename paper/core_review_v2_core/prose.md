# Evidence-bounded prose

## EXP-P04-A

We test the transmission path with four matched interventions over 20 paired held-out seeds. The uncontrolled Baseline and Near-zero variants finish at rewards 2.22e-06 and 2.18e-06 and undergo task-performance collapse in 20/20 and 20/20 seeds. Removing the far-field contribution instead yields 0.739 [0.739, 0.740] for Far-zero, while capping it yields 0.733 [0.733, 0.734] for Far-cap; neither intervention collapses in any seed. The registered fixed-variance budget controls also remain non-collapsed: Global-scale reaches 0.599, and transferring the far budget to the near component reaches 0.875. These controls are diagnostic and do not define a method ranking. Near-field removal is therefore not a rescue, whereas deleting or bounding the far-field path is. Within this controlled same-distribution held-out-context setting, the comparison identifies the far-field component as the dominant causal transmission path; it does not establish a universal method ranking.

## EXP-P04-B

The learnable-variance branch separates the type of failure. Baseline and Near-zero reach the registered support/variance-contraction boundary in 20/20 and 20/20 seeds, with mean onsets at steps 72.9 and 73.1. Far-zero and Far-cap record 0/20 and 0/20 support-boundary events. All four methods keep finite parameters and record 0/20 NaN/Inf failures. Thus this branch is evidence for support contraction rather than numerical collapse, and the intervention again isolates far-field negative influence as the removable path in C-U1.
