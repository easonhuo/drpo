# E8 Multitask Reciprocal Matrix 01 — Cold-Start Binding

Experiment ID: `EXT-C-E8-MULTITASK-RECIPROCAL-MATRIX-01`

RunSpec ID: `E8_MULTITASK_RECIPROCAL_MATRIX_20260914_01`

Status: `not_run`

Frozen config commit: `a8fac23b713e4060273a067a5072631264834742`

Target branch: `dev/e8-multitask-exp-coldstart-01`

Config: `configs/e8_multitask_reciprocal_matrix_formal.yaml`

Entrypoint: `bash scripts/bootstrap_e8_multitask_exp_coldstart.sh full`

Run class: `formal`

The execution snapshot must set `E8_COLDSTART_TARGET_REF=refs/heads/dev/e8-multitask-exp-coldstart-01`, `E8_COLDSTART_EXPERIMENT_ID=EXT-C-E8-MULTITASK-RECIPROCAL-MATRIX-01`, `E8_COLDSTART_CONFIG=configs/e8_multitask_reciprocal_matrix_formal.yaml`, and `E8_COLDSTART_RUN_ID=E8_MULTITASK_RECIPROCAL_MATRIX_20260914_01`.

The RunSpec uses deferred registration with `closure_required: true`. Its purpose is to keep one immutable run identity across execution, recovery, packaging, and `drpo-results` delivery; it does not change the scientific protocol frozen in the config.

No training has been launched by this binding.
