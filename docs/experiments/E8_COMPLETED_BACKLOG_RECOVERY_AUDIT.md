# E8 completed-backlog recovery audit

Provenance recovery record only; not a competing research master or scientific-status upgrade.

## Recovery diagnostics

- git-show exit: 0
- base64 exit: 1
- gzip exit: 1
- recovered bytes: 36894
- recovered SHA-256: 113ef41cb9564ae829ddc32af320a90f8e43718e49995b28b0be46219e9d83a3

```text
base64: invalid input

gzip: /tmp/e8_patch.gz: unexpected end of file
```

## File inventory

diff --git a/docs/handoff.md b/docs/handoff.md
diff --git a/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/FULL_ACCEPTANCE_REPORT.json b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/FULL_ACCEPTANCE_REPORT.json
diff --git a/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml
diff --git a/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/MATERIALIZATION_REPORT.json b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/MATERIALIZATION_REPORT.json
diff --git a/docs/handoff_shadow/stage4/minimal/generated/MODULE_INDEX.json b/docs/handoff_shadow/stage4/minimal/generated/MODULE_INDEX.json

## Experiment and evidence references

 - **Countdown E8 base-start RL/replay 0.5B pilot：**登记 `EXT-C-E8-BASE-RL-REPLAY-0.5B-01`，作为移除 Countdown SFT warmstart 后的基模起点诊断。该实验只回答：Qwen pretrained base 是否能通过 oracle-offline fixed positive corpus 学起；base-specific calibrated offline negatives 是否能超过 positive-only；online on-policy self-sampled positives 是否能冷启动；dynamic replay buffer 累积历史自采 positives/negatives 是否优于 immediate on-policy 更新。所有 RL 分支从 Qwen pretrained base + fresh LoRA 开始，禁止 Countdown SFT warmstart、随机初始化主实验、taper 方法族和正式方法排名声明。固定预算 pilot 只报告有限步 evidence；结果必须分开报告 task performance、online signal sparsity/replay support、valid structure boundary 和 NaN/Inf numerical failure。
+- **Countdown V2 continuous-EXP 四轮 development-pilot 结果闭环（`EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01`、`EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01`、`EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01`、`EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01`）：**四个 validation-only 包共 `158/158` cells 完成，全部跑到固定 `1200` steps，terminal audit 均为 `PASS`，test split 未访问，NaN/Inf 为 `0/158`。任务性能、valid-rate structure proxy 与 NaN/Inf 继续分开；valid rate 不是正式 support-boundary audit，固定 horizon 也不是 convergence/steady state。两-seed 31-point grid 在 `alpha=0.5,c=1` 显示非单调剂量响应和描述性高点，但存在两-seed 选优偏差；后续 paired seed blocks 表明效果具有强 trajectory/configuration interaction。固定 `alpha=1` 的重复证据最集中于 `c≈3--4`，更宽松地说 `c≈2.25--6`；`c=8` 的单批高点未在下一 seed block 稳定复现，`c=16--128` 未出现持续第二高值区。该证据不证明 `alpha` 可由 `c` 替代，不授权方法排名、通用最优参数、confirmatory claim、OOD generalization 或 controlled causal identification。四轮共享的 evaluator 会在 validation 时重置 Python/NumPy/Torch/CUDA RNG 且不恢复训练 RNG；该缺陷不能单独解释最新一轮下降，但会限制 seed-independence 与精确复现，后续 `EXT-C-E8-ORACLE-OFFLINE-V2-REPRO-RNG-AUDIT-0.5B-01` 必须按独立协议处理。完整包哈希、运行 commit、逐参数 late/terminal 指标与包装缺口记录在 `experiments/results/e8_continuous_exp_four_pilot_closure_20260715/` 和 `docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md`。
 - **Countdown E8 low-SFT / capacity diagnostic dirty pilots：**本线记录 `EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01` 与一次性 `EXT-C-E8-LOWSFT-RFT-0.5B-01` 试错结果，只能作为 single-seed pilot evidence，不得升级为正式多 seed 结论或方法排名。capacity diagnostic 的 single seed `2026070701` 显示 `same_lora_rft`、`fresh_lora_rft`、`full_param_rft` 的 `best_attempt` 均为 0；terminal 端总体表现为 greedy 持平或小升、pass@8/pass@64 下降，说明 naive verifier-correct positive-only on-policy RFT 没有超过 LoRA SFT 起点。low-SFT 试错从按 validation greedy≈0.08 选出的 epoch-3 LoRA SFT checkpoint 起跑；该 checkpoint 的 pass@8 实际已接近 full-SFT（不是 pass@8≈0.08 起点），RFT 后 `best_attempt=0`，terminal test greedy 0.100→0.113、pass@8 0.174→0.133、pass@64 0.265→0.149。解释必须保留以下限制：运行源码为 dirty pilot / one-off orchestration；不是 convergence；没有证明 3B 或更强模型无效；尚需 no-update、parameter-delta、probe-loss、Qwen pretrained-base no-SFT、ultra-low pass@8 checkpoint 与 offline fixed-corpus controls。工程上允许把 `cmd_sft --save_every_epoch` 作为 opt-in 本地 checkpoint 功能合入，以便后续选择更细粒度 ultra-low SFT 起点；模型权重与结果包不得进入 Git 更新包。
diff --git a/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/FULL_ACCEPTANCE_REPORT.json b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/FULL_ACCEPTANCE_REPORT.json
+++ b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/FULL_ACCEPTANCE_REPORT.json
+      "EXT-C-E8-BASE-RL-REPLAY-0.5B-01-2026-07-09",
+      "EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15",
+      "EXT-C-E8-FULLBANK-GRADIENT-PILOT-V75-2026-07-04",
+      "EXT-C-E8-LOWSFT-RFT-0.5B-01-RESULTS-2026-07-08",
+      "EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01-2026-07-07",
+      "EXT-C-E8-ONPOLICY-UNPOLISHED-0.5B-01-2026-07-07",
+      "EXT-C-E8-ORACLE-OFFLINE-BANK-V2-0.5B-01-2026-07-09",
+      "EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-PILOT-CLOSURE-2026-07-11",
+      "EXT-C-E8-TAPER-0.5B-ACTIVE-TAIL-REPAIR-V79-2026-07-06",
+      "EXT-C-E8-TAPER-0.5B-CORRECTED-V73-2026-07-03",
+      "EXT-C-E8-TAPER-0.5B-DIAGNOSTIC-BUGFIX-2026-07-05",
+      "EXT-C-E8-TAPER-0.5B-REGISTRATION-2026-07-01",
+      "EXT-C-E8-V2-GLOBAL-MILESTONE-AND-TAPER-SWEEP-2026-07-11",
+      "EXT-C-E8-V4.3-DYNAMIC-CONTROL-2026-06-27",
+      "EXT-C-E8-V4.4-OFFLINE-BANK-2026-06-28",
+      "EXT-C-E8-V4.5-OFFLINE-BANK-TUNING-2026-06-28",
+      "EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY-2026-06-29",
+    "EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15"
diff --git a/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml
+++ b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml
+update_id: EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15
+  content: '- **Countdown V2 continuous-EXP 四轮 development-pilot 结果闭环（`EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01`、`EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01`、`EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01`、`EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01`）：**四个
+    时重置 Python/NumPy/Torch/CUDA RNG 且不恢复训练 RNG；该缺陷不能单独解释最新一轮下降，但会限制 seed-independence 与精确复现，后续 `EXT-C-E8-ORACLE-OFFLINE-V2-REPRO-RNG-AUDIT-0.5B-01`
+    必须按独立协议处理。完整包哈希、运行 commit、逐参数 late/terminal 指标与包装缺口记录在 `experiments/results/e8_continuous_exp_four_pilot_closure_20260715/`
+    和 `docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md`。'
+    entity_id: EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01
+    - docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml
+    - docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md
+    - docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json
+    - docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt
+    - docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256
+    - experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json
+    - experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv
+    entity_id: EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01
+    entity_id: EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01
+    entity_id: EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01
diff --git a/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/MATERIALIZATION_REPORT.json b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/MATERIALIZATION_REPORT.json
+++ b/docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/MATERIALIZATION_REPORT.json
+        "entity_id": "EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01",
+            "path": "docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv",
+        "entity_id": "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01",
+            "path": "docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv",
+        "entity_id": "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01",
+            "path": "docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv",
+        "entity_id": "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01",
+            "path": "docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv",
+        "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01",
+        "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01",
+        "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01",
+        "EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01"
+        "entity_id": "EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01",
+            "path": "docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv",
+        "entity_id": "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01",
+            "path": "docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv",
+        "entity_id": "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01",
+            "path": "docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv",
+        "entity_id": "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01",
+            "path": "docs/handoff_deltas/EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15/HANDOFF_DELTA.yaml",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_2026-07-15.md",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_VALIDATION.json",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_SOURCE_PACKAGES_SHA256.txt",
+            "path": "docs/results/E8_CONTINUOUS_EXP_FOUR_PILOT_CLOSURE_CHECKSUMS.sha256",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/RESULT_CLOSURE.json",
+            "path": "experiments/results/e8_continuous_exp_four_pilot_closure_20260715/PARAMETER_SUMMARY.csv",
+        "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01",
+        "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01",
+        "EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01",
+        "EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01"
+      "docs/handoff_shadow/stage4/minimal/generated/modules/terminal_audit.md",
+  "update_id": "EXT-C-E8-CONTINUOUS-EXP-FOUR-PILOT-RESULT-CLOSURE-2026-07-15"
         "docs/handoff.md: HANDOFF-DELTA-BLOCK after_heading:v52-ext-c-e8-v43-dynamic-control",
