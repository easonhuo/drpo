# E8 V2 Linear / Quadratic / Exp 大规模调参 Pilot

**Experiment ID:** `EXT-C-E8-ORACLE-OFFLINE-V2-TAPER-SWEEP-0.5B-01`  
**证据等级:** pilot  
**职责:** Countdown Transformer external validity；不替代 C-U1 / D-U1 受控机制识别。  

## 1. 背景

V2 model-independent bank 的 paired Global 小权重扫描首次显示：`x1/32` 在 early-best checkpoint 上相对 Positive-only 产生明显收益，但固定 Global 压力未将收益保持到 terminal。该结果支持“负优势可利用”，同时提示不区分距离的持续负压力仍可能累积伤害。

本轮停止继续调 Global，只调论文当前 active families：

- Linear: `w(u)=1/(1+lambda u)`
- Quadratic: `w(u)=1/(1+lambda u^2)`
- Exp: `w(u)=exp(-lambda u)`

SBRC、Hybrid、squared-distance exponential 不进入本轮。

## 2. 冻结对象

- 模型：Qwen2.5-0.5B-Instruct pretrained base + fresh LoRA。
- 数据：冻结 V2 offline bank、validation、test。
- 负样本选择：每个 optimizer microbatch 用 current policy 在 16-slot bank 中重选 current near/far。
- near/far mix：0.5 / 0.5。
- 训练与评估协议：读取 `configs/countdown_e8_base_rl_replay_0p5b.yaml`，不修改其中 1200-step budget、min steps、patience、optimizer、learning rate、batch、evaluation cadence 或 pass@k。
- 证据状态：pilot；fixed horizon 或 early stop 不称为 convergence。

## 3. 距离与公平性

定义 sequence-level remoteness：

`u = sqrt(relu(surprisal - tau) / scale)`。

`tau` 与 `scale` 只用冻结 calibration rows 和 base initialization 计算；不读取 validation/test task metric。

每个 family 用 `rho=w(1)` 参数化：

- reciprocal families: `lambda=rho^{-1}-1`
- Exp: `lambda=-log(rho)`

每个 `(family,rho)` 额外校准一个全局 `negative_scale`，使其 initialization aggregate negative-gradient RMS 匹配已完成 Global `x1/32` 的同一预算。这样网格主要改变 near/far 距离形状，而不是同时改变初始总负梯度预算。

## 4. 网格与资源

- methods: Linear, Quadratic, Exp
- rho: `0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.125, 0.03125`
- paired seed offsets: `0, 1000, 2000`
- total cells: `3 x 8 x 3 = 72`
- GPUs: exactly 8; one process per GPU; 9 waves at most
- Global cells: 0

这三组 seeds 是 tuning seeds，不是未来 formal confirmatory seeds。任何最终方法排名必须冻结超参后换全新 seeds。

## 5. 报告

每 cell 同时保存：

- validation-selected best checkpoint；
- terminal 或 last-finite checkpoint；
- best/terminal greedy、pass@8、pass@64、valid rate；
- current near/far surprisal；
- current near/far taper weight；
- numerical failure；
- code/config/data/calibration hashes。

Best 与 terminal post-hoc evaluation 使用同一 generation seed，避免 checkpoint 比较混入不同采样 seed。

任务性能退化、valid/support-structure proxy 和 NaN/Inf 必须分开报告。当前 valid rate 只是外部结构 proxy，不升级为正式 support-boundary audit。

## 6. 一键命令

从最新 main、干净工作树执行：

```bash
python scripts/run_countdown_e8_oracle_offline_v2_taper_sweep.py
```

默认使用服务器现有模型、V2 bank、base calibration 和 GPU 0--7 路径；所有路径均可通过 CLI 覆盖。Runner 支持 identity-checked resume，任何 identity 漂移 fail closed。
