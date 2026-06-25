# DRPO C-U1 E1-E4 分阶段复现

## 正式运行顺序

E3 与 E4 是两个独立的正式交付边界。先运行、审计、打包并交付 E3，之后才能启动 E4。

```bash
python src/drpo/drpo_cu1_e1_e4_oneclick.py \
  --stage e3 \
  --output-root outputs/cu1_e3_adam

python src/drpo/drpo_cu1_e1_e4_oneclick.py \
  --stage e4 \
  --output-root outputs/cu1_e4_adam
```

E1/E2 如需独立复核：

```bash
python src/drpo/drpo_cu1_e1_e4_oneclick.py \
  --stage e1_e2 \
  --output-root outputs/cu1_e1_e2
```

`--stage all` 只允许在 `DRPO_CU1_SMOKE=1` 时做集成 smoke，不能用于正式运行。

## 优化器与初始化

- E2、E3、E4 的论文主训练统一使用 Adam；
- E3/E4 从同 seed 的 2000-step positive-only Adam checkpoint 开始；
- E2 后续 LBFGS、2× continuation 与 adaptive polish 只用于 E2 终态审计，不再改变 E3/E4 初始化；
- 输出同时记录 raw gradient norm 与 Adam 实际 parameter-update norm，二者不得混称；
- learnable-variance 主实验不使用 variance clamp；理论主事件为 support/variance contraction；正向 log-sigma 越界只记录为 unexpected event，不作为第二种科学分支。

## 依赖

- Python 3.10+
- PyTorch 2.x
- NumPy
- Matplotlib（缺失时只跳过绘图，不影响训练和 CSV/JSON）

## 输出

`--output-root` 中包括：

- `manifest.json`：冻结配置、stage、优化器、环境与脚本哈希；
- `environment_audit.json`：几何不变量；
- `positive_checkpoints/`：E2 审计 checkpoint 与 E3/E4 的 2000-step Adam 初始化 checkpoint；
- 对应 stage 的逐 seed JSON、trajectory CSV、aggregate CSV；
- `reference_regression.json`：机制方向与输出完整性检查，不预设 Distance、Global 或其他方法排名；
- `RUN_COMPLETE.json`：当前 stage 的计算完成标记。

正式完成仍须经过 terminal audit、守护器打包、校验和交付；进程退出或 `RUN_COMPLETE.json` 单独存在不等于正式结果完成。

## 中断恢复

对同一个 `--output-root` 再次运行同一 stage。脚本只跳过已同时存在 summary 与 trajectory 的 seed。协议、脚本哈希或 stage 不兼容时，旧目录会被改名归档，不与新结果混写。

## 术语

训练状态和测试状态都来自 `N(0,I_6)`。这些结果是同分布 held-out-context generalization / 未见状态泛化，不是 OOD generalization。

## 当前验证边界

本版本只完成代码、文档、registry 与 smoke 测试更新。Adam 版正式 E3/E4 尚未运行。
