# DRPO C-U1 E1-E4 一键复现

## 运行

```bash
python src/drpo/drpo_cu1_e1_e4_oneclick.py
```

不需要编辑文件，不需要设置命令行参数。代码自动选择 CUDA；没有 CUDA 时也能运行，但正式 20-seed 全量实验在 CPU 上会明显更慢。

## 依赖

- Python 3.10+
- PyTorch 2.x
- NumPy
- Matplotlib（缺失时只跳过绘图，不影响训练和 CSV/JSON）

## 输出

脚本同目录生成：

```text
drpo_cu1_reproduction_results/
```

其中包括：

- `manifest.json`：冻结配置、环境、脚本哈希；
- `environment_audit.json`：几何不变量；
- `positive_checkpoints/`：公共 positive-only 初始化；
- `e1/`、`e2/`、`e3/`、`e4/`：逐 seed 和逐步轨迹；
- `variance_robustness/`：可学习方差边界稳健性；
- `reference_regression.json`：与既有结果区间的自动对照；
- `RUN_COMPLETE.json`：完成状态。

## 中断恢复

直接再次执行同一命令。已完成的 seed 会被跳过。若脚本协议版本不兼容，旧目录会被自动改名归档，不会与新结果混写。

## 术语

训练状态和测试状态都来自 `N(0,I_6)`。这些结果是同分布 held-out-context generalization，不是严格 OOD generalization。

## 当前验证状态

- Python 语法检查通过；
- CPU 开发 smoke 全流程通过；
- 正式 4096/4096、20-seed 全量重跑尚未执行，需用户审阅代码后启动。
