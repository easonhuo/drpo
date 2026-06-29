# Terminal-state and collapse audit rules

> Generated Stage 4 minimal-context shadow module. Do not edit manually.
> `docs/handoff.md` and `experiments/registry.yaml` remain authoritative.

- Module ID: `terminal_audit`
- Responsibility: Define convergence, persistent drift, task-performance collapse, support or variance boundaries, and numerical failure reporting.
- Source hash: `510d90335e9f056948dcc48103c06565b247321be8f2e79b34655f60c127b283`

## Source 1: docs/handoff.md: ## 4.1 动力学实验统一收敛标准 -> # 5. 当前真实完成状态

## 4.1 动力学实验统一收敛标准

不能用固定的 500/1000/10000 步替代收敛判断。所有 E2/E3/E4/E6/E7 需要：

1. 预先定义最大训练步数；
2. 连续多个评估窗口中，核心状态量斜率低于阈值；
3. 更新向量/梯度场残差足够小，或明确持续 runaway；
4. 将训练步数延长至少 2 倍，状态分类、主要结论和方法排序不反转；
5. 检查是否由 clamp、temperature floor 或数值溢出造成假平台。

---
