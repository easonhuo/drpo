#!/usr/bin/env python3
"""Build DRPO-specific figures, tables, and proof assets.

This module is a project plugin. Generic manuscript orchestration lives in
``scripts/manuscript_release_pipeline.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DETERMINISTIC_PDF_METADATA = {
    "Creator": "drpo manuscript pipeline",
    "Producer": "matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def save_e3(root: Path, fig_dir: Path, gen_dir: Path) -> None:
    rows = read_csv(root / "outputs/cu1_e3_adam/learnable_variance_aggregate.csv")
    wanted = ["baseline", "near_zero", "far_zero", "far_cap", "global_scale", "positive_only"]
    by = {r["method"]: r for r in rows}
    selected = [by[m] for m in wanted if m in by]
    labels = [r["method"].replace("_", "-") for r in selected]
    rewards = [f(r, "reward") for r in selected]
    lows = [f(r, "reward_ci_low") for r in selected]
    highs = [f(r, "reward_ci_high") for r in selected]
    task = [f(r, "task_failure_onset_event_rate", 0.0) for r in selected]
    support = [f(r, "support_boundary_onset_event_rate", 0.0) for r in selected]
    x = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    axes[0].bar(
        x,
        rewards,
        yerr=[np.array(rewards) - np.array(lows), np.array(highs) - np.array(rewards)],
        capsize=3,
    )
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("Held-out-context reward")
    axes[0].set_title("C-U1 targeted interventions")
    axes[0].grid(axis="y", alpha=0.25)
    width = 0.36
    axes[1].bar(x - width / 2, task, width, label="task collapse")
    axes[1].bar(x + width / 2, support, width, label="support/variance boundary")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Event rate")
    axes[1].set_title("Failure taxonomy")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = fig_dir / "fig3_cu1_causal.pdf"
    fig.savefig(out, bbox_inches="tight", metadata=DETERMINISTIC_PDF_METADATA)
    plt.close(fig)
    lines = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=0.93\textwidth]{figures/generated/fig3_cu1_causal.pdf}",
        r"\caption{C-U1 causal intervention with learnable variance. The left panel reports registered held-out-context reward with confidence intervals; the right panel separates task-performance collapse from support or variance-boundary events. Near-zero tracks the uncontrolled baseline, while Far-zero and Far-cap remove the task-collapse pathway without introducing NaN/Inf failures.}",
        r"\label{fig:cu1-causal}",
        r"\end{figure*}",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{C-U1 intervention summary (20 seeds).}",
        r"\label{tab:cu1-causal}",
        r"\begin{tabular}{lccc}",
        r"\toprule Method & Reward & Task event & Boundary event \\",
        r"\midrule",
    ]
    for r in selected:
        lines.append(
            f"{r['method'].replace('_', '-')} & {f(r, 'reward'):.3f} & {f(r, 'task_failure_onset_event_rate', 0):.2f} & {f(r, 'support_boundary_onset_event_rate', 0):.2f}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (gen_dir / "exp_p04_assets.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_e4(root: Path, fig_dir: Path, gen_dir: Path) -> None:
    rows = read_csv(root / "outputs/cu1_e4_adam/learnable_variance_aggregate.csv")
    rows = sorted(rows, key=lambda r: f(r, "alpha"))
    a = np.array([f(r, "alpha") for r in rows])
    reward = np.array([f(r, "reward") for r in rows])
    disp = np.array([f(r, "normalized_extrapolation_displacement") for r in rows])
    boundary = np.array([f(r, "support_boundary_onset_event_rate", 0) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    axes[0].plot(a, reward, marker="o", label="reward")
    axes[0].plot(a, disp, marker="s", label="normalized extrapolation")
    axes[0].set_xlabel("Negative coefficient $\\alpha$")
    axes[0].set_title("Stable extrapolation to degradation")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(a, boundary, marker="o")
    axes[1].set_xlabel("Negative coefficient $\\alpha$")
    axes[1].set_ylabel("Boundary-event rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Support/variance boundary")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "fig4_cu1_phase.pdf",
        bbox_inches="tight",
        metadata=DETERMINISTIC_PDF_METADATA,
    )
    plt.close(fig)
    taper = json.loads(
        (root / "outputs/cu1_e4_taper_near_retention/RESULT_SUMMARY.json").read_text()
    )
    p = taper["primary_paired_results_at_near_retention_0_75"]
    methods = ["reciprocal_quadratic", "current_exponential", "squared_distance_exponential"]
    deltas = [p[m]["mean_held_out_context_reward_delta"] for m in methods]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.bar(np.arange(3), deltas)
    ax.set_xticks(
        np.arange(3), ["Recip.-quad.", "Exp.", "Squared-dist. exp."], rotation=20, ha="right"
    )
    ax.set_ylabel("Paired reward delta vs. recip.-linear")
    ax.set_title("Matched near-retention comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "fig5_taper_shape.pdf",
        bbox_inches="tight",
        metadata=DETERMINISTIC_PDF_METADATA,
    )
    plt.close(fig)
    text = r"""\begin{figure*}[t]
\centering
\begin{subfigure}{0.62\textwidth}
\includegraphics[width=\linewidth]{figures/generated/fig4_cu1_phase.pdf}
\caption{Strength sweep and boundary events.}
\end{subfigure}\hfill
\begin{subfigure}{0.34\textwidth}
\includegraphics[width=\linewidth]{figures/generated/fig5_taper_shape.pdf}
\caption{Matched near-retention taper comparison.}
\end{subfigure}
\caption{C-U1 phase and control evidence. The strength sweep maps the transition from the Positive-only reference through stable extrapolation toward boundary events. Under the frozen near-retention protocol, all three faster-decaying tapers improve finite-horizon held-out-context reward on 20/20 paired seeds relative to reciprocal-linear; this result remains finite-step validated rather than a universal steady-state ranking.}
\label{fig:cu1-phase-control}
\end{figure*}
"""
    (gen_dir / "exp_p05_assets.tex").write_text(text, encoding="utf-8")


def save_du1(root: Path, fig_dir: Path, gen_dir: Path) -> None:
    for name in ("causal_terminal_reward.png", "causal_support_boundary.png"):
        shutil.copy2(root / "outputs/du1_e5_longrun" / name, fig_dir / name)
    agg = json.loads((root / "outputs/du1_e5_longrun/aggregate_summary.json").read_text())[
        "methods"
    ]
    text = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\begin{subfigure}{0.48\textwidth}\includegraphics[width=\linewidth]{figures/generated/causal_terminal_reward.png}\caption{Terminal reward.}\end{subfigure}\hfill",
        r"\begin{subfigure}{0.48\textwidth}\includegraphics[width=\linewidth]{figures/generated/causal_support_boundary.png}\caption{Support-boundary events.}\end{subfigure}",
        r"\caption{D-U1 long-run causal results. Baseline and Near-zero preserve the rare/far negative pathway and reach the support boundary in 20/20 seeds; Far-zero and Far-cap remain bounded in 20/20 seeds. Global scaling can preserve task reward while still reaching the support boundary, illustrating why task, boundary, and numerical events must be reported separately.}",
        r"\label{fig:du1-causal}",
        r"\end{figure*}",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{D-U1 E5 long-run summary.}",
        r"\label{tab:du1-e5}",
        r"\begin{tabular}{lccc}",
        r"\toprule Method & Reward & Task collapse & Boundary \\",
        r"\midrule",
    ]
    for m in ["baseline", "near_zero", "far_zero", "far_cap", "global_scale", "positive_only"]:
        if m in agg:
            v = agg[m]
            text.append(
                f"{m.replace('_', '-')} & {v['terminal_reward_mean']:.3f} & {v['task_collapse_count']}/20 & {v['support_collapse_count']}/20"
                + r" \\"
            )
    text += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (gen_dir / "exp_p03_assets.tex").write_text("\n".join(text) + "\n", encoding="utf-8")


def save_proofs(gen_dir: Path) -> None:
    (gen_dir / "app_theorem_proof.tex").write_text(
        r"""
\subsection{Proof of Proposition~\ref{prop:score-remoteness}}
\label{app:proof-score-remoteness}
\begin{proof}
Let $\delta=a-\mu$. For a Gaussian with fixed covariance,
\begin{equation}
\begin{aligned}
D_\mu(a)-C_\Sigma&=\tfrac12\delta^\top\Sigma^{-1}\delta,\\
R_\mu(a)&=\|\Sigma^{-1}\delta\|_2^2
=\delta^\top\Sigma^{-2}\delta.
\end{aligned}
\end{equation}
With $x=\Sigma^{-1/2}\delta$, these become
$D_\mu-C_\Sigma=\|x\|_2^2/2$ and $R_\mu=x^\top\Sigma^{-1}x$.
The Rayleigh quotient therefore gives
\begin{equation}
\begin{aligned}
R_\mu(a)&\ge2\lambda_{\min}(\Sigma^{-1})(D_\mu-C_\Sigma),\\
R_\mu(a)&\le2\lambda_{\max}(\Sigma^{-1})(D_\mu-C_\Sigma).
\end{aligned}
\end{equation}
If $\delta=rv$ for a fixed direction, eliminating $r^2$ gives the exact directional coefficient; when $\Sigma=\sigma^2I$, it reduces to $2/\sigma^2$.
For a categorical policy, $\nabla_z\log\pi_z(a)=e_a-\pi_z$. Writing $p_a=e^{-D_z(a)}$ yields
$|\partial_{z_a}\log\pi_z(a)|=1-p_a=1-e^{-D_z(a)}$.
Moreover,
\begin{equation}
\|e_a-\pi_z\|_2^2=(1-p_a)^2+\sum_{j\ne a}\pi_z(j)^2
\le 2(1-p_a)^2\le2.
\end{equation}
Thus the selected-logit response increases with surprisal but remains bounded.
\end{proof}

\subsection{Proof of Theorem~\ref{thm:reuse}}
\label{app:proof-reuse}
\begin{proof}
Write $f(u)=D(u)$, $g_t=\nabla f(u_t)$, and $R_t=\|g_t\|_2^2$.
For a negative coefficient $\widehat A=-c$ with $c>0$, the update is $u_{t+1}=u_t+\alpha g_t$, where $\alpha=\eta c$.
Convexity gives
\begin{equation}
f(u_{t+1})\ge f(u_t)+g_t^\top(u_{t+1}-u_t)
=f(u_t)+\alpha\|g_t\|_2^2.
\end{equation}
The gradient of a differentiable convex function is monotone, hence
$(g_{t+1}-g_t)^\top(u_{t+1}-u_t)\ge0$.
Substituting $u_{t+1}-u_t=\alpha g_t$ gives
$g_{t+1}^\top g_t\ge\|g_t\|_2^2$; Cauchy--Schwarz then implies
$\|g_{t+1}\|_2\ge\|g_t\|_2$.
For a positive coefficient $\widehat A=c>0$, the update is $u_{t+1}=u_t-\alpha g_t$.
The descent lemma yields
\begin{equation}
f(u_{t+1})\le f(u_t)-\alpha(1-L\alpha/2)\|g_t\|_2^2
\le f(u_t)-\tfrac\alpha2R_t
\end{equation}
when $\alpha\le1/L$.
For a convex function with $L$-Lipschitz gradient, cocoercivity gives
$(g_t-g_{t+1})^\top(u_t-u_{t+1})\ge L^{-1}\|g_t-g_{t+1}\|_2^2$.
Using $u_t-u_{t+1}=\alpha g_t$ and expanding $\|g_{t+1}\|_2^2$ proves
$R_{t+1}\le R_t$ for $\alpha L\le1$.
\end{proof}

\subsection{Detailed derivation for Theorem~\ref{thm:equilibrium}}
\label{app:proof-theorem-equilibrium}
\begin{proof}
The signed population objective is
\begin{equation}
L(\eta)=(p\mathbf m_+-q\mathbf m_-)^\top\eta-(p-q)\psi(\eta)+C,
\end{equation}
so
$F(\eta)=\nabla L(\eta)=p\mathbf m_+-q\mathbf m_--(p-q)\nabla\psi(\eta)$ and
$\nabla^2L(\eta)=-(p-q)\nabla^2\psi(\eta)$.
For $p>q$, $F(\eta)=0$ is equivalent to
\begin{equation}
\nabla\psi(\eta)=\mathbf m^\star=\frac{p\mathbf m_+-q\mathbf m_-}{p-q}.
\end{equation}
A regular minimal exponential family has strictly convex log-partition $\psi$, so $\nabla\psi$ is one-to-one and maps the natural-parameter space onto the interior mean-parameter set. An interior $\mathbf m^\star$ therefore has a unique finite preimage $\eta^\star$. Subtracting $\mathbf m_+$ yields
\begin{equation}
\mathbf m^\star-\mathbf m_+=\frac{q}{p-q}(\mathbf m_+-\mathbf m_-).
\end{equation}
The field Jacobian is $J_F(\eta^\star)=-(p-q)\nabla^2\psi(\eta^\star)\prec0$, proving continuous-time local asymptotic stability. The discrete Jacobian is
$I-\alpha(p-q)\nabla^2\psi(\eta^\star)$, whose spectral radius is below one whenever
\begin{equation}
0<\alpha<\frac{2}{(p-q)\lambda_{\max}(\nabla^2\psi(\eta^\star))}.
\end{equation}
If $\mathbf m^\star$ approaches the feasible boundary, continuity and invertibility imply that its natural parameter leaves every compact subset; outside the feasible set no finite solution exists.
When $p=q$, the field is the constant $p(\mathbf m_+-\mathbf m_-)$. It produces linear drift if the moments differ and a non-isolated stationary continuum otherwise.
When $p<q$, $\nabla^2L=(q-p)\nabla^2\psi\succ0$; any finite stationary point has a positive-definite continuous Jacobian and a discrete Jacobian with eigenvalues strictly larger than one, so it is unstable.
\end{proof}

\subsection{Proof of Theorem~\ref{thm:family-runaway}}
\label{app:proof-family-runaway}
\begin{proof}
Let $r=q-p>0$. For the fixed-covariance Gaussian,
\begin{equation}
F_\mu(\mu)=\Sigma^{-1}[p\mu_+-q\mu_--(p-q)\mu]
=r\Sigma^{-1}(\mu-\mu^\dagger).
\end{equation}
With $\delta=\mu-\mu^\dagger$, continuous time gives
$\dot\delta=r\Sigma^{-1}\delta$ and hence
$\delta(t)=\exp(r\Sigma^{-1}t)\delta(0)$.
Every nonzero displacement grows at least as
$\exp(r\lambda_{\min}(\Sigma^{-1})t)$.
The discrete recurrence is
$\delta_{t+1}=(I+\alpha r\Sigma^{-1})\delta_t$, whose eigenvalues exceed one for every $\alpha>0$.
Thus every nonstationary trajectory diverges. For fixed $a$,
$\|\nabla_\mu\log\pi_\mu(a)\|_2=\|\Sigma^{-1}(a-\mu)\|_2\to\infty$.

For the categorical policy, fix the gauge $\mathbf1^\top z=0$ and write the signed objective as
\begin{equation}
L(z)=b^\top z+r\log\sum_j e^{z_j}+C.
\end{equation}
Its Hessian on the gauge-fixed subspace is
$r(\operatorname{Diag}(\pi_z)-\pi_z\pi_z^\top)\succ0$, so $L$ is strictly convex there.
Along continuous gradient ascent, $dL/dt=\|\nabla L\|_2^2$; along the discrete update, convexity gives
$L(z_{t+1})\ge L(z_t)+\alpha\|\nabla L(z_t)\|_2^2$.
A bounded nonstationary trajectory would therefore have an accumulation point with zero gradient. If a finite stationary point exists, strict convexity makes it unique and the squared distance from it increases under ascent; if none exists, zero-gradient accumulation is impossible. Hence every nonstationary trajectory leaves each compact set.
On the gauge-fixed subspace, probabilities bounded away from zero imply bounded logits, so an unbounded trajectory has a subsequence approaching the simplex boundary. Finally,
$\|e_a-\pi_z\|_2^2\le2(1-\pi_z(a))^2\le2$, proving bounded per-sample scores.
\end{proof}

\subsection{Proof of Proposition~\ref{prop:vanishing}}
\label{app:proof-far-field}
\begin{proof}
Assume the unweighted score satisfies $\|\nabla_\theta\log\pi_\theta(a\mid s)\|\le C(1+r_\theta)^k$ for finite $k$, and let $w_\lambda(r)=\exp(-\lambda r^2)$. Then
\begin{equation}
\|w_\lambda(r)A^-\nabla_\theta\log\pi_\theta\|
\le |A^-|C(1+r)^k e^{-\lambda r^2}.
\end{equation}
For every finite $k$ and $\lambda>0$, the Gaussian tail dominates the polynomial factor, so the right-hand side converges to zero as $r\to\infty$. This proves vanishing weighted far-field influence without assuming that the sample's utility decays exponentially.
\end{proof}
""",
        encoding="utf-8",
    )
    (gen_dir / "app_gaussian_derivation.tex").write_text(
        r"""
\subsection{Population mean--variance fixed point}
With $\xi=\log\sigma$, positive mass $p$, effective negative mass $q$, conditional moments $(m_+,v_+)$ and $(m_-,v_-)$. Define $M_\pm(\mu)=v_\pm+(\mu-m_\pm)^2$. Then
\begin{align}
\dot\mu &= \frac{p(m_+-\mu)-q(m_--\mu)}{\sigma^2},\\
\dot\xi &= \frac{pM_+(\mu)-qM_-(\mu)}{\sigma^2}-(p-q).
\end{align}
For $p>q$, the mean candidate is $\mu^\star=(pm_+-qm_-)/(p-q)$. Substitution into the scale equation gives
\begin{equation}
(\sigma^2)^\star=\frac{pM_+(\mu^\star)-qM_-(\mu^\star)}{p-q},
\end{equation}
A finite joint equilibrium therefore requires both $p>q$ and a positive numerator. This second condition is stricter in the far field because $M_-$ contains squared displacement.
""",
        encoding="utf-8",
    )
    (gen_dir / "app_categorical_derivation.tex").write_text(
        r"""
\subsection{Repeated negative updates and log-odds}
For softmax logits $z$, $\nabla_z\log p_y=e_y-p$. Under a fixed negative coefficient $c>0$ and small step size $h$, the selected logit changes by $-hc(1-p_y)$ while competitor $j$ changes by $+hc p_j$. Hence
\begin{equation}
(z_y-z_j)_{t+1}-(z_y-z_j)_t=-hc(1-p_y+p_j),
\end{equation}
which remains negative as $p_y\to0$. The Euclidean score is bounded, but the non-vanishing log-odds decrement yields persistent support suppression and can drive $p_y$ exponentially toward the simplex boundary.
""",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.repo_root.resolve()
    overleaf = root / "paper/overleaf"
    fig_dir = overleaf / "figures/generated"
    gen_dir = overleaf / "generated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    save_du1(root, fig_dir, gen_dir)
    save_e3(root, fig_dir, gen_dir)
    save_e4(root, fig_dir, gen_dir)
    save_proofs(gen_dir)
    print(json.dumps({"status": "BUILT", "project": "drpo"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
