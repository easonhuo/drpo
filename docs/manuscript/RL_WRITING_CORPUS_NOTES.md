# RL Writing Corpus Notes v1.1

**Status:** source and extraction record. New entries do not automatically change Guidance.

The corpus has two purposes:

1. identify recurring story, theory, method, and evidence patterns in strong RL papers;
2. extract operational techniques from open-source writing skills without executing untrusted code.

---

## 1. RL paper patterns retained

| Paper family | Retained writing move | DRPO use |
|---|---|---|
| TRPO / PPO | define one operational optimization problem and a memorable control principle | keep repulsive dynamics and the control object easy to restate |
| SAC / MPO | derive a practical update from an idealized policy-improvement object | Theorem 1 and DRPO must share the aggregate negative term |
| CQL / pessimistic offline RL | identify one failure and prove one property that directly counters it | badness--distance isolation and far-field tail control |
| IQL | ask one sharp question and refuse adjacent scope expansion | do not turn the paper into all of offline RL |
| TD3+BC / ReBRAC | let broad fair experiments defend a simple method; separate core method from implementation | matched budgets, paired seeds, sensitivity and terminal audits |
| AWAC / offline-to-online work | anchor the problem in realistic data flow before returning to controlled analysis | external occurrence -> controlled identification -> external closure |
| DPO and derivation-led objectives | make derivation, loss, and implementation act on the same variable | $q\mathbf m_-\rightarrow q_\lambda\mathbf m_{-,\lambda}$ |
| Decision Transformer / broad reframing papers | maintain one memorable reframing across title, method, figures, and experiments | negative feedback as a resource with a far-field failure mode |

---

## 2. Open-source skills reviewed

### Master-cai / Research-Paper-Writing-Skills

Retained:

- paragraph-level rhetorical moves;
- topic-sentence discipline;
- claim-evidence alignment;
- reviewer-mindset audit;
- section-specific editing.

Rejected/limited:

- generic prose rules that ignore experiment status or repository evidence.

### SNL-UCSB / paper-writing-skill

Retained:

- Brainstorm -> Draft 0 -> Evaluate -> Write -> Compress;
- introduction-twice ordering;
- require the author to state who suffers, what breaks, and what evidence exists before prose.

### NousResearch / research-paper-writing

Retained:

- literature search saturation rather than endless accumulation;
- mandatory citation verification;
- one-sentence contribution;
- every experiment serves a named claim;
- separate research, writing, review, and revision rounds.

### jin-s13 / ai-research-writing-skill

Retained:

- claim-evidence engineering;
- repository evidence inventory;
- trace claims to code, logs, results, or verified citations;
- build-ready LaTeX and submission package;
- reviewer-risk and rebuttal-risk audit.

### Auto-claude-code-research-in-sleep paper pipeline

Retained:

- explicit paper-plan -> figure -> write -> compile -> improvement loop;
- deterministic figure specifications where practical;
- five scientific writing passes;
- clean bibliography checks;
- final compiled PDF as an actual gate.

Rejected/limited:

- autonomous improvement loops that can alter approved scientific claims without authorization.

### Imbad0202 / academic-research-skills

Retained:

- specialist review roles;
- style calibration as a separate stage;
- LaTeX hardening;
- visualization and citation audits;
- revision coaching rather than one-pass generation.

Rejected/limited:

- agent consensus as authority; handoff and user approval remain authoritative.

### K-Dense-AI / claude-scientific-writer

Retained:

- section-specific skills;
- separation of research, citation, writing, and document-generation tasks;
- structured output and publication formatting.

### andrehuang / academic-writing-agents

Retained:

- independent reviewer roles;
- severity-ranked synthesis;
- iterative fix/review cycles.

### labarba / SciWrite

Retained:

- clutter extraction;
- information flow and stress position;
- strong subjects and verbs;
- targeted review modes;
- paragraph-by-paragraph revision with concrete findings.

### Research-Equality / RE-paper-writing and related catalogs

Retained:

- normalize skills into planning, literature, drafting, citation, LaTeX, review, revision, and rebuttal stages;
- do not mix upstream snapshots with the authoritative local skill set.

---

## 3. Cross-source techniques promoted to Playbook

The following appeared across multiple sources and are therefore operationally retained:

1. claim-evidence map before prose;
2. one-sentence contribution and one central tension;
3. paragraph rhetorical moves;
4. introduction written twice;
5. theorem-to-method equation chain;
6. every experiment serves a claim and rival explanation;
7. Figure 1 carries the story;
8. independent specialist reviews;
9. writing and compression as separate passes;
10. five-pass scientific prose audit;
11. citation verification from primary metadata;
12. clean LaTeX build and visual PDF inspection;
13. evidence-backed submission package;
14. stable-ID cascade and conflict-intolerant synchronization.

---

## 4. Techniques deliberately not adopted

- automatic invention of claims, theorems, experiments, or result values;
- free-form reverse inference from prose to scientific structure;
- rewriting an approved version under the same version label;
- downloading or executing third-party skill code without review;
- treating stylistic consensus among agents as scientific evidence;
- citation generation from model memory;
- defensive limitation inventories in the main story;
- framing a rewrite of the same manuscript as a prior paper plus a sequel.

---

## 5. Corpus update rule

A source is added only if it contributes a concrete technique, counterexample, or review gate. A new source modifies stable Guidance only when its principle generalizes across papers and resolves an observed failure. Otherwise it updates this corpus or the Playbook.
