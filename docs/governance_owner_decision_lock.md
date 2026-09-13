# Repository-Owner Decision Lock Protocol

Policy ID: `GOV-OWNER-DECISION-LOCK-01`

This policy makes explicit repository-owner decisions durable project constraints.

## Locked decisions

A decision becomes `LOCKED` when the repository owner clearly settles an issue, selects an option, says not to revisit it, or limits the task to a specific change. Interpret the lock at the narrowest stated scope; do not infer a broader lock from tentative discussion.

Repository files, configs, prior implementations, and historical provenance remain evidence sources. They may reveal consequences or conflicts, but they do not automatically override a locked owner decision.

## Conflict handling

For an ordinary conflict, preserve and execute the locked decision. Do not reopen it, ask for the same approval again, or silently replace it with another option.

A hard conflict exists only when literal execution would require a knowingly false factual claim, fabrication of experiment/repository state, violation of a system or repository hard constraint, or claiming an unavailable operation succeeded. Raise such a conflict once, concisely, with the smallest permissible implementation that preserves the owner's intent. After the owner selects or confirms a permissible implementation, execute it without further advocacy.

## No third option

If the owner chooses A and historical evidence points to B, do not silently invent C as a compromise. Propose a third option only when the owner explicitly asks for alternatives or a hard conflict makes literal execution impossible.

## Scope lock

If the owner says to change only a specified file, section, figure, table, parameter, claim, or behavior, modify only that scope. Do not add opportunistic cleanup, reframing, renaming, or unrelated improvements. If an unavoidable cascade falls outside scope, disclose it before applying it.

## Supersession

A locked decision remains active until the repository owner explicitly changes, supersedes, or reopens it. New evidence, another model's recommendation, a different config, or agent disagreement does not supersede it.

## Persistence

Locks that materially affect future research, experiments, manuscript structure, or project governance must be persisted through the repository's existing canonical authority path. Research/experiment locks go through the schema-v3 handoff-delta authority; manuscript structural locks go through the active outline/blueprint hierarchy; repository-wide agent behavior belongs in `AGENTS.md` and this policy. Do not create a competing master-status document.

## Relationship to independent judgment

Anti-sycophancy governs scientific and engineering verdicts; this policy governs project decisions after the owner explicitly settles them. Do not change a scientific verdict merely to agree with the owner, and do not use an independent verdict as permission to override an explicit owner choice about scope, presentation, implementation, or execution. If a lock would force a false factual statement, use the one-time hard-conflict procedure rather than silently changing the choice or repeatedly arguing.

## Ten-pass pre-delivery review

Before delivery, review the change ten times against these distinct checks:

1. relevant owner locks identified;
2. scope unchanged outside the approved area;
3. no config/history silently overrides the lock;
4. no unapproved third option introduced;
5. any hard conflict raised at most once;
6. factual provenance remains accurate;
7. no active lock treated as obsolete without explicit supersession;
8. required cascades updated and unrelated content preserved;
9. no result status or completed action falsely claimed;
10. final diff contains only approved scope plus disclosed mechanical cascades.

If any pass fails, fix it before delivery rather than asking the owner to rediscover the violation.
