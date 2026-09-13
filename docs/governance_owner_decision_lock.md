# Repository-Owner Decision Lock Protocol

Policy ID: `GOV-OWNER-DECISION-LOCK-01`

This policy makes explicit repository-owner decisions durable project constraints.

## Locked decisions

A decision becomes `LOCKED` when the repository owner clearly settles an issue, selects an option, says not to revisit it, or limits the task to a specific change. Interpret the lock at the narrowest stated scope; do not infer a broader lock from tentative discussion.

Repository files, configs, prior implementations, and historical provenance remain evidence sources. They may reveal consequences or conflicts, but they do not automatically override a locked owner decision.

## Conflict consultation

When a current owner instruction conflicts with a repository document, config, historical implementation, provenance record, or prior `LOCKED` decision, and the owner has not already acknowledged that conflict and selected which instruction should govern, the agent must not resolve the conflict itself. State the conflict once in neutral factual terms, identify the two conflicting instructions or sources, and ask the owner which one should govern.

The conflict question is a request for the owner's decision, not a recommendation. Do not advise the owner to choose one side, rank or prefer the alternatives, pressure the owner, issue a command, imply that one choice is the default, or silently decide on the owner's behalf. Do not implement the disputed point until the owner answers.

If the owner has already explicitly acknowledged the conflict and chosen a direction, do not ask again. Treat that choice as the governing decision for the stated scope and execute it without further advocacy.

A hard conflict exists only when one requested choice would require a knowingly false factual claim, fabrication of experiment/repository state, violation of a system or repository hard constraint, or claiming an unavailable operation succeeded. State that constraint once as a factual limitation and ask the owner for a permissible choice. The agent still must not decide among the remaining permissible choices on the owner's behalf.

## No third option

If the owner chooses A and historical evidence points to B, do not silently invent C as a compromise. Propose a third option only when the owner explicitly asks for alternatives or a hard conflict makes literal execution impossible.

## Scope lock

If the owner says to change only a specified file, section, figure, table, parameter, claim, or behavior, modify only that scope. Do not add opportunistic cleanup, reframing, renaming, or unrelated improvements. If an unavoidable cascade falls outside scope, disclose it before applying it.

## Supersession

A locked decision remains active until the repository owner explicitly changes, supersedes, or reopens it. A conflicting new instruction is not silently treated as supersession when the conflict has not been acknowledged; use the conflict-consultation procedure above. Once the owner acknowledges the conflict and selects the new instruction, that choice supersedes the old lock within the stated scope. New evidence, another model's recommendation, a different config, or agent disagreement does not supersede an owner decision.

## Persistence

Locks that materially affect future research, experiments, manuscript structure, or project governance must be persisted through the repository's existing canonical authority path. Research/experiment locks go through the schema-v3 handoff-delta authority; manuscript structural locks go through the active outline/blueprint hierarchy; repository-wide agent behavior belongs in `AGENTS.md` and this policy. Do not create a competing master-status document.

## Relationship to independent judgment

Anti-sycophancy governs scientific and engineering verdicts; this policy governs project decisions and conflict resolution after the owner is asked to choose. Do not change a scientific verdict merely to agree with the owner, and do not use an independent verdict as permission to override an explicit owner choice about scope, presentation, implementation, or execution. When repository evidence conflicts with an owner instruction, present the conflict neutrally and request the owner's decision rather than converting the independent verdict into a recommendation or command.

## Ten-pass pre-delivery review

Before delivery, review the change ten times against these distinct checks:

1. relevant owner locks identified;
2. conflicts between the current instruction and documents/configs/prior locks identified;
3. every unresolved conflict was presented neutrally and the owner's decision obtained before implementation;
4. no recommendation, ranking, pressure, command, or implied default was inserted into conflict consultation;
5. no config/history silently overrode the owner's selected choice;
6. no unapproved third option was introduced;
7. any hard impossibility was stated once as a factual constraint and was not used to decide among permissible choices;
8. scope and explicit supersession were applied only where the owner chose them;
9. factual provenance, experiment status, and completed-action claims remain accurate;
10. final diff contains only approved scope plus disclosed mechanical cascades.

If any pass fails, fix it before delivery rather than asking the owner to rediscover the violation.
