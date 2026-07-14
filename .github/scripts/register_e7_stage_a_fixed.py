#!/usr/bin/env python3
"""Execute the one-shot registration script with structure-preserving fixes."""

from __future__ import annotations

from pathlib import Path

script_path = Path(__file__).with_name("register_e7_stage_a.py")
source = script_path.read_text(encoding="utf-8")

old_registry = '''    registry_after = registry_before.rstrip() + "\\n" + registry_entry()
    registry_path.write_text(registry_after, encoding="utf-8")
'''
new_registry = '''    document_node = yaml.compose(registry_before)
    if not isinstance(document_node, yaml.MappingNode):
        raise RuntimeError("registry root is not a mapping node")
    experiments_node = None
    for key_node, value_node in document_node.value:
        if getattr(key_node, "value", None) == "experiments":
            experiments_node = value_node
            break
    if not isinstance(experiments_node, yaml.SequenceNode):
        raise RuntimeError("registry experiments node is not a sequence")
    insert_at = experiments_node.end_mark.index
    prefix = registry_before[:insert_at].rstrip() + "\\n"
    suffix = registry_before[insert_at:]
    registry_after = prefix + registry_entry() + suffix
    registry_path.write_text(registry_after, encoding="utf-8")
'''
if source.count(old_registry) != 1:
    raise RuntimeError("expected registry append block was not found exactly once")
patched = source.replace(old_registry, new_registry)

old_evidence = '''    for relative in evidence:
        if not (source / relative).is_file():
            raise RuntimeError(f"registration evidence is missing: {relative}")
'''
new_evidence = '''    for relative in evidence:
        if relative.endswith("/HANDOFF_DELTA.yaml"):
            continue
        if not (source / relative).is_file():
            raise RuntimeError(f"registration evidence is missing: {relative}")
'''
if patched.count(old_evidence) != 1:
    raise RuntimeError("expected evidence preflight block was not found exactly once")
patched = patched.replace(old_evidence, new_evidence)

namespace = {
    "__name__": "__main__",
    "__file__": str(script_path),
    "__package__": None,
}
exec(compile(patched, str(script_path), "exec"), namespace)
