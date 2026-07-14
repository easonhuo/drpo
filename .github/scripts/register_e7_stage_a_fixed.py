#!/usr/bin/env python3
"""Execute the one-shot registration script with a structure-preserving registry insert."""

from __future__ import annotations

from pathlib import Path

script_path = Path(__file__).with_name("register_e7_stage_a.py")
source = script_path.read_text(encoding="utf-8")
old = '''    registry_after = registry_before.rstrip() + "\\n" + registry_entry()
    registry_path.write_text(registry_after, encoding="utf-8")
'''
new = '''    document_node = yaml.compose(registry_before)
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
if source.count(old) != 1:
    raise RuntimeError("expected registry append block was not found exactly once")
patched = source.replace(old, new)
namespace = {
    "__name__": "__main__",
    "__file__": str(script_path),
    "__package__": None,
}
exec(compile(patched, str(script_path), "exec"), namespace)
