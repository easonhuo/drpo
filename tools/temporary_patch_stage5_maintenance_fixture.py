from pathlib import Path

path = Path("tests/stage5_candidate_integration.py")
text = path.read_text(encoding="utf-8")
old = '''    fixture_source = source_head
    if committed_authority["mode"] == "delta":
        fixture_source = committed_authority["delta_authority"][
            "activation_parent_commit"
        ]
'''
new = '''    fixture_source = source_head
    maintenance_candidate = (
        os.environ.get("DRPO_STAGE5_MAINTENANCE_CANDIDATE") == "1"
    )
    if committed_authority["mode"] == "delta" and not maintenance_candidate:
        fixture_source = committed_authority["delta_authority"][
            "activation_parent_commit"
        ]
'''
if text.count(old) != 1:
    raise SystemExit("Stage 5 fixture-source block not found exactly once")
path.write_text(text.replace(old, new), encoding="utf-8")
