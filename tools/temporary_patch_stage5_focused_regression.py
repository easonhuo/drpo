from pathlib import Path

path = Path("tests/stage5_candidate_integration.py")
text = path.read_text(encoding="utf-8")
old = '''    central = copy_repository(tmp_path / "preintegration-report-revision")
    _, cutover = activate_delta_mode(central)
'''
new = '''    central = copy_repository(tmp_path / "preintegration-report-revision")
    current_authority = yaml.safe_load(
        (central / "docs/handoff_versions/AUTHORITY.yaml").read_text(encoding="utf-8")
    )
    if current_authority["mode"] == "delta":
        cutover = git_text(central, "rev-parse", "HEAD")
    else:
        _, cutover = activate_delta_mode(central)
'''
if text.count(old) != 1:
    raise SystemExit("focused Stage 5 regression setup not found exactly once")
path.write_text(text.replace(old, new), encoding="utf-8")
