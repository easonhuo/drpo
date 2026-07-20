from pathlib import Path

path = Path('src/drpo/countdown_e8_alpha1_highc_scan_common.py')
text = path.read_text(encoding='utf-8')
old = '''    predecessor_compatible = copy.deepcopy(config)
    predecessor_compatible["remoteness"]["weight"] = "alpha*exp(-c*u^2)"
    with activated(profile):
        _BASE_VALIDATE_GRID_CONFIG(predecessor_compatible)
    if config["remoteness"].get("weight") != "alpha*exp(-c*u)":
        raise ValueError("The paper-aligned weight must be alpha*exp(-c*u)")
    points = tuple(
'''
new = '''    predecessor_compatible = copy.deepcopy(config)
    predecessor_compatible["remoteness"]["weight"] = "alpha*exp(-c*u^2)"
    if config["remoteness"].get("weight") != "alpha*exp(-c*u)":
        raise ValueError("The paper-aligned weight must be alpha*exp(-c*u)")
    points = tuple(
'''
if text.count(old) != 1:
    raise SystemExit('validation prelude not found exactly once')
text = text.replace(old, new)
needle = '''        if tuple(reference.get("seed_offsets", ())) != SEED_OFFSETS:
            raise ValueError("Historical reference seed offsets changed")
    if config["execution"].get("default_gpus") != list(range(8)):
'''
replacement = '''        if tuple(reference.get("seed_offsets", ())) != SEED_OFFSETS:
            raise ValueError("Historical reference seed offsets changed")
    with activated(profile):
        _BASE_VALIDATE_GRID_CONFIG(predecessor_compatible)
    if config["execution"].get("default_gpus") != list(range(8)):
'''
if text.count(needle) != 1:
    raise SystemExit('validation insertion point not found exactly once')
path.write_text(text.replace(needle, replacement), encoding='utf-8')
