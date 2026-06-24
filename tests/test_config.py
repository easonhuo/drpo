from pathlib import Path

from drpo.config import load_config


def test_load_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("experiment:\n  seed: 1\n", encoding="utf-8")

    config = load_config(path)

    assert config["experiment"]["seed"] == 1
