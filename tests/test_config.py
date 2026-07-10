from pathlib import Path

import pytest

from drpo.config import load_config


def test_load_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("experiment:\n  seed: 1\n", encoding="utf-8")

    config = load_config(path)

    assert config["experiment"]["seed"] == 1


@pytest.mark.parametrize("contents", ["", "- item\n"])
def test_load_config_rejects_non_mapping_yaml(tmp_path: Path, contents: str):
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="Config must be a mapping"):
        load_config(path)
