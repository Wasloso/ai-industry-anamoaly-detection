from dataclasses import replace

import pytest

from defect_detection.config import load_data_config


def test_paths_are_relative_to_config_file(tmp_path, monkeypatch):
    path = tmp_path / "configs" / "bottle.yaml"
    path.parent.mkdir()
    path.write_text("root: ../data\ncategory: bottle\nmanifest: ../manifest.json\n")
    monkeypatch.chdir("/")
    config = load_data_config(path)
    assert config.root == tmp_path / "data"
    assert config.manifest == tmp_path / "manifest.json"


@pytest.mark.parametrize(
    "changes",
    [
        {"image_size": 0},
        {"batch_size": True},
        {"num_workers": -1},
        {"category": "../bottle"},
        {"validation_ratio": 0.6, "calibration_ratio": 0.5},
        {"calibration_ratio": float("nan")},
    ],
)
def test_invalid_settings_are_rejected(data_config, changes):
    with pytest.raises(ValueError):
        replace(data_config, **changes)
