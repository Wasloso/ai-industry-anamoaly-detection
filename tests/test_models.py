from pathlib import Path

import pytest
import torch
import yaml

from defect_detection.config import load_model_config
from defect_detection.models.anomalib_models import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name, expected", [("patchcore", "Patchcore"), ("padim", "Padim")])
def test_model_configs_build_with_common_preprocessing(name, expected):
    torch.set_num_threads(2)
    data, settings = load_model_config(ROOT / f"configs/mvtec-bottle-{name}.yaml")
    model = build_model(settings, data.image_size, pretrained=False)
    assert type(model).__name__ == expected
    assert settings["backbone"] == "resnet18"
    assert settings["layers"] == ["layer2", "layer3"]
    assert model.post_processor is None
    image = torch.zeros(1, 3, 20, 24)
    processed = model.pre_processor(image)
    expected_pixel = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
    assert processed.shape == (1, 3, data.image_size, data.image_size)
    torch.testing.assert_close(processed[0, :, 0, 0], expected_pixel)


def test_parameters_from_another_method_are_rejected(tmp_path):
    values = yaml.safe_load((ROOT / "configs/mvtec-bottle-patchcore.yaml").read_text())
    values["model"]["parameters"]["n_features"] = 100
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="Unsupported parameters for patchcore"):
        load_model_config(path)
