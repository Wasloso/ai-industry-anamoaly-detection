from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: Path
    category: str
    manifest: Path
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 0
    validation_ratio: float = 0.1
    calibration_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.category, str) or not self.category.isidentifier():
            raise ValueError("category must be a single name, such as bottle or metal_nut")
        for field in ("image_size", "batch_size", "num_workers", "seed"):
            value = getattr(self, field)
            minimum = 1 if field in ("image_size", "batch_size") else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        for field in ("validation_ratio", "calibration_ratio"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not 0 < value < 1:
                raise ValueError(f"{field} must be between 0 and 1")
        if self.validation_ratio + self.calibration_ratio >= 1:
            raise ValueError("validation_ratio + calibration_ratio must be less than 1")


def load_data_config(path: Path) -> DataConfig:
    path = path.resolve()
    with path.open() as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError("The configuration must be a YAML mapping")
    for field in ("root", "manifest"):
        if not isinstance(values.get(field), str) or not values[field]:
            raise ValueError(f"{field} must be a non-empty path")
        values[field] = (path.parent / values[field]).resolve()
    try:
        return DataConfig(**values)
    except TypeError as error:
        raise ValueError(f"Invalid data configuration: {error}") from error
