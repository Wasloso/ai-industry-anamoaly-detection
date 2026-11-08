import logging
import time

import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from torchvision.transforms.functional import normalize

from defect_detection.config import PatchCoreConfig

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
logger = logging.getLogger(__name__)


class PatchCore:
    def __init__(self, config: PatchCoreConfig, image_size=256, device="auto", pretrained=True):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        self.device = torch.device(device)
        self.config = config
        self.image_size = image_size
        self.model = PatchcoreModel(
            backbone=config.backbone,
            layers=config.layers,
            pre_trained=pretrained,
            num_neighbors=config.num_neighbors,
        ).to(self.device)

    def preprocess(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(
                f"Expected images shaped [batch, 3, {self.image_size}, {self.image_size}]"
            )
        return normalize(images.to(self.device, dtype=torch.float32), MEAN, STD)

    @torch.no_grad()
    def fit(self, loader):
        started = time.perf_counter()
        self.model.embedding_store.clear()
        self.model.train()
        self.model.feature_extractor.eval()
        image_count = 0
        for index, batch in enumerate(loader):
            if torch.any(batch["label"] != 0):
                raise ValueError("PatchCore fitting requires normal images only")
            self.model(self.preprocess(batch["image"]))
            image_count += len(batch["image"])
            logger.info("Extracted features: batch %s/%s", index + 1, len(loader))
        patch_count = sum(len(embedding) for embedding in self.model.embedding_store)
        if int(patch_count * self.config.coreset_ratio) < 1:
            raise ValueError("Not enough patches for this coreset_ratio")
        self.model.subsample_embedding(self.config.coreset_ratio)
        self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "fitting_images": image_count,
            "patches": patch_count,
            "memory_bank_shape": list(self.model.memory_bank.shape),
            "fit_seconds": round(time.perf_counter() - started, 3),
            "device": str(self.device),
        }

    @torch.inference_mode()
    def predict(self, images):
        if not self.model.memory_bank.numel():
            raise ValueError("Fit or load a model before predicting")
        self.model.eval()
        output = self.model(self.preprocess(images))
        if (
            not torch.isfinite(output.pred_score).all()
            or not torch.isfinite(output.anomaly_map).all()
        ):
            raise ValueError("Model produced non-finite scores or maps")
        return output.pred_score.flatten().cpu(), output.anomaly_map.cpu()
