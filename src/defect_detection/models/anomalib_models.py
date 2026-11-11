from anomalib.models import Padim, Patchcore
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize


def build_model(settings: dict, image_size: int, *, pretrained: bool):
    transform = Compose(
        [
            Resize((image_size, image_size), antialias=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    models = {"patchcore": Patchcore, "padim": Padim}
    return models[settings["name"]](
        backbone=settings["backbone"],
        layers=settings["layers"],
        pre_trained=pretrained,
        pre_processor=PreProcessor(transform),
        post_processor=False,
        evaluator=False,
        visualizer=False,
        **settings["parameters"],
    )
