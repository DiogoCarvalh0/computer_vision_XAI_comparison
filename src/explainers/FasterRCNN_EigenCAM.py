from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform


class FasterRCNN_EigenCAM:
    def __init__(self, model) -> None:
        self.model = model

    def generate_saliency_map(self, image, target_label, target_box, use_cuda=False):
        explanation_targets = [
            FasterRCNNBoxScoreTarget(labels=target_label, bounding_boxes=target_box)
        ]
        target_layers = [self.model.backbone]

        cam = EigenCAM(
            self.model,
            target_layers,
            use_cuda=use_cuda,
            reshape_transform=fasterrcnn_reshape_transform,
        )

        grayscale_cam = cam(image.unsqueeze(0), targets=explanation_targets)
        grayscale_cam = grayscale_cam[0, :]

        return grayscale_cam
