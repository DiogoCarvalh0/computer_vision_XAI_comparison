import numpy as np

import torch
import torchvision.transforms as transforms

from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function

from src.prediction import predict_single_image


class LIME:
    def __init__(self, model, device="cpu") -> None:
        self.model = model
        self.device = device

        self.model.to(device)

    def generate_saliency_map(
        self,
        image,
        target_label,
        target_box,
        n_samples=5000,
        super_pixel_size=35,
        show_progress=True,
    ):
        exp_eucl_distance = get_exp_kernel_similarity_function(
            "euclidean", kernel_width=1000
        )

        lr_lime = Lime(
            self._wrapper,
            interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
            similarity_func=exp_eucl_distance,
        )

        feature_mask = self._define_super_pixeis(image, super_pixel_size)

        lime_explanation = lr_lime.attribute(
            image,
            target=0,
            additional_forward_args=(target_label, target_box),
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=1,
            show_progress=show_progress,
        ).squeeze(0)

        normalizes_lime_explanation = (
            lime_explanation - lime_explanation.min().item()
        ) / (lime_explanation.max().item() - lime_explanation.min().item())

        lime_explanation_grayscale = transforms.functional.rgb_to_grayscale(
            normalizes_lime_explanation
        )

        return lime_explanation_grayscale[0].numpy()

    def _wrapper(self, img, target_label, target_box):
        pred = predict_single_image(
            self.model, img, device=self.device, use_nms=True, iou_threshold=0.1
        )

        metric = max(
            [
                self._iou(target_box, box) * score.item()
                for box, label, score in zip(
                    pred["boxes"], pred["labels"], pred["scores"]
                )
                if label.item() == target_label
            ],
            default=0,
        )
        return torch.tensor([[metric]])

    def _define_super_pixeis(self, img, super_pixel_size):
        feature_mask = img[0].clone().zero_()

        count = 0

        for x in range(int(np.ceil(feature_mask.shape[0] / super_pixel_size))):
            x_start = x * super_pixel_size
            x_end = (x + 1) * super_pixel_size

            for y in range(int(np.ceil(feature_mask.shape[1] / super_pixel_size))):
                y_start = y * super_pixel_size
                y_end = (y + 1) * super_pixel_size

                feature_mask[x_start:x_end, y_start:y_end] = count
                count += 1

        return feature_mask.type(torch.long)

    def _iou(self, box1, box2):
        top_left = torch.vstack([box1[:2], box2[:2]]).max(axis=0).values
        bottom_right = torch.vstack([box1[2:], box2[2:]]).min(axis=0).values

        intersection = torch.prod(bottom_right - top_left) * torch.all(
            top_left < bottom_right
        ).type(torch.float)

        area1 = torch.prod(box1[2:] - box1[:2])
        area2 = torch.prod(box2[2:] - box2[:2])

        iou = intersection / (area1 + area2 - intersection)

        return iou.item()
