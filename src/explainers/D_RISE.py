from tqdm import tqdm
import gc

import math
import numpy as np

import cv2

import torch


class D_RISE:
    def __init__(self, model, device="cpu") -> None:
        self.model = model
        self.device = device

        self.model.to(device)

    def generate_saliency_map(
        self,
        image: torch.tensor,
        target_label,
        target_box,
        batch_size=32,
        nr_iterations=5000,
        grid_size=(16, 16),
        prob_thresh=0.5,
        show_progess_bar=True,
    ):
        """_summary_

        Args:
            image (torch.tensor): Image on format CxHxW
            target_class_index (_type_): _description_
            target_box (_type_): _description_
            nr_iterations (int, optional): _description_. Defaults to 5000.
            grid_size (tuple, optional): _description_. Defaults to (16, 16).
            prob_thresh (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """

        image_h, image_w = image.shape[1:]

        res = np.zeros((image_h, image_w), dtype=np.float32)

        masks = [
            self._generate_mask(
                image_size=(image_w, image_h),
                grid_size=grid_size,
                prob_thresh=prob_thresh,
            )
            for _ in range(nr_iterations)
        ]

        for start in tqdm(
            range(0, nr_iterations, batch_size), disable=not show_progess_bar
        ):
            batch_masks = masks[start : min(start + batch_size, nr_iterations)]

            masked_imgs = torch.stack(
                [self._mask_image(image, mask) for mask in batch_masks]
            )

            preds = self.model(masked_imgs.to(self.device))

            for pred, mask in zip(preds, batch_masks):
                score = self._compute_score(target_label, target_box, pred)

                res += mask.detach().numpy() * score

            del masked_imgs
            gc.collect()

        # Normalize values
        res = (res - res.min()) / (res.max() - res.min())

        return res

    def _generate_mask(
        self, image_size: list[int, int], grid_size: list[int, int], prob_thresh: float
    ):
        image_w, image_h = image_size
        grid_w, grid_h = grid_size

        cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
        up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

        mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) < prob_thresh).astype(
            np.float32
        )
        mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)

        offset_w = np.random.randint(0, cell_w)
        offset_h = np.random.randint(0, cell_h)
        mask = mask[offset_h : offset_h + image_h, offset_w : offset_w + image_w]

        return torch.tensor(mask, dtype=torch.float32)

    def _mask_image(self, image, mask):
        masked = image.type(torch.float32) * torch.permute(
            torch.dstack([mask] * 3), (2, 0, 1)
        )

        return masked

    def _compute_score(self, target_label, target_box, prediction):
        score = max(
            [
                self._iou(target_box, box) * score.item()
                for box, label, score in zip(
                    prediction["boxes"], prediction["labels"], prediction["scores"]
                )
                if label.item() == target_label
            ],
            default=0,
        )

        return score

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
