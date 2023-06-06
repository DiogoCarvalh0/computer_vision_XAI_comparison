from tqdm import tqdm

import numpy as np

from sklearn.metrics import auc

import torch


class Deletion:
    def __init__(self, model, step, substrate_fn, device="cpu"):
        """Create deletion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            device (str, optional): Device to use. Defaults to 'cpu'.
        """
        self.model = model
        self.step = step
        self.substrate_fn = substrate_fn
        self.device = device

        self.model.to(self.device)

    def compute_metric(
        self, img_tensor, explanation, target_label, target_box, show_progess_bar=True
    ):
        """
        Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.

        Return:
            (np.array): Array containing scores at every step.
        """
        img_area = np.prod(img_tensor.shape[1:])
        n_steps = (img_area + self.step - 1) // self.step

        start = img_tensor.clone()
        finish = self.substrate_fn(img_tensor)

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(
            np.argsort(explanation.reshape(-1, img_area), axis=1), axis=-1
        )

        for i in tqdm(range(n_steps + 1), disable=not show_progess_bar):
            pred = self.model([start.to(self.device)])[0]

            scores[i] = self._compute_score(target_label, target_box, pred)

            if i < n_steps:
                coords = salient_order[:, self.step * i : self.step * (i + 1)]

                start.cpu().numpy().reshape(1, 3, img_area)[0, :, coords] = (
                    finish.cpu().numpy().reshape(1, 3, img_area)[0, :, coords]
                )

        return scores, auc(x=np.arange(len(scores)) / len(scores), y=scores)

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
