import copy

import numpy as np

import torch
from torchvision import transforms


class FasterRCNN_GradCAM:
    def __init__(self, model, target_layer):
        self.model = copy.copy(model)
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.activation_maps = None
        self.hook_layer()

    def hook_layer(self):
        def hook_fn(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach().cpu()

        def forward_fn(module, input, output):
            try:
                self.feature_maps = output.detach().cpu()
            except:
                self.feature_maps = output["0"].detach().cpu()

        # target_layer = self.model.roi_heads.box_roi_pool  # Change to the target layer of your choice
        self.target_layer.register_forward_hook(forward_fn)
        self.target_layer.register_backward_hook(hook_fn)

    def forward(self, image):
        self.model.zero_grad()
        output = self.model(image)
        self.output = output
        output[0]["scores"][0].backward()
        self.activation_maps = self.gradients[0]

        return output

    def generate_saliency_map(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(image.shape) != 4:
            raise ValueError(
                f"Image with wrong dimensions ({image.shape}). Expected image to be 3-dimensions [C, H, W] or 4-dimensions [1, C, H, W]."
            )

        img_shape = image.shape[2:]
        self.output = self.forward(image)

        weight = torch.mean(self.activation_maps, axis=(1, 2))  # [C]

        cam = self.feature_maps[0] * weight[:, np.newaxis, np.newaxis]  # [C, H, W]
        self.log = cam
        resizer = transforms.Resize(img_shape)
        cam = resizer(cam)
        cam = torch.sum(cam, axis=0)
        cam = torch.relu(cam)

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
