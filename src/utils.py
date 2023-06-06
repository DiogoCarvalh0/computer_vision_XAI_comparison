from pathlib import Path

import torch
from torchvision.utils import draw_bounding_boxes

import matplotlib.pyplot as plt


def collate_fn(batch):
    return tuple(zip(*batch))


def imshow_with_box(img, boxes, labels, ax=None, **kwargs):
    img_int = (img.clone().detach() * 255).to(torch.uint8)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(
        draw_bounding_boxes(
            image=img_int, boxes=boxes, labels=labels, **kwargs
        ).permute(1, 2, 0)
    )


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
    """
    Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
