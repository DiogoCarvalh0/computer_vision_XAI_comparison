import torch
import torchvision
from torchvision.ops import batched_nms


def predict(
    model: torchvision.models,
    images: torch.tensor,
    device,
    prediction_confidence: float = 0.5,
    use_nms: bool = True,
    iou_threshold: float = 0.1,
) -> list:
    """Batch model prediction.

    Args:
        model (torchvision.models): Model
        images (torch.tensor): Batch of images: [N, C, H, W]
        prediction_confidence (float, optional): Prediction confidence. Only will consider predictions with score > prediction_confidence. Defaults to 0.5.
        use_nms (bool, optional): Whether to use or not Non-Maximum Suppression. Defaults to True.
        iou_threshold (float, optional): Non-Maximum Suppression threshold. Defaults to 0.1.

    Returns:
        list: List with predictions for every image.
    """
    model.to(device)
    images = images.to(device)
    predictions = model(images)

    for i, prediction in enumerate(predictions):
        predictions[i] = {
            key: value[prediction["scores"] > prediction_confidence]
            for key, value in prediction.items()
        }

    if use_nms:
        for i, prediction in enumerate(predictions):
            idx_to_keep = batched_nms(
                boxes=prediction["boxes"],
                scores=prediction["scores"],
                idxs=prediction["labels"],
                iou_threshold=iou_threshold,
            )

        predictions[i] = {
            key: values[idx_to_keep] for key, values in prediction.items()
        }

    return predictions


def predict_single_image(
    model: torchvision.models,
    image: torch.tensor,
    device,
    prediction_confidence: float = 0.5,
    use_nms: bool = True,
    iou_threshold: float = 0.1,
) -> dict:
    """Model prediction for a single image.

    Args:
        model (torchvision.models): Model
        image (torch.tensor): Image: [C, H, W]
        prediction_confidence (float, optional): Prediction confidence. Only will consider predictions with score > prediction_confidence. Defaults to 0.5.
        use_nms (bool, optional): Whether to use or not Non-Maximum Suppression. Defaults to True.
        iou_threshold (float, optional): Non-Maximum Suppression threshold. Defaults to 0.1.

    Returns:
        dict: Dict with predictions for the image. Dict has 'boxes', 'scores', 'labels'.
    """
    return predict(
        model=model,
        images=image.unsqueeze(0),
        device=device,
        prediction_confidence=prediction_confidence,
        use_nms=use_nms,
        iou_threshold=iou_threshold,
    )[0]
