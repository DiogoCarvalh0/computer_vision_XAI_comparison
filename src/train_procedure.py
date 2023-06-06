from typing import List, Dict
from tqdm import tqdm

import pandas as pd

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Dict: loss_total, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
    """
    model.train()

    all_losses_dict = []

    for X, y in tqdm(dataloader, desc="train", leave=False):
        X = list(image.to(device) for image in X)
        y = [
            {k: v.clone().detach().to(device) for k, v in target.items()}
            for target in y
        ]

        loss_dict = model(X, y)

        total_loss = sum(loss for loss in loss_dict.values())

        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_dict_append["loss_total"] = total_loss.item()
        all_losses_dict.append(loss_dict_append)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return pd.DataFrame(all_losses_dict).mean().to_dict()


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metric: MeanAveragePrecision = None,
) -> MeanAveragePrecision:
    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        metric: MeanAveragePrecision to compute the metrics
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        MeanAveragePrecision
    """
    model.eval()

    metric = MeanAveragePrecision() if not metric else metric

    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="test", leave=False):
            X = list(image.to(device) for image in X)
            y = [
                {k: v.clone().detach().to(device) for k, v in target.items()}
                for target in y
            ]

            preds = model(X)

            metric.update(preds, y)

    return metric


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    # writer: torch.utils.tensorboard.writer.SummaryWriter
) -> Dict[str, List]:
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    It uses Reduce LR On Plateau regularization based on test loss.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.
    """

    results = {
        "train_loss_total": [],
        "train_loss_classifier": [],
        "train_loss_box_reg": [],
        "train_loss_objectness": [],
        "train_loss_rpn_box_reg": [],
        "map": [],
        "map_50": [],
        "map_75": [],
    }

    model.to(device)

    test_metric = MeanAveragePrecision()

    for epoch in tqdm(range(epochs)):
        train_results = train_step(
            model=model, dataloader=train_dataloader, optimizer=optimizer, device=device
        )

        test_metric = test_step(
            model=model, dataloader=test_dataloader, device=device, metric=test_metric
        )

        test_metrics = test_metric.compute()

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss_total: {train_results['loss_total']:.4f} | "
            f"train_loss_classifier: {train_results['loss_classifier']:.4f} | "
            f"train_loss_box_reg: {train_results['loss_box_reg']:.4f} | "
            f"train_loss_objectness: {train_results['loss_objectness']:.4f} | "
            f"train_loss_rpn_box_reg: {train_results['loss_rpn_box_reg']:.4f} | "
            f"test mAP: {test_metrics['map'].item():.4f} | "
            f"test mAP50: {test_metrics['map_50'].item():.4f} | "
            f"test mAP75: {test_metrics['map_75'].item():.4f}"
        )

        results["train_loss_total"].append(train_results["loss_total"])
        results["train_loss_classifier"].append(train_results["loss_classifier"])
        results["train_loss_box_reg"].append(train_results["loss_box_reg"])
        results["train_loss_objectness"].append(train_results["loss_objectness"])
        results["train_loss_rpn_box_reg"].append(train_results["loss_rpn_box_reg"])
        results["map"].append(test_metrics["map"].item())
        results["map_50"].append(test_metrics["map_50"].item())
        results["map_75"].append(test_metrics["map_75"].item())

        # if writer:
        # writer.add_scalar(tag="Loss/train", scalar_value=train_loss, global_step=epoch)
        # writer.add_scalar(tag="Loss/test", scalar_value=test_loss, global_step=epoch)
        # writer.add_scalar(tag="Accuracy/train", scalar_value=train_acc, global_step=epoch)
        # writer.add_scalar(tag="Accuracy/test", scalar_value=test_acc, global_step=epoch)
        # writer.add_scalar(tag="AUC/train", scalar_value=train_auc, global_step=epoch)
        # writer.add_scalar(tag="AUC/test", scalar_value=test_auc, global_step=epoch)

        # writer.close()

    return results
