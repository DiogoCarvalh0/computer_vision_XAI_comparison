import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import DogVSCatObjectDetectionDataset
from src.utils import collate_fn, save_model
from src.models import create_fasterrcnn_mobilenet_v3_large_model
from src.train_procedure import train


DATA_PATH = "./data/"
TRAIN_FOLDER = "train_artifacts/"
TEST_FOLDER = "test_artifacts/"
IMAGES_FOLDER = "images/"
ANNOTATIONS_FOLDER = "annotations/"

MODEL_FOLDER = "./models/"
MODEL_NAME = "Artifacts_Faster_RCNN_mobilenet_v3.pt"  # .pt or .pth

LABELS_TO_IDX = {"background": 0, "cat": 1, "dog": 2}

NUMBER_CLASSES = len(LABELS_TO_IDX)

TRANSFORM = transforms.Compose([transforms.ToTensor()])

BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 10

# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # MPS give an error
DEVICE = "cpu"


def main():
    train_dataset = DogVSCatObjectDetectionDataset(
        images_path=os.path.join(DATA_PATH, TRAIN_FOLDER, IMAGES_FOLDER),
        annotations_path=os.path.join(DATA_PATH, TRAIN_FOLDER, ANNOTATIONS_FOLDER),
        class_to_idx_mapping=LABELS_TO_IDX,
        transform=TRANSFORM,
    )

    test_dataset = DogVSCatObjectDetectionDataset(
        images_path=os.path.join(DATA_PATH, TEST_FOLDER, IMAGES_FOLDER),
        annotations_path=os.path.join(DATA_PATH, TEST_FOLDER, ANNOTATIONS_FOLDER),
        class_to_idx_mapping=LABELS_TO_IDX,
        transform=TRANSFORM,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = create_fasterrcnn_mobilenet_v3_large_model(NUMBER_CLASSES)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4
    )
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=DEVICE,
    )

    save_model(model=model, target_dir=MODEL_FOLDER, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
