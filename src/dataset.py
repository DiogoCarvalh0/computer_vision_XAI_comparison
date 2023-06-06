import os
from typing import Tuple, Dict
from pathlib import Path
import glob

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import xml.etree.ElementTree as ET


class DogVSCatObjectDetectionDataset(Dataset):
    def __init__(
        self,
        images_path: Path,
        annotations_path: Path,
        class_to_idx_mapping: Dict,
        transform: transforms.Compose = None,
    ) -> None:
        super().__init__()
        self.images_path = images_path

        self.annotations_path = annotations_path
        self.annotations = glob.glob(os.path.join(annotations_path, "*.xml"))

        self.class_to_idx_mapping = class_to_idx_mapping
        self.classes = list(class_to_idx_mapping.keys())
        self.idx_to_classes_mapping = {
            value: key for key, value in class_to_idx_mapping.items()
        }

        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(
            self.images_path,
            os.path.basename(self.annotations[index]).split(".")[0] + ".png",
        )
        annotation_path = self.annotations[index]

        image = Image.open(img_path).convert("RGB")
        targets = self._read_xml_file(annotation_path)
        targets["image_id"] = torch.tensor(index, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, targets

    def get_file_name(self, index: int) -> str:
        return os.path.basename(self.annotations[index]).split(".")[0]

    def _read_xml_file(self, file: Path) -> Dict:
        targets = {}
        boxes = []
        labels = []
        areas = []

        root = ET.parse(file).getroot()

        for obj in root.findall("object"):
            label = self.class_to_idx_mapping[obj.find("name").text]
            labels.append(label)

            object_bndbox = obj.find("bndbox")

            xmin = int(object_bndbox.find("xmin").text)
            ymin = int(object_bndbox.find("ymin").text)
            xmax = int(object_bndbox.find("xmax").text)
            ymax = int(object_bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])

            area = (xmax - xmin) * (ymax - ymin)
            areas.append(area)

        targets["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        targets["labels"] = torch.tensor(labels, dtype=torch.int64)
        targets["area"] = torch.tensor(areas, dtype=torch.float32)

        return targets
