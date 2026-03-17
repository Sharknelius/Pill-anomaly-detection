import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

# Don't worry about this too much
# This is how we load in the dataset and annotations for training and validation
class CustomDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.annotation_file = os.path.join(dataset_dir, "_annotations.coco.json")

        with open(self.annotation_file) as f:
            annotations = json.load(f)

        self.images = annotations["images"]
        self.annotations = annotations["annotations"]
        self.categories = annotations["categories"]

        self.image_id_to_annotations = {}

        for annotation in self.annotations:
            image_id = annotation["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]

        self.image_dir = os.path.join(self.dataset_dir, "images")

        img_path = os.path.join(self.dataset_dir, "images", image_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        img_id = image_info["id"]

        anns = self.image_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:

            x, y, w, h = ann["bbox"]

            boxes.append([
                x,
                y,
                x + w,
                y + h
            ])

            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transform:
            img = self.transform(img)

        return img, target