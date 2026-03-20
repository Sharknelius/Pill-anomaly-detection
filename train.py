from torch.utils.data import DataLoader
from dataset import CustomDataset
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
from torch.amp import GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def collate_fn(batch):
    return tuple(zip(*batch))

def train(model, weights, dataset_dir):

    num_epochs = 20
    batch_size = 4
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = weights.transforms()

    # Find and load training and validation datasets
    train_path = os.path.join(dataset_dir, "train")
    val_path = os.path.join(dataset_dir, "valid")

    train_dataset = CustomDataset(train_path, transforms)
    val_dataset = CustomDataset(val_path, transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )
    print("Training set loaded successfully.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )
    print("Validation set loaded successfully.")

    model.to(device)

    # Freeze backbone until epoch 3
    for param in model.backbone.parameters():
        param.requires_grad = False

    # SGD
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()

    # Track mAP for each epoch and save the best model
    metric = MeanAveragePrecision(class_metrics=True).to(device)
    best_map = 0.0

    # Training
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")

        # Unfreeze backbone after 3 epochs
        if epoch == 3:
            for param in model.backbone.parameters():
                param.requires_grad = True

        model.train()
        total_loss = 0

        for images, targets in train_loader:

            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()

        with torch.no_grad():
            print("Starting Validation...")
            with torch.no_grad():
                with torch.autocast(device_type='cuda'):
                    for images, targets in val_loader:

                        images = [img.to(device, non_blocking=True) for img in images]
                        targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]

                        outputs = model(images)
                        preds = [{k: v.to("cpu") for k, v in out.items()} for out in outputs]
                        gt = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

                        metric.update(preds, gt)

        map_results = metric.compute()
        curr_map = map_results['map'].item()

        print(f"Epoch {epoch+1} mAP: {curr_map:.4f}")
        print(f"Epoch {epoch+1} mAP@50: {map_results['map_50'].item():.4f}")

        if curr_map > best_map:
            best_map = curr_map
            torch.save(model.state_dict(), "best_faster_rcnn.pt")
        
        # Reset metric for next epoch
        metric.reset()
        scheduler.step()

def start_training(load_custom, model_path, dataset_path):
    if load_custom: # Load the last trained model and continue training
        model, weights, categories = create_model(
            num_classes=3, pretrained=False, coco_model=False, categories= ["background", "capsules", "tablets"]
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else: # Start training from pretrained COCO model
        model, weights, categories = create_model(
            num_classes=3, pretrained=True, coco_model=False, categories= ["background", "capsules", "tablets"]
        )

    dataset_dir = os.path.join(os.getcwd(), dataset_path)
    train(model, weights, dataset_dir)
        
if __name__ == "__main__":
    from faster_rcnn import create_model

    # Use the last trained model and train on a specific dataset stored in dataset path
    # True if using last trained model, False if using pretrained COCO model
    start_training(True, "best_faster_rcnn.pt", "dataset\\pill_detection.v3i.coco")