from torch.utils.data import DataLoader
from dataset import CustomDataset
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
from torch.amp import GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from matplotlib import pyplot as plt
from torchvision.ops import box_iou

def collate_fn(batch):
    return tuple(zip(*batch))

def get_f1_score(preds, targets, iou_threshold=0.5):
    # IOU is intersection over union, a common metric for object detection
    # Basically, if bounding boxes overlap enough (IOU > threshold), we consider it a true positive
    tp, fp, fn = 0, 0, 0

    for pred, tgt in zip(preds, targets):
        if len(pred["boxes"]) == 0:
            fn += len(tgt["boxes"])
            continue
        if len(tgt["boxes"]) == 0:
            fp += len(pred["boxes"])
            continue

        ious = box_iou(pred["boxes"], tgt["boxes"])
        # Condition to match
        matches = ious > iou_threshold

        matched_gt = set()

        # If prediction matches ground truth, then TP
        # If prediction doesn't match any ground truth (box or label), then FP
        # If ground truth doesn't match any prediction, then FN
        for i in range(matches.size(0)):
            match_found = False
            for j in range(matches.size(1)):
                # Check bounding box and labels to see if they match
                if matches[i, j] and j not in matched_gt and pred["labels"][i] == tgt["labels"][j]:
                    tp += 1
                    matched_gt.add(j)
                    match_found = True

                    break
            if not match_found:
                fp += 1

        fn += len(tgt["boxes"]) - len(matched_gt)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    # Similar to micro F1-score, but we add a small epsilon to avoid division by zero
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1

def train(model, weights, dataset_dir):
    num_epochs = 40
    batch_size = 16
    lr = 0.001
    train_losses, val_losses = [], [] # loss
    val_scores = [] # f1-score
    map_scores = [] # mAP

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

    # For early stopping
    patience = 5
    epochs_no_improve = 0

    # Training
    for epoch in range(num_epochs):
        print(f"{"*" * 20}Starting Epoch {epoch+1}/{num_epochs}{"*" * 20}")

        # Unfreeze backbone after 3 epochs
        if epoch == 3:
            for param in model.backbone.parameters():
                param.requires_grad = True

        model.train()
        training_loss = 0.0

        # Iterate once per batch in training set
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

            training_loss += losses.item()

        train_loss = training_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_gts, all_preds = [], []

        with torch.no_grad():
            print("Starting Validation...")
            with torch.autocast(device_type='cuda'):
                for images, targets in val_loader:
                    images = [img.to(device, non_blocking=True) for img in images]
                    targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]

                    # Loss
                    model.train() # Set to train mode to compute loss
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    val_loss += loss.item()

                    # Predictions
                    model.eval() # Set to eval mode for inference
                    outputs = model(images)

                    preds = [{k: v.to("cpu") for k, v in out.items()} for out in outputs]
                    gt = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

                    all_preds.extend(preds)
                    all_gts.extend(gt)

                    metric.update(preds, gt)

        # Metrics
        val_losses.append(val_loss / len(val_loader))

        f1 = get_f1_score(all_preds, all_gts)
        val_scores.append(f1)

        map_results = metric.compute()
        curr_map = map_results['map'].item()
        map_scores.append(curr_map)

        print(f"{"*" * 10}Epoch [{epoch+1}/{num_epochs}] metrics{"*" * 10}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_losses[-1]:.4f}")
        print(f"Validation F1-Score: {val_scores[-1]:.4f}")

        print(f"mAP: {curr_map:.4f}")
        print(f"mAP@50: {map_results['map_50'].item():.4f}")

        if curr_map > best_map:
            best_map = curr_map
            torch.save(model.state_dict(), "best_faster_rcnn.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Stopping early since no improvement was observed")
            break
        
        # Reset metric for next epoch
        metric.reset()
        scheduler.step()
    
    # Plot Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss per Epoch")
    # Plot F1 scores
    plt.subplot(1, 3, 2)
    plt.plot(val_scores, label="Validation F1-Score")
    plt.legend()
    plt.title("F1-Scores per Epoch")
    # Plot mAP scores
    plt.subplot(1, 3, 3)
    plt.plot(map_scores, label="Validation mAP")
    plt.legend()
    plt.title("mAP per Epoch")

    plt.tight_layout()
    plt.savefig("overall_metrics.png")
    plt.show()

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