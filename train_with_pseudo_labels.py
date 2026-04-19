from torch.utils.data import DataLoader
from dataset import CustomDataset, MergedDataset
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torchvision.ops import nms
import os
from torch.amp import GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from matplotlib import pyplot as plt
from torchvision.ops import box_iou
from datetime import datetime, timezone
import random

# Constant
MAX_PSEUDO = 1500
CAPSULE_ID = 1
TABLET_ID = 4
NUM_EPOCH = 50
BATCH_SIZE = 16

def collate_fn(batch):
    images, targets = zip(*batch)

    if targets[0] is None:
        return list(images), None

    return list(images), list(targets)

def data_loader(dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True
        )

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

def predict_with_aug(model, images):
    all_outputs = []
    
    for img in images:
        preds_list = []
        
        # Original image
        out = model([img])[0]
        preds_list.append(out)
        
        # Horizontal flip
        flipped = torch.flip(img, [-1])
        out_f = model([flipped])[0]

        # Mirror boxes back
        w = img.shape[-1]
        boxes_f = out_f["boxes"].clone()
        boxes_f[:, [0, 2]] = w - out_f["boxes"][:, [2, 0]]
        preds_list.append({**out_f, "boxes": boxes_f})
        
        # Merge with NMS
        boxes = torch.cat([p["boxes"] for p in preds_list])
        scores = torch.cat([p["scores"] for p in preds_list])
        labels = torch.cat([p["labels"] for p in preds_list])
        keep = nms(boxes, scores, iou_threshold=0.5)
        
        all_outputs.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
    
    return all_outputs

def train(model, weights, labeled_dataset_dir, unlabeled_dataset_dir, new_name, labeled=True):
    num_epochs = NUM_EPOCH
    batch_size = BATCH_SIZE
    lr = 0.0005
    train_losses, val_losses = [], [] # loss
    val_scores = [] # f1-score
    map_scores = [] # mAP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = weights.transforms()

    # Find and load labeled training and validation datasets
    train_path = os.path.join(labeled_dataset_dir, "train")
    val_path = os.path.join(labeled_dataset_dir, "valid")

    train_dataset = CustomDataset(train_path, transforms, True)
    val_dataset = CustomDataset(val_path, transforms, True)

    train_loader = data_loader(train_dataset, batch_size, True)
    print("Labeled training set loaded successfully.")

    val_loader = data_loader(val_dataset, batch_size, False)
    print("Labeled validation set loaded successfully.")

    # Load unlabeled dataset
    if not labeled:
        train_path = os.path.join(unlabeled_dataset_dir, "train")
        val_path = os.path.join(unlabeled_dataset_dir, "valid")

        train_unlabeled_dataset = CustomDataset(train_path, transforms, False)

        unlabeled_train_loader = data_loader(train_unlabeled_dataset, batch_size, False)
        print("Unlabeled training set loaded successfully.")

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
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Track mAP for each epoch and save the best model
    metric = MeanAveragePrecision(class_metrics=True).to(device)
    best_map = 0.0

    # For early stopping
    patience = 10
    epochs_no_improve = 0

    # Training
    for epoch in range(num_epochs):
        print(f"{"*" * 20} Starting Epoch [{epoch+1}/{num_epochs}] {"*" * 20}")

        # Unfreeze backbone after 3 epochs
        if epoch == 3:
            for param in model.backbone.parameters():
                param.requires_grad = True
        
        # Start pseudo labeling
        if epoch >= 9 and epoch % 3 == 0 and not labeled:
            print("Generating pseudo-labels...")

            pseudo_threshold = max(0.8, 0.95 - 0.02 * epoch) # Dynamic threshold, updated from low 0.7 to 0.8

            # Generate pseudo labels
            pseudo_data = psuedo_label(model, unlabeled_train_loader, device, epoch, pseudo_threshold)
            if len(pseudo_data) > MAX_PSEUDO: # Keep pseudo labels from overpowering data
                pseudo_data = random.sample(pseudo_data, MAX_PSEUDO)

            print(f"Generated {len(pseudo_data)} pseudo-labeled samples")

            # Load merged training datasets (labeled + unlabeled)
            merged_dataset = MergedDataset(train_dataset, pseudo_data)
            train_loader = data_loader(merged_dataset, batch_size, True)

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
            # Gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            training_loss += losses.item()

        train_loss = training_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        val_loss = 0.0
        all_gts, all_preds = [], []

        print("Starting Validation...")
        # To avoid slipping modes, keep them separate for loss and inference
        model.train() # Set to train mode to compute loss
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]

                # Loss
                with torch.autocast(device_type='cuda'):
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())

                val_loss += loss.item()

        model.eval() # Set to eval mode for inference
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]

                # Predictions
                outputs = predict_with_aug(model, images) # Add augmentations to validation set
                # Filter outputs for confidence filtered f1-score
                filtered_outputs = []
                for out in outputs:
                    keep = out["scores"] > 0.3
                    filtered_outputs.append({
                        "boxes": out["boxes"][keep],
                        "labels": out["labels"][keep],
                        "scores": out["scores"][keep],
                    })

                preds = [{k: v.to("cpu") for k, v in out.items()} for out in filtered_outputs]
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

        print(f"{"*" * 5} Epoch [{epoch+1}/{num_epochs}] metrics {"*" * 5}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_losses[-1]:.4f}")
        print(f"Validation F1-Score: {val_scores[-1]:.4f}")

        print(f"mAP: {curr_map:.4f}")
        print(f"mAP@50: {map_results['map_50'].item():.4f}")

        if curr_map > best_map:
            best_map = curr_map
            torch.save(model.state_dict(), new_name) # Update best model name as needed
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Stopping early since no improvement was observed")
            break
        
        # Reset metric for next epoch
        metric.reset()
        scheduler.step()

    plot_metrics(train_losses, val_losses, val_scores, map_scores, new_name)

def plot_metrics(train_losses, val_losses, val_scores, map_scores, new_name):
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
    plt.savefig(f"{new_name}_overall_metrics.png")
    plt.show()

# Pseudo labeling function
def psuedo_label(model, unlabeled_loader, device, epoch, threshold=0.9):
    model.eval()
    pseudo_data = []

    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = [img.to(device) for img in images]

            outputs = model(images)

            for img, out in zip(images, outputs):
                if epoch >= 15:
                    keep = out["scores"] > threshold
                else: # Do not label specific anomalies yet
                    valid_labels = torch.tensor([CAPSULE_ID, TABLET_ID], device=out["labels"].device)
                    keep = (out["scores"] > threshold) & torch.isin(out["labels"], valid_labels)

                if keep.sum() == 0:
                    continue  # Skip empty predictions

                boxes = out["boxes"][keep]
                scores = out["scores"][keep]

                keep_idx = nms(boxes, scores, iou_threshold=0.5)

                pseudo_target = {
                    "boxes": out["boxes"][keep_idx].detach().cpu(),
                    "labels": out["labels"][keep_idx].detach().cpu()
                }

                pseudo_data.append((img.cpu(), pseudo_target))

    return pseudo_data

def start_training(load_custom, model_path=None, labeled_dataset_path=None, unlabel_dataset_path=None, new_name=None, not_semi=False):
    if load_custom: # Load the last trained model and continue training
        model, weights, categories = create_model(
            num_classes=5, pretrained=False, coco_model=False, categories= ["background", "capsule", "damaged-pill", "foreign-matter", "tablet"]
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else: # Start training from pretrained COCO model
        model, weights, categories = create_model(
            num_classes=5, pretrained=True, coco_model=False, categories= ["background", "capsule", "damaged-pill", "foreign-matter", "tablet"]
        )

    labeled_dataset_dir = os.path.join(os.getcwd(), labeled_dataset_path)
    if not not_semi:
        unlabeled_dataset_dir = os.path.join(os.getcwd(), unlabel_dataset_path)
    else:
        unlabeled_dataset_dir = None

    train(model, weights, labeled_dataset_dir, unlabeled_dataset_dir, new_name, not_semi)
        
if __name__ == "__main__":
    from faster_rcnn import create_model

    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S") # Get current time for model labeling
    new_model_name = utc_now + "_best_frcnn.pt"
    
    # Use the last trained model and train on a specific dataset stored in dataset path, then save as best_faster_rcnn{n+1}.pt
    # True if using last trained model, False if using pretrained COCO model
    start_training(True, "2026-04-18_15-12-48_best_frcnn.pt", "dataset\\Labeled_anomaly_pill.v2i.coco", "dataset\\Unlabeled_full_pill.v1i.coco", new_model_name, not_semi=False)