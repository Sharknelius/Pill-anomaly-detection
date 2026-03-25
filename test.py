import faster_rcnn
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import os
from torch.utils.data import DataLoader
from dataset import CustomDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision.ops import box_iou

#custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))

def match_predictions_to_targets(
    pred_boxes,
    pred_labels,
    pred_scores,
    true_boxes,
    true_labels,
    score_threshold=0.5,
    iou_threshold=0.5,
    background_label=0
):
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]

    matched_true = set()
    y_true = []
    y_pred = []

    if len(pred_boxes) > 0 and len(true_boxes) > 0:
        ious = box_iou(pred_boxes, true_boxes)

        for pred_idx in range(len(pred_boxes)):
            best_iou, best_true_idx = torch.max(ious[pred_idx], dim=0)

            best_iou = best_iou.item()
            best_true_idx = best_true_idx.item()

            if best_iou >= iou_threshold and best_true_idx not in matched_true:
                matched_true.add(best_true_idx)
                y_true.append(int(true_labels[best_true_idx].item()))
                y_pred.append(int(pred_labels[pred_idx].item()))
            else:
                # false positive
                y_true.append(background_label)
                y_pred.append(int(pred_labels[pred_idx].item()))
    else:
        for pred_idx in range(len(pred_boxes)):
            y_true.append(background_label)
            y_pred.append(int(pred_labels[pred_idx].item()))

    # unmatched true boxes become false negatives
    for true_idx in range(len(true_boxes)):
        if true_idx not in matched_true:
            y_true.append(int(true_labels[true_idx].item()))
            y_pred.append(background_label)

    return y_true, y_pred

# Test on a single image
def single_test(device=None, model_path="faster_rcnn_pills.pt", image_path="test_pill.jpg"):
    # Setup model
    # Make sure to set pretrained=False and coco_model=False to load the custom trained model
    
    #gave me errors if i didnt have this - Thomas
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, weights, categories = faster_rcnn.create_model(
            num_classes=3, pretrained=False, coco_model=False, categories= ["background", "capsules", "tablets"]
        )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.to(device)
    model.eval()

    # Load image
    img = Image.open(image_path).convert("RGB")
    preprocess = weights.transforms()
    # Convert image to tensor
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Image inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Confidence thresholding (the higher, the pickier the model is)
    # Possibly less recall but higher precision if raised
    # Threshold for detection
    threshold = 0.1
    # Threshold for pill classification
    true_threshold = 0.97
    keep = predictions["scores"] >= threshold
    # Convert tensors to CPU for visualization
    boxes = predictions["boxes"][keep].cpu()
    labels = predictions["labels"][keep].cpu()
    scores = predictions["scores"][keep].cpu()

    # Plot image
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor="blue",
            facecolor="none"
        )
        ax.add_patch(rect)

        # Label as known category or anomaly based on confidence score
        if score > true_threshold:
            label_name = categories[label.item()]
        else:
            label_name = "anomaly"

        ax.text(
            x1,
            y1 - 5,
            f"{label_name}: {score:.2f}",
            color="white",
            fontsize=10,
            backgroundcolor="blue"
        )
    plt.axis("off")
    plt.show()

def quick_test_metrics(device=None, model_path="best_faster_rcnn.pt"):
    #test cases (image, expected label)
    test_cases = [
        ("test_pill.jpg", "tablets"),
        ("broken_pill.png", "tablets"),
        ("capsules.jpg", "capsules"),
        ("dog_original.png", "anomaly")
    ]

    #gave me errors if i didnt have this - Thomas
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, weights, categories = faster_rcnn.create_model(
        num_classes=3,
        pretrained=False,
        coco_model=False,
        categories=["background", "capsules", "tablets"]
    )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    all_y_true = []
    all_y_pred = []

    for image_path, true_label in test_cases:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # If no detections → anomaly
        if len(prediction["scores"]) == 0:
            pred_label = "anomaly"
        else:
            best_idx = torch.argmax(prediction["scores"])
            score = prediction["scores"][best_idx].item()

            # Confidence threshold
            if score < 0.5:
                pred_label = "anomaly"
            else:
                pred_class = prediction["labels"][best_idx].item()
                pred_label = categories[pred_class]

        all_y_true.append(true_label)
        all_y_pred.append(pred_label)

    # Compute metrics
    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)

    print("Quick Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

# Metric testing on test set for later
def full_test(device=None, model_path="faster_rcnn_pills.pt", dataset_path="dataset\\pill_detection.v3i.coco"):
    # Test on all images in the test set
    # Return metrics like mAP, precision, recall, etc.

    #gave me errors if i didnt have this - Thomas
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, weights, categories = faster_rcnn.create_model(
            num_classes=3, pretrained=False, coco_model=False, categories=["background", "capsules", "tablets"]
        )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.to(device)
    model.eval()

    # Load test dataset instead of one image
    preprocess = weights.transforms()
    test_path = os.path.join(dataset_path, "test")
    test_dataset = CustomDataset(test_path, preprocess)

    # DataLoader for batching
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    # Metric trackers
    metric = MeanAveragePrecision(class_metrics=True)
    all_y_true = []
    all_y_pred = []

    threshold = 0.5
    iou_threshold = 0.5

    # Image inference over full test set
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device, non_blocking=True) for img in images]
            predictions = model(images)

            # Move tensors to CPU
            preds = [{k: v.to("cpu") for k, v in out.items()} for out in predictions]
            gt = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

            # Update mAP metric
            metric.update(preds, gt)

            # Convert detection outputs to labels for F1/accuracy
            for pred, true in zip(preds, gt):
                y_true, y_pred = match_predictions_to_targets(
                    pred_boxes=pred["boxes"],
                    pred_labels=pred["labels"],
                    pred_scores=pred["scores"],
                    true_boxes=true["boxes"],
                    true_labels=true["labels"],
                    score_threshold=threshold,
                    iou_threshold=iou_threshold,
                    background_label=0
                )

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

    # Compute final metrics
    results = metric.compute()

    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)

    print(f"Test mAP: {results['map'].item():.4f}")
    print(f"Test mAP@50: {results['map_50'].item():.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision (Macro): {precision:.4f}")
    print(f"Test Recall (Macro): {recall:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_test(device, "best_faster_rcnn.pt", "test_pill.jpg")
    single_test(device, "best_faster_rcnn.pt", "broken_pill.png")
    single_test(device, "best_faster_rcnn.pt", "capsules.jpg")
    single_test(device, "best_faster_rcnn.pt", "dog_original.png")

    quick_test_metrics(device, "best_faster_rcnn.pt")