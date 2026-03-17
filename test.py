import faster_rcnn
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Test on a single image
def single_test(device=None, model_path="faster_rcnn_pills.pt", image_path="test_pill.jpg"):
    # Setup model
    # Make sure to set pretrained=False and coco_model=False to load the custom trained model
    model, weights, categories = faster_rcnn.create_model(
            num_classes=2, pretrained=False, coco_model=False, categories= ["background", "pill"]
        )

    model.load_state_dict(torch.load(model_path, weights_only=True))

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
    threshold = 0.7
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
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        # Label text
        label_name = categories[label.item()]

        ax.text(
            x1,
            y1 - 5,
            f"{label_name}: {score:.2f}",
            color="white",
            fontsize=10,
            backgroundcolor="red"
        )

    plt.axis("off")
    plt.show()

# Metric testing on test set for later
def full_test(device=None, model_path="faster_rcnn_pills.pt"):
    # Test on all images in the test set
    # Return metrics like mAP, precision, recall, etc.

    model, weights, categories = faster_rcnn.create_model(
            num_classes=2, pretrained=False, coco_model=False, categories= ["background", "pill"]
        )

    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.to(device)
    model.eval()

    # Load image
    img = Image.open("test_pill.jpg").convert("RGB")
    preprocess = weights.transforms()

    # Convert image to tensor
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Image inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    threshold = 0.7
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
            linewidth=2,
            edgecolor="blue",
            facecolor="none"
        )
        ax.add_patch(rect)

        # Label text
        label_name = categories[label.item()]

        ax.text(
            x1,
            y1 - 5,
            f"{label_name}: {score:.2f}",
            color="white",
            fontsize=12,
            backgroundcolor="blue"
        )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_test(device, "faster_rcnn_pills.pt", "test_pill.jpg")
    single_test(device, "faster_rcnn_pills.pt", "dog_original.png")