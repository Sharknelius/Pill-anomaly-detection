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
            num_classes=5, pretrained=False, coco_model=False, categories= ["background", "capsule", "damaged-pill", "foreign-matter", "tablet"]
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
    # Threshold for detection
    threshold = 0.15
    # Threshold for pill classification
    true_threshold = 0.8
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

        # Label as known category or unknown based on confidence score
        if score > true_threshold:
            label_name = categories[label.item()]
        else:
            label_name = "unknown-anomaly"

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

def basic_test(model_name):
    single_test(device, model_name, "test_pill.jpg") # Update these images with your own test images
    single_test(device, model_name, "broken_pill.png")
    single_test(device, model_name, "capsules.jpg")
    single_test(device, model_name, "broken_tablet.jpg")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    basic_test("2026-04-18_15-12-48_best_frcnn.pt")