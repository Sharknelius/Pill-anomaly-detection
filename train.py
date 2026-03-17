from torch.utils.data import DataLoader
from dataset import CustomDataset
import torch
import torch.optim as optim
import os
from torch.amp import GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def collate_fn(batch):
    return tuple(zip(*batch))

def train(model, weights, dataset_dir):

    num_epochs = 2
    batch_size = 4
    lr = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = weights.transforms()

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

    # Stochastic Gradient Descent with Momentum and Weight Decay
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )

    scaler = GradScaler()

    metric = MeanAveragePrecision(class_metrics=True).to(device)

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")

        model.train()
        total_loss = 0

        for images, targets in train_loader:

            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]

            # loss_dict = model(images, targets)

            # losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            #losses.backward()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            # optimizer.step()

            total_loss += losses.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        # print(f"Training Epoch {epoch+1} completed.")

        # Validation
        model.eval()
        #val_loss = 0

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
        print(f"Epoch {epoch+1} mAP: {map_results['map'].item():.4f}")
        print(f"Epoch {epoch+1} mAP@50: {map_results['map_50'].item():.4f}")
        
        # IMPORTANT: Reset the metric for the next epoch
        metric.reset()

if __name__ == "__main__":
    from faster_rcnn import create_model

    # Setup model
    model, weights, categories = create_model(
            num_classes=2, pretrained=True, coco_model=False, categories= ["background", "pill"]
        )

    dataset_dir = os.path.join(os.getcwd(), "dataset\\pill_detection.v3i.coco")
    train(model, weights, dataset_dir)

    torch.save(model.state_dict(), "faster_rcnn_pills.pt")