from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

# Referenced from https://github.com/hubert10/fasterrcnn_resnet50_fpn_v2_new_dataset/tree/main
def create_model(num_classes, pretrained=True, coco_model=False, categories=None):
    # Load Faster RCNN pre-trained model
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights if pretrained else None)

    if categories is None:
        categories = weights.meta["categories"]

    if coco_model: # Return the COCO pretrained model for COCO classes.
        return model, weights, categories
    
    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model, weights, categories