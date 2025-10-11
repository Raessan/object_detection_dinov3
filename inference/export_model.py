import torch
from src.model_backbone import DinoBackbone
from src.model_head import DinoFCOSHead
import config.config as cfg
from pathlib import Path

IMG_SIZE = cfg.IMG_SIZE
DINOV3_DIR = cfg.DINOV3_DIR
DINO_MODEL = cfg.DINO_MODEL
DINO_WEIGHTS = cfg.DINO_WEIGHTS
MODEL_TO_NUM_LAYERS = cfg.MODEL_TO_NUM_LAYERS
MODEL_TO_EMBED_DIM = cfg.MODEL_TO_EMBED_DIM
FPN_CH = cfg.FPN_CH
MODEL_PATH_INFERENCE = cfg.MODEL_PATH_INFERENCE
CLASS_NAMES_PATH = cfg.CLASS_NAMES_PATH

PATH_SAVE_BACKBONE_ONNX = Path(DINO_WEIGHTS).with_suffix(".onnx")
PATH_SAVE_HEAD_ONNX = Path(MODEL_PATH_INFERENCE).with_suffix(".onnx")

def export_model():

    ################## EXPORT BACKBONE ##################

    dino_backbone_loader = torch.hub.load(
        repo_or_dir=DINOV3_DIR,
        model=DINO_MODEL,
        source="local",
        weights=DINO_WEIGHTS
    )

    n_layers_dino = MODEL_TO_NUM_LAYERS[DINO_MODEL]

    # Instantiate the model and load initial weights
    dino_model = DinoBackbone(dino_backbone_loader, n_layers_dino)
    dino_model.eval()

    # Define the input shape
    dummy_image = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # Export backbone model to ONNX
    torch.onnx.export(
        dino_model,
        dummy_image,
        PATH_SAVE_BACKBONE_ONNX,
        input_names=["image"],
        output_names=["features"],
        opset_version=17,
        do_constant_folding=True
    )

    ################## EXPORT HEAD ########################
    # Get class names from COCO
    with open(CLASS_NAMES_PATH) as f:
        class_names = [line.strip() for line in f]

    dummy_features = dino_model(dummy_image)

    model_head = DinoFCOSHead(backbone_out_channels=dummy_features.shape[1], fpn_channels=FPN_CH, num_classes=len(class_names))


    model_head.load_state_dict(torch.load(MODEL_PATH_INFERENCE))
    model_head.eval()

    # Export head model to ONNX
    torch.onnx.export(
        model_head,
        dummy_features,
        PATH_SAVE_HEAD_ONNX,
        input_names=["features"],
        output_names=["boxes", "scores", "labels"],
        opset_version=17,
        do_constant_folding=True
    )


if __name__ == "__main__":
    export_model()