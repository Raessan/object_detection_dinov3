import torch
import numpy as np
from src.model_head import DinoFCOSHead
from src.model_backbone import DinoBackbone
from src.dataset_coco import DatasetCOCO
from src.utils import resize_transform, image_to_tensor, tensor_to_image, decode_outputs, plot_detections
import config.config as cfg
import cv2
import sys
import time

COCO_ROOT = cfg.COCO_ROOT
IMG_SIZE = cfg.IMG_SIZE
PATCH_SIZE = cfg.PATCH_SIZE
IMG_MEAN = np.array(cfg.IMG_MEAN, dtype=np.float32)[:, None, None]
IMG_STD = np.array(cfg.IMG_STD, dtype=np.float32)[:, None, None]
FPN_CH = cfg.FPN_CH
DINOV3_DIR = cfg.DINOV3_DIR
DINO_MODEL = cfg.DINO_MODEL
DINO_WEIGHTS = cfg.DINO_WEIGHTS
MODEL_TO_NUM_LAYERS = cfg.MODEL_TO_NUM_LAYERS
MODEL_TO_EMBED_DIM = cfg.MODEL_TO_EMBED_DIM
MODEL_PATH_INFERENCE = cfg.MODEL_PATH_INFERENCE
IMG_INFERENCE_PATH = cfg.IMG_INFERENCE_PATH
NUM_CLASSES = 80

# Get class names from COCO
val_set = DatasetCOCO(COCO_ROOT, "val", IMG_SIZE, PATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_layers_dino = MODEL_TO_NUM_LAYERS[DINO_MODEL]
dino_model = torch.hub.load(
        repo_or_dir=DINOV3_DIR,
        model=DINO_MODEL,
        source="local",
        weights=DINO_WEIGHTS
)
dino_backbone = DinoBackbone(dino_model, n_layers_dino).to(device)

embed_dim = MODEL_TO_EMBED_DIM[DINO_MODEL]
model_head = DinoFCOSHead(backbone_out_channels=embed_dim, fpn_channels=FPN_CH, num_classes=NUM_CLASSES).to(device)
model_head.load_state_dict(torch.load(MODEL_PATH_INFERENCE))

image = cv2.imread(IMG_INFERENCE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = resize_transform(image, IMG_SIZE, PATCH_SIZE)
image_tensor = image_to_tensor(image, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)

# Inference
dino_backbone.eval()
model_head.eval()

with torch.no_grad():
    feat = dino_backbone(image_tensor)
    outputs = model_head(feat)

first_stride = IMG_SIZE / outputs['cls'][0].shape[2]
strides = [first_stride, first_stride*2, first_stride*4]

boxes, scores, labels = decode_outputs(outputs, image_tensor.shape[2:], strides, score_thresh=0.2, nms_thresh = 0.6)
plot_detections(image, boxes.cpu(), scores.cpu(), labels.cpu(), val_set.class_names)
print("End of inference")

