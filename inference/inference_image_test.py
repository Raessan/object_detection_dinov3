import torch
import numpy as np
from src.model_head import DinoFCOSHead
from src.model_backbone import DinoBackbone
from src.dataset_coco import DatasetCOCO
from src.utils import resize_transform, image_to_tensor, tensor_to_image, decode_outputs, plot_detections, build_outputs_from_targets
import config.config as cfg
import cv2
import sys
import time
from src.loss import create_ground_truth_targets, compute_loss

COCO_ROOT = cfg.COCO_ROOT
IMG_SIZE = 768
PATCH_SIZE = cfg.PATCH_SIZE
IMG_MEAN = np.array(cfg.IMG_MEAN, dtype=np.float32)[:, None, None]
IMG_STD = np.array(cfg.IMG_STD, dtype=np.float32)[:, None, None]
FPN_CH = cfg.FPN_CH
DINOV3_DIR = cfg.DINOV3_DIR
DINO_MODEL = cfg.DINO_MODEL
MODEL_TO_NUM_LAYERS = cfg.MODEL_TO_NUM_LAYERS
MODEL_TO_EMBED_DIM = cfg.MODEL_TO_EMBED_DIM
MODEL_PATH_INFERENCE = cfg.MODEL_PATH_INFERENCE
IMG_INFERENCE_PATH = cfg.IMG_INFERENCE_PATH
NUM_CLASSES = 80

# Get class names from COCO
val_set = DatasetCOCO(COCO_ROOT, "val", IMG_SIZE, PATCH_SIZE, 0.0)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_layers_dino = MODEL_TO_NUM_LAYERS[DINO_MODEL]
dino_model = torch.hub.load(
        repo_or_dir=DINOV3_DIR,
        model=DINO_MODEL,
        source="local"
)
dino_backbone = DinoBackbone(dino_model, n_layers_dino).to(device)

embed_dim = MODEL_TO_EMBED_DIM[DINO_MODEL]
model_head = DinoFCOSHead(backbone_out_channels=embed_dim, fpn_channels=FPN_CH, num_classes=NUM_CLASSES).to(device)
model_head.load_state_dict(torch.load(MODEL_PATH_INFERENCE))

data = val_set.__getitem__(0)

image, boxes, labels = data

image = image.unsqueeze(0).to(device, dtype=torch.float)
boxes = [boxes.to(device, dtype=torch.float)]
labels = [labels.to(device, dtype=torch.int)]

# Simulate displacement of boxes
boxes_simul = [box.clone() for box in boxes]
boxes_simul[0][:, 0] += 0.01
# rows_to_keep = range(5)
# boxes_simul[0] = boxes_simul[0][rows_to_keep]
# print(boxes_simul[0].shape)

# Inference
dino_backbone.eval()
model_head.eval()

with torch.no_grad():
    feat = dino_backbone(image)
    outputs = model_head(feat)

first_stride = IMG_SIZE / outputs['cls'][0].shape[2]
strides = [first_stride, first_stride*2, first_stride*4]

# Probar a pasar el boxes por el loss y ver si es 0
cls_targets, reg_targets, ctr_targets, pos_masks = create_ground_truth_targets(boxes_simul, labels, image.shape[2:], 
                                                                               [t for t in outputs['cls']], strides)
outputs_logits = build_outputs_from_targets(cls_targets, reg_targets, ctr_targets, eps=1e-4)
loss = compute_loss(outputs_logits, boxes, labels,
                   image.shape[2:], strides=strides, focal_alpha=0.25, focal_gamma=2.0,
                   weight_reg=1.0, weight_ctr=1.0)
print("Cls loss :", loss[1])
print("Reg loss :", loss[2])
print("Ctr loss :", loss[3])

boxes_plot, scores_plot, labels_plot = decode_outputs(outputs_logits, image.shape[2:], strides, score_thresh=0.1)
print(boxes_plot.shape)
plot_detections(tensor_to_image(image, IMG_MEAN, IMG_STD), boxes_plot.cpu(), scores_plot.cpu(), labels_plot.cpu(), val_set.class_names)
print("End of inference")

