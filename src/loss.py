import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
import time

def compute_locations(feature: torch.Tensor, stride: int):
    """Return a tensor of shape (H, W, 2) with the center (x,y) coordinates (in pixels)
    of each feature cell, assuming image coords start at (0,0) in the top-left.
    stride: how many input pixels each feature cell corresponds to.
    """
    _, _, H, W = feature.shape
    shifts_x = (torch.arange(0, W, device=feature.device) + 0.5) * stride
    shifts_y = (torch.arange(0, H, device=feature.device) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    locations = torch.stack((shift_x, shift_y), dim=-1)  # (H, W, 2)
    return locations


def create_ground_truth_targets(batch_boxes, batch_labels, image_size, feature_maps, strides,
                                center_sampling_radius: float = 1.5, num_classes=80, scale_ranges = None):
    """
    Create per-level training targets for an FCOS-style head.

    Args:
      batch_boxes: list of length B, each is tensor (N_i, 4) in relative coords (xmin, ymin, w, h) in [0,1]
      batch_labels: list of length B, each is tensor (N_i,) with class indices in [0, num_classes-1]
      image_size: (height, width) in pixels
      feature_maps: list of feature tensors [P3, P4, P5] (only shapes are used)
      strides: list of ints, one per feature level, e.g. [8, 16, 32]
      center_sampling_radius: multiplier for the center sampling radius used by FCOS

    Returns:
      cls_targets: list per level of tensors (B, num_classes, H, W) with 0/1 targets (sigmoid-style)
      reg_targets: list per level of tensors (B, 4, H, W) with l,t,r,b distances in pixels
      ctr_targets: list per level of tensors (B, 1, H, W) with centerness targets in [0,1]
      pos_mask: list per level of tensors (B, H, W) boolean indicating positive locations
    """
    B = len(batch_boxes)
    img_h, img_w = image_size
    device = feature_maps[0].device

    num_levels = len(feature_maps)

    # prepare target containers
    cls_targets = []
    reg_targets = []
    ctr_targets = []
    pos_masks = []

    # Precompute location grids per level
    locations_per_level = [compute_locations(feature_maps[l], strides[l]) for l in range(num_levels)]

    for l, feature in enumerate(feature_maps):
        _, _, H, W = feature.shape
        cls_targets.append(torch.zeros((B, num_classes, H, W), device=device))
        reg_targets.append(torch.zeros((B, 4, H, W), device=device))
        ctr_targets.append(torch.zeros((B, 1, H, W), device=device))
        pos_masks.append(torch.zeros((B, H, W), dtype=torch.bool, device=device))

    # For each image in the batch, assign GTs to locations
    for b in range(B):
        boxes_rel = batch_boxes[b]
        labels = batch_labels[b]
        if boxes_rel.numel() == 0:
            continue
        # Convert boxes to absolute coords [x1,y1,x2,y2] in pixels
        x1 = boxes_rel[:, 0] * img_w
        y1 = boxes_rel[:, 1] * img_h
        x2 = x1 + boxes_rel[:, 2] * img_w
        y2 = y1 + boxes_rel[:, 3] * img_h
        areas = (x2 - x1) * (y2 - y1)

        for l in range(num_levels):
            locations = locations_per_level[l]  # (H, W, 2)
            H_l, W_l = locations.shape[:2]
            locs = locations.reshape(-1, 2)  # (H*W, 2)

            # Expand to (num_locs, num_gts)
            lx = locs[:, 0].unsqueeze(1)  # (L,1)
            ly = locs[:, 1].unsqueeze(1)

            l_off = lx - x1.unsqueeze(0)  # (L, G)
            t_off = ly - y1.unsqueeze(0)
            r_off = x2.unsqueeze(0) - lx
            b_off = y2.unsqueeze(0) - ly

            # mask for locations inside GT box
            inside_box = (l_off > 0) & (t_off > 0) & (r_off > 0) & (b_off > 0)  # (L, G)

            # center sampling: restrict positives to a radius around GT center (in pixels)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = center_sampling_radius * strides[l]
            dist_x = (lx - cx.unsqueeze(0)).abs()
            dist_y = (ly - cy.unsqueeze(0)).abs()
            center_mask = (dist_x <= radius) & (dist_y <= radius)  # (L, G)

            if scale_ranges:
                # Compute object size (max side in pixels):
                gt_w = x2 - x1
                gt_h = y2 - y1
                gt_size = torch.max(gt_w, gt_h)  # (G,)

                # Build a bool mask size_mask of shape (1, G) indicating which GTs belong to this level
                min_s, max_s = scale_ranges[l]
                size_mask = (gt_size >= min_s) & (gt_size <= max_s)  # (G,)
                # broadcast to (L, G) and incorporate into is_pos:

                if not size_mask.all():
                    # set all columns corresponding to invalid GTs to False
                    inside_box[:, ~size_mask] = False
                    center_mask[:, ~size_mask] = False


            is_pos = inside_box & center_mask

            if is_pos.sum() == 0:
                # fallback: if no center-sampled positives, allow inside_box
                is_pos = inside_box

            # For locations matching multiple GTs, choose GT with smallest area
            # Prepare an (L, G) tensor of areas (broadcasted)
            areas_expand = areas.unsqueeze(0).expand(is_pos.shape).clone()
            areas_expand[~is_pos] = float('inf')
            min_area, min_idx = areas_expand.min(dim=1)  # per-location
            pos_inds = min_area != float('inf')  # (L,)

            if pos_inds.sum() == 0:
                continue

            chosen = min_idx[pos_inds]  # indices of GTs assigned to these locations
            loc_idx = torch.nonzero(pos_inds, as_tuple=False).squeeze(1)  # linear loc indices

            # Fill targets for these locations
            l_vals = l_off[pos_inds, chosen]
            t_vals = t_off[pos_inds, chosen]
            r_vals = r_off[pos_inds, chosen]
            b_vals = b_off[pos_inds, chosen]

            # convert linear loc indices to 2D coords
            ys = (loc_idx // W_l)
            xs = (loc_idx % W_l)

            for k_idx, (yy, xx, gi) in enumerate(zip(ys.tolist(), xs.tolist(), chosen.tolist())):
                cls_targets[l][b, labels[gi], yy, xx] = 1.0
                reg_targets[l][b, 0, yy, xx] = l_vals[k_idx]
                reg_targets[l][b, 1, yy, xx] = t_vals[k_idx]
                reg_targets[l][b, 2, yy, xx] = r_vals[k_idx]
                reg_targets[l][b, 3, yy, xx] = b_vals[k_idx]
                # centerness target
                left = l_vals[k_idx]
                right = r_vals[k_idx]
                top = t_vals[k_idx]
                bottom = b_vals[k_idx]
                ctr = torch.sqrt((torch.min(left, right) / (torch.max(left, right) + 1e-9)) *
                                 (torch.min(top, bottom) / (torch.max(top, bottom) + 1e-9)))
                ctr_targets[l][b, 0, yy, xx] = ctr
                pos_masks[l][b, yy, xx] = True

    return cls_targets, reg_targets, ctr_targets, pos_masks


def focal_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
    """Compute focal loss (sigmoid) from logits and targets.
    logits & targets shape: arbitrary but broadcastable. targets in {0,1}.
    Returns scalar loss (sum over elements).
    """
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss = alpha_factor * modulating_factor * ce_loss
    return loss.sum()


def giou_loss(pred_boxes, target_boxes):
    """Generalized IoU loss for boxes provided as [x1,y1,x2,y2]
    pred_boxes and target_boxes are (N,4)
    returns sum of (1 - giou) over N
    """
    # intersection
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_tgt = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * \
               (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

    union = area_pred + area_tgt - inter
    iou = inter / (union + 1e-7)

    # convex hull
    cx1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    cy1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    cx2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    cy2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)

    giou = iou - (c_area - union) / (c_area + 1e-7)
    loss = (1 - giou).sum()
    return loss


def compute_loss(outputs: Dict[str, List[torch.Tensor]], batch_boxes, batch_labels,
                   image_size, strides=[16,32,64], focal_alpha=0.25, focal_gamma=2.0,
                   weight_reg=1.0, weight_ctr=1.0, scale_ranges = None):
    """
    Given model outputs (dict of lists per level) and batch GTs, compute
    classification (focal), regression (GIoU), and centerness losses.

    Args:
      outputs: dict with keys 'cls','reg','ctr' each a list per level of tensors
      batch_boxes: list of length B, each (Ni,4) rel coords (xmin,ymin,w,h)
      batch_labels: list of length B, each (Ni,) ints
      image_size: (H,W)
      strides: list of ints per level

    Returns: dict with losses { 'cls':, 'reg':, 'ctr':, 'total': }
    """
    device = outputs['cls'][0].device
    B = outputs['cls'][0].shape[0]

    # Create targets
    cls_targets, reg_targets, ctr_targets, pos_masks = create_ground_truth_targets(
        batch_boxes, batch_labels, image_size,
        [t for t in outputs['cls']], strides, num_classes=outputs['cls'][0].shape[1], scale_ranges=scale_ranges)

    # Flatten predictions and targets across levels for easier loss computation
    flatten_logits = []
    flatten_targets = []
    flatten_reg_preds = []
    flatten_reg_tgts = []
    flatten_ctr_preds = []
    flatten_ctr_tgts = []

    for l in range(len(outputs['cls'])):
        cls_l = outputs['cls'][l]
        reg_l = outputs['reg'][l]
        ctr_l = outputs['ctr'][l]

        B, C, H, W = cls_l.shape
        cls_l_flat = cls_l.view(B, C, -1).permute(0,2,1).reshape(-1, C)  # (B*L, C)
        tgt_flat = cls_targets[l].view(B, C, -1).permute(0,2,1).reshape(-1, C)  # (B*L, C)

        reg_l_flat = reg_l.view(B, 4, -1).permute(0,2,1).reshape(-1,4)  # (B*L,4)
        reg_tgt_flat = reg_targets[l].view(B,4,-1).permute(0,2,1).reshape(-1,4)

        ctr_l_flat = ctr_l.view(B,1,-1).permute(0,2,1).reshape(-1,1)
        ctr_tgt_flat = ctr_targets[l].view(B,1,-1).permute(0,2,1).reshape(-1,1)

        flatten_logits.append(cls_l_flat)
        flatten_targets.append(tgt_flat)
        flatten_reg_preds.append(reg_l_flat)
        flatten_reg_tgts.append(reg_tgt_flat)
        flatten_ctr_preds.append(ctr_l_flat)
        flatten_ctr_tgts.append(ctr_tgt_flat)

    all_logits = torch.cat(flatten_logits, dim=0)
    all_targets = torch.cat(flatten_targets, dim=0)
    all_reg_preds = torch.cat(flatten_reg_preds, dim=0)
    all_reg_tgts = torch.cat(flatten_reg_tgts, dim=0)
    all_ctr_preds = torch.cat(flatten_ctr_preds, dim=0)
    all_ctr_tgts = torch.cat(flatten_ctr_tgts, dim=0)

    # positive locations mask across all levels
    pos_mask_all = torch.cat([m.view(B, -1) for m in pos_masks], dim=1).reshape(-1)  # (B*sum(L),)
    num_pos = pos_mask_all.sum().float().clamp(min=1.0)

    # Classification focal loss (sigmoid)
    cls_loss = focal_loss_from_logits(all_logits, all_targets, alpha=focal_alpha, gamma=focal_gamma)
    cls_loss = cls_loss / num_pos

    # Regression loss: compute predicted boxes and target boxes for positives only
    if num_pos > 0:
        pos_idx = torch.nonzero(pos_mask_all, as_tuple=False).squeeze(1)
        pred_reg_pos = all_reg_preds[pos_idx]  # (P,4)
        tgt_reg_pos = all_reg_tgts[pos_idx]

        # We need centers for each positive location to convert ltrb -> x1y1x2y2
        # Reconstruct centers by iterating feature maps
        centers = []
        for l_idx, feat in enumerate(outputs['reg']):
            _, _, H, W = feat.shape
            stride = strides[l_idx]
            locs = compute_locations(feat, stride).reshape(-1,2)
            centers.append(locs)
        centers_all = torch.cat(centers, dim=0)  # (total_locs, 2)
        centers_all = centers_all.repeat(B,1,1).reshape(-1,2)  # (B*total_locs, 2)
        centers_pos = centers_all[pos_idx]

        # pred boxes
        x = centers_pos[:, 0]
        y = centers_pos[:, 1]
        l = pred_reg_pos[:,0]; t = pred_reg_pos[:,1]; r = pred_reg_pos[:,2]; b = pred_reg_pos[:,3]
        pred_boxes_xy = torch.stack([x - l, y - t, x + r, y + b], dim=1)

        lt = tgt_reg_pos[:,0]; tt = tgt_reg_pos[:,1]; rt = tgt_reg_pos[:,2]; bt = tgt_reg_pos[:,3]
        tgt_boxes_xy = torch.stack([centers_pos[:,0] - lt, centers_pos[:,1] - tt,
                                    centers_pos[:,0] + rt, centers_pos[:,1] + bt], dim=1)

        reg_loss = giou_loss(pred_boxes_xy, tgt_boxes_xy) / num_pos

        # Centerness loss (BCE with logits) only where pos
        ctr_loss = F.binary_cross_entropy_with_logits(all_ctr_preds[pos_idx], all_ctr_tgts[pos_idx], reduction="sum") / num_pos
    else:
        reg_loss = torch.tensor(0.0, device=device)
        ctr_loss = torch.tensor(0.0, device=device)

    total = cls_loss + reg_loss*weight_reg + ctr_loss*weight_ctr
    return total, cls_loss, reg_loss, ctr_loss



# ----------------- Quick smoke test / example -----------------
if __name__ == '__main__':
    
    from dataset_coco import DatasetCOCO
    from model_backbone import DinoBackbone
    from model_head import DinoFCOSHead

    COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO'
    MODE = "val"
    IMG_SIZE = 640
    PATCH_SIZE = 16
    DINOV3_DIR = "/home/rafa/deep_learning/projects/object_detection_dinov3/dinov3"
    N_LAYERS_DINO = 12
    FPN_CH = 192

    dataset = DatasetCOCO(COCO_ROOT, MODE, IMG_SIZE, PATCH_SIZE)
    data = dataset.__getitem__(0)

    image, boxes, labels = data

    image = image.unsqueeze(0)
    boxes = boxes.unsqueeze(0)
    labels = labels.unsqueeze(0)

    dino_model = torch.hub.load(
        repo_or_dir=DINOV3_DIR,
        model="dinov3_vits16plus",
        source="local"
    )

    dino_backbone = DinoBackbone(dino_model, N_LAYERS_DINO)
    feat = dino_backbone(image)
    
    num_classes = len(dataset.class_names)
    full_head = DinoFCOSHead(backbone_out_channels=feat.shape[1], fpn_channels=FPN_CH, num_classes=num_classes)

    # Forward pass
    outputs = full_head(feat)
    
    first_stride = IMG_SIZE / outputs['cls'][0].shape[2]
    strides = [first_stride, first_stride*2, first_stride*4]

    losses = compute_loss(outputs, boxes, labels, image.shape[2:], strides=strides)
    print(losses)
