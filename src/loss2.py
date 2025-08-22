import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms  # used in inference


# ---------- utilities ----------
def boxes_xywh_to_xyxy(boxes):
    # boxes: (N,4) xmin,ymin,w,h -> x1,y1,x2,y2
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    return torch.stack([x1,y1,x2,y2], dim=-1)

def xyxy_to_xywh(boxes):
    x1,y1,x2,y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2-x1, y2-y1], dim=-1)

def compute_iou(boxes1, boxes2):
    # boxes1: (N,4), boxes2: (N,4) -> elementwise IoU
    x1 = torch.max(boxes1[:,0], boxes2[:,0])
    y1 = torch.max(boxes1[:,1], boxes2[:,1])
    x2 = torch.min(boxes1[:,2], boxes2[:,2])
    y2 = torch.min(boxes1[:,3], boxes2[:,3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:,2] - boxes1[:,0]).clamp(min=0) * (boxes1[:,3] - boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2] - boxes2[:,0]).clamp(min=0) * (boxes2[:,3] - boxes2[:,1]).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    return iou

def create_ground_truth_targets(batch_boxes, batch_labels, image_size, feature_maps, strides,
                                center_sampling_radius: float = 1.5, num_classes=80):
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
    device = feature_maps[0].device
    B = len(batch_boxes)
    H_img, W_img = image_size

    # Convert relative boxes to absolute pixel coords
    batch_boxes_pix = []
    for boxes in batch_boxes:
        if boxes.numel() == 0:
            batch_boxes_pix.append(boxes.to(device))
        else:
            b = boxes.clone().to(device)
            b[:,0] *= W_img
            b[:,1] *= H_img
            b[:,2] *= W_img
            b[:,3] *= H_img
            batch_boxes_pix.append(b)

    cls_targets = []
    reg_targets = []
    ctr_targets = []
    pos_masks = []

    # Loop over pyramid levels
    for lvl, (feat, stride) in enumerate(zip(feature_maps, strides)):
        B_, C, H, W = feat.shape
        assert B_ == B

        # location centers in image space
        shifts_x = (torch.arange(0, W, device=device) + 0.5) * stride
        shifts_y = (torch.arange(0, H, device=device) + 0.5) * stride
        ys, xs = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        locs = torch.stack([xs, ys], dim=-1)   # (Nloc,2)
        Nloc = locs.shape[0]

        # Prepare targets per image in batch
        cls_t = torch.zeros((B, num_classes, H, W), device=device)
        reg_t = torch.zeros((B, 4, H, W), device=device)
        ctr_t = torch.zeros((B, 1, H, W), device=device)
        pos_m = torch.zeros((B, H, W), device=device, dtype=torch.bool)

        for b in range(B):
            boxes = batch_boxes_pix[b]
            labels = batch_labels[b].to(device)

            if boxes.numel() == 0:
                continue

            gt_xyxy = boxes_xywh_to_xyxy(boxes)  # (G,4)
            G = gt_xyxy.shape[0]

            # Expand locations against GTs
            xs_exp = locs[:,0].unsqueeze(1).expand(Nloc, G)
            ys_exp = locs[:,1].unsqueeze(1).expand(Nloc, G)

            x1_gt = gt_xyxy[:,0].unsqueeze(0).expand(Nloc, G)
            y1_gt = gt_xyxy[:,1].unsqueeze(0).expand(Nloc, G)
            x2_gt = gt_xyxy[:,2].unsqueeze(0).expand(Nloc, G)
            y2_gt = gt_xyxy[:,3].unsqueeze(0).expand(Nloc, G)

            l = xs_exp - x1_gt
            t = ys_exp - y1_gt
            r = x2_gt - xs_exp
            b_ = y2_gt - ys_exp

            inside_box = (l > 0) & (t > 0) & (r > 0) & (b_ > 0)

            # ---- center sampling (FCOS trick) ----
            if center_sampling_radius > 0:
                box_centers_x = (x1_gt + x2_gt) / 2
                box_centers_y = (y1_gt + y2_gt) / 2
                radius = stride * center_sampling_radius

                x_min = box_centers_x - radius
                y_min = box_centers_y - radius
                x_max = box_centers_x + radius
                y_max = box_centers_y + radius

                cb_l = xs_exp - x_min
                cb_t = ys_exp - y_min
                cb_r = x_max - xs_exp
                cb_b = y_max - ys_exp
                inside_center = (cb_l > 0) & (cb_t > 0) & (cb_r > 0) & (cb_b > 0)
                inside_box = inside_box & inside_center

            # valid candidates
            max_side = torch.stack([l,t,r,b_], dim=2).max(dim=2).values
            size_mask = (max_side >= stride) & (max_side <= stride*8)  # FCOS heuristic ranges
            candidate_mask = inside_box & size_mask

            if candidate_mask.sum() == 0:
                continue

            # choose GT per location by smallest area
            gt_areas = ( (x2_gt - x1_gt) * (y2_gt - y1_gt) )  # (Nloc,G)
            gt_areas_masked = gt_areas.clone()
            gt_areas_masked[~candidate_mask] = 1e12
            min_areas, min_inds = gt_areas_masked.min(dim=1)

            pos_inds = (min_areas < 1e11).nonzero(as_tuple=False).squeeze(1)
            if pos_inds.numel() == 0:
                continue

            gt_inds = min_inds[pos_inds]
            l_sel = l[pos_inds, gt_inds]
            t_sel = t[pos_inds, gt_inds]
            r_sel = r[pos_inds, gt_inds]
            b_sel = b_[pos_inds, gt_inds]

            # centerness
            lr_min = torch.min(l_sel, r_sel)
            lr_max = torch.max(l_sel, r_sel)
            tb_min = torch.min(t_sel, b_sel)
            tb_max = torch.max(t_sel, b_sel)
            centerness = torch.sqrt(
                (lr_min / (lr_max + 1e-6)) *
                (tb_min / (tb_max + 1e-6))
            )

            # map pos_inds to grid coords
            iy = (pos_inds // W).long()
            ix = (pos_inds % W).long()

            reg_t[b, :, iy, ix] = torch.stack([l_sel, t_sel, r_sel, b_sel], dim=1).permute(1,0)
            ctr_t[b, 0, iy, ix] = centerness
            pos_m[b, iy, ix] = True

            labels_sel = labels[gt_inds]
            cls_t[b, labels_sel, iy, ix] = 1.0

        cls_targets.append(cls_t)
        reg_targets.append(reg_t)
        ctr_targets.append(ctr_t)
        pos_masks.append(pos_m)

    return cls_targets, reg_targets, ctr_targets, pos_masks

# ---------- losses ----------
def sigmoid_focal_loss_logits(pred_logits, target_onehot, alpha=0.25, gamma=2.0):
    """
    pred_logits: (N, C) logits
    target_onehot: (N, C) targets 0/1 (float)
    returns scalar loss (mean over all elements)
    """
    pred_sigmoid = torch.sigmoid(pred_logits)
    p_t = pred_sigmoid * target_onehot + (1 - pred_sigmoid) * (1 - target_onehot)
    alpha_factor = target_onehot * alpha + (1 - target_onehot) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_onehot, reduction='none')
    loss = alpha_factor * modulating_factor * bce
    return loss.sum() / max(1.0, target_onehot.sum())  # normalize by #positive (like FCOS)

def reg_iou_loss(pred_ltrb, gt_ltrb, centers):
    """
    pred_ltrb, gt_ltrb: (Npos,4) l,t,r,b (positive locations only)
    centers: (Npos,2) x,y center coordinates for each location in image space
    Returns mean (1 - IoU)
    """
    # convert to xyxy
    px, py = centers[:,0], centers[:,1]
    p_x1 = px - pred_ltrb[:,0]
    p_y1 = py - pred_ltrb[:,1]
    p_x2 = px + pred_ltrb[:,2]
    p_y2 = py + pred_ltrb[:,3]
    pred_boxes = torch.stack([p_x1,p_y1,p_x2,p_y2], dim=1)

    g_x1 = px - gt_ltrb[:,0]
    g_y1 = py - gt_ltrb[:,1]
    g_x2 = px + gt_ltrb[:,2]
    g_y2 = py + gt_ltrb[:,3]
    gt_boxes = torch.stack([g_x1,g_y1,g_x2,g_y2], dim=1)

    iou = compute_iou(pred_boxes, gt_boxes)
    loss = (1.0 - iou).mean()
    return loss

def centerness_bce_loss(pred_ctr_logits, target_ctr):
    # BCE with logits, only consider positive locations in target (we can mask)
    target = target_ctr.view(-1)
    pred = pred_ctr_logits.view(-1)
    # use BCEWithLogits; normalize by number of positives
    positive_mask = (target > 0)
    if positive_mask.sum() == 0:
        # no positives -> return zero
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    loss = F.binary_cross_entropy_with_logits(pred[positive_mask], target[positive_mask], reduction='mean')
    return loss

# ---------- wrapper to compute losses from model outputs ----------
def compute_loss(
    model_outputs,        # dict with 'cls','reg','ctr', each list of [B,C,H,W]
    batch_boxes,          # list of length B, each (Ni,4) in relative coords [0,1]
    batch_labels,         # list of length B, each (Ni,) class indices
    image_size,           # (H_img, W_img)
    strides = [16, 32, 64],  # must match your FPN
    focal_alpha=0.25, focal_gamma=2.0,
    weight_reg=1.0, weight_ctr=1.0
):
    """
    Compute FCOS losses given outputs and ground truth.
    """
    device = model_outputs['cls'][0].device
    B = len(batch_boxes)

    # Create GT targets
    cls_targets, reg_targets, ctr_targets, pos_masks = create_ground_truth_targets(
        batch_boxes, batch_labels, image_size,
        feature_maps=model_outputs['cls'],  # only need shape
        strides=strides,
        center_sampling_radius=1.5,
        num_classes=model_outputs['cls'][0].shape[1]
    )

    # Initialize accumulators
    total_cls_loss, total_reg_loss, total_ctr_loss = 0., 0., 0.
    total_pos = 0

    # Loop over batch
    for b in range(B):
        cls_preds_flat, cls_tgts_flat = [], []
        reg_preds_pos, reg_tgts_pos, centers_pos = [], [], []
        ctr_preds_logits, ctr_tgts = [], []

        for lvl, stride in enumerate(strides):
            cls_pred = model_outputs['cls'][lvl][b]   # (C,H,W)
            reg_pred = model_outputs['reg'][lvl][b]   # (4,H,W)
            ctr_pred = model_outputs['ctr'][lvl][b]   # (1,H,W)

            cls_tgt = cls_targets[lvl][b]             # (C,H,W)
            reg_tgt = reg_targets[lvl][b]             # (4,H,W)
            ctr_tgt = ctr_targets[lvl][b]             # (1,H,W)
            pos_mask = pos_masks[lvl][b]              # (H,W)

            C, H, W = cls_pred.shape
            Nloc = H * W

            # --- classification ---
            cls_preds_flat.append(cls_pred.permute(1,2,0).reshape(-1,C))   # (Nloc,C)
            cls_tgts_flat.append(cls_tgt.permute(1,2,0).reshape(-1,C))     # (Nloc,C)

            # --- positives for regression/centerness ---
            if pos_mask.any():
                pos_inds = pos_mask.flatten().nonzero(as_tuple=False).squeeze(1)

                # regression preds and tgts
                reg_pred_flat = reg_pred.permute(1,2,0).reshape(-1,4)      # (Nloc,4)
                reg_pred_pos = F.relu(reg_pred_flat[pos_inds])             # enforce positive
                reg_tgt_pos = reg_tgt.permute(1,2,0).reshape(-1,4)[pos_inds]

                # location centers
                shifts_x = (torch.arange(0,W,device=device)+0.5)*stride
                shifts_y = (torch.arange(0,H,device=device)+0.5)*stride
                ys, xs = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
                centers = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
                centers_pos.append(centers[pos_inds])

                # centerness
                ctr_pred_flat = ctr_pred.view(-1)
                ctr_tgt_flat = ctr_tgt.view(-1)
                ctr_preds_logits.append(ctr_pred_flat[pos_inds])
                ctr_tgts.append(ctr_tgt_flat[pos_inds])

                # collect reg
                reg_preds_pos.append(reg_pred_pos)
                reg_tgts_pos.append(reg_tgt_pos)

        # --- classification focal loss ---
        cls_preds_all = torch.cat(cls_preds_flat, dim=0)
        cls_tgts_all = torch.cat(cls_tgts_flat, dim=0)
        cls_loss = sigmoid_focal_loss_logits(cls_preds_all, cls_tgts_all,
                                             alpha=focal_alpha, gamma=focal_gamma)

        # --- regression IoU loss ---
        if len(reg_preds_pos) > 0:
            reg_preds_all = torch.cat(reg_preds_pos, dim=0)
            reg_tgts_all = torch.cat(reg_tgts_pos, dim=0)
            centers_all = torch.cat(centers_pos, dim=0)
            reg_loss = reg_iou_loss(reg_preds_all, reg_tgts_all, centers_all)

            ctr_preds_all = torch.cat(ctr_preds_logits, dim=0)
            ctr_tgts_all = torch.cat(ctr_tgts, dim=0)
            ctr_loss = centerness_bce_loss(ctr_preds_all, ctr_tgts_all)
            n_pos = reg_preds_all.shape[0]
        else:
            reg_loss, ctr_loss, n_pos = 0., 0., 0

        total_cls_loss += cls_loss
        total_reg_loss += reg_loss
        total_ctr_loss += ctr_loss
        total_pos += n_pos

    # Average over batch
    total_cls_loss /= B
    total_reg_loss /= B
    total_ctr_loss /= B

    total_loss = total_cls_loss + weight_reg*total_reg_loss + weight_ctr*total_ctr_loss

    return total_loss, total_cls_loss, total_reg_loss, total_ctr_loss