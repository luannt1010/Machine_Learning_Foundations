import math
import torch

def ciou(boxes1, boxes2, eps=1e-7):
    a_x1, a_y1, a_x2, a_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b_x1, b_y1, b_x2, b_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = torch.maximum(a_x1, b_x1)
    inter_y1 = torch.maximum(a_y1, b_y1)
    inter_x2 = torch.minimum(a_x2, b_x2)
    inter_y2 = torch.minimum(a_y2, b_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (a_x2 - a_x1).clamp(min=0) * (a_y2 - a_y1).clamp(min=0)
    area2 = (b_x2 - b_x1).clamp(min=0) * (b_y2 - b_y1).clamp(min=0)
    union = area1 + area2 - inter_area
    iou = inter_area / (union + eps)

    cxa = (a_x1 + a_x2) / 2
    cya = (a_y1 + a_y2) / 2
    cxb = (b_x1 + b_x2) / 2
    cyb = (b_y1 + b_y2) / 2
    center_dist = (cxa - cxb).pow(2) + (cya - cyb).pow(2)

    cw = torch.maximum(a_x2, b_x2) - torch.minimum(a_x1, b_x1)
    ch = torch.maximum(a_y2, b_y2) - torch.minimum(a_y1, b_y1)
    c_diag = cw.pow(2) + ch.pow(2) + eps

    wa = (a_x2 - a_x1).clamp(min=eps)
    ha = (a_y2 - a_y1).clamp(min=eps)
    wb = (b_x2 - b_x1).clamp(min=eps)
    hb = (b_y2 - b_y1).clamp(min=eps)

    v = 4 / math.pi**2 * (torch.atan(wb / hb) - torch.atan(wa / ha)).pow(2)
    alpha = v / (1 - iou + v + eps)
    return iou - (center_dist / c_diag + alpha * v)


def bbox_iou_matrix(boxes1, boxes2, eps=1e-7):
    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]

    inter_x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter_area
    return inter_area / (union + eps)


def bbox_ciou_matrix(boxes1, boxes2, eps=1e-7):
    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]

    inter_x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    w1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=eps)
    h1 = (boxes1[..., 3] - boxes1[..., 1]).clamp(min=eps)
    w2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=eps)
    h2 = (boxes2[..., 3] - boxes2[..., 1]).clamp(min=eps)
    union = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / (union + eps)

    center_x1 = (boxes1[..., 0] + boxes1[..., 2]) / 2
    center_y1 = (boxes1[..., 1] + boxes1[..., 3]) / 2
    center_x2 = (boxes2[..., 0] + boxes2[..., 2]) / 2
    center_y2 = (boxes2[..., 1] + boxes2[..., 3]) / 2
    center_distance = (center_x1 - center_x2).pow(2) + (center_y1 - center_y2).pow(2)

    enclosing_w = torch.maximum(boxes1[..., 2], boxes2[..., 2]) - torch.minimum(boxes1[..., 0], boxes2[..., 0])
    enclosing_h = torch.maximum(boxes1[..., 3], boxes2[..., 3]) - torch.minimum(boxes1[..., 1], boxes2[..., 1])
    enclosing_diag = enclosing_w.pow(2) + enclosing_h.pow(2) + eps

    v = 4 / math.pi**2 * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    alpha = v / (1 - iou + v + eps)
    return iou - (center_distance / enclosing_diag + alpha * v)


def flatten_cls_fm(fms):
    res = []
    for fm in fms:
        B, C, H, W = fm.shape
        fm = fm.permute(0, 2, 3, 1).contiguous()
        fm = fm.reshape(B, H*W, C)
        res.append(fm)
    return torch.cat(res, 1)

def decode_bboxes(reg_fms, imgsize, offset=0.5):
    decoded_boxes = []
    strides = [imgsize // fm.shape[-1] for fm in reg_fms]

    for fm, stride in zip(reg_fms, strides):
        B, C, H, W = fm.shape
        reg_max = C // 4
        reg_logits = fm.permute(0, 2, 3, 1).contiguous()
        reg_logits = reg_logits.view(B, H * W, 4, reg_max)

        probs = torch.softmax(reg_logits, dim=-1)
        proj = torch.arange(reg_max, device=fm.device, dtype=fm.dtype)
        dist = (probs * proj).sum(dim=-1)

        sx = torch.arange(W, device=fm.device, dtype=fm.dtype)
        sy = torch.arange(H, device=fm.device, dtype=fm.dtype)
        y, x = torch.meshgrid(sy, sx, indexing="ij")
        x = x.reshape(-1).unsqueeze(0)
        y = y.reshape(-1).unsqueeze(0)

        l, t, r, b = dist.unbind(dim=-1)
        x1 = (x + offset - l) * stride
        y1 = (y + offset - t) * stride
        x2 = (x + offset + r) * stride
        y2 = (y + offset + b) * stride
        decoded_boxes.append(torch.stack([x1, y1, x2, y2], dim=-1))

    return torch.cat(decoded_boxes, dim=1)


def decode_predictions(preds, imgsize):
    cls_p3, reg_p3 = preds["p3"]
    cls_p4, reg_p4 = preds["p4"]
    cls_p5, reg_p5 = preds["p5"]

    cls_logits = flatten_cls_fm([cls_p3, cls_p4, cls_p5])
    scores = cls_logits.sigmoid()
    boxes = decode_bboxes([reg_p3, reg_p4, reg_p5], imgsize).clamp(0, imgsize)
    return scores, cls_logits, boxes

def _empty_detection(device):
    return (
        torch.empty((0, 4), device=device),
        torch.empty((0,), device=device),
        torch.empty((0,), dtype=torch.long, device=device)
    )


def nms(boxes, scores, labels, confidence_threshold=None, iou_threshold=0.7, max_det=300):
    if confidence_threshold is not None:
        keep_conf = scores >= confidence_threshold
        boxes = boxes[keep_conf]
        scores = scores[keep_conf]
        labels = labels[keep_conf]

    if boxes.numel() == 0:
        return _empty_detection(boxes.device)

    keep = []

    for cls_id in labels.unique():
        cls_idx = (labels == cls_id).nonzero(as_tuple=False).squeeze(1)
        cls_boxes = boxes[cls_idx]
        cls_scores = scores[cls_idx]
        order = cls_scores.argsort(descending=True)

        while order.numel() > 0:
            best_idx = order[0]
            keep.append(cls_idx[best_idx])

            if order.numel() == 1:
                break

            remain_idx = order[1:]
            ious = bbox_iou_matrix(cls_boxes[best_idx].unsqueeze(0), cls_boxes[remain_idx]).squeeze(0)
            order = remain_idx[ious <= iou_threshold]

    keep = torch.stack(keep)
    keep = keep[scores[keep].argsort(descending=True)[:max_det]]
    return boxes[keep], scores[keep], labels[keep]


def batch_nms(boxes, scores, confidence_threshold=0.001, iou_threshold=0.7, max_det=300):
    detections = []
    B, _, _ = scores.shape

    for b in range(B):
        img_boxes = boxes[b]
        img_scores = scores[b]
        candidate_box_idx, candidate_labels = torch.where(img_scores >= confidence_threshold)

        if candidate_box_idx.numel() == 0:
            detections.append(_empty_detection(img_boxes.device))
            continue

        candidate_boxes = img_boxes[candidate_box_idx]
        candidate_scores = img_scores[candidate_box_idx, candidate_labels]
        detections.append(nms(
            candidate_boxes,
            candidate_scores,
            candidate_labels,
            confidence_threshold=None,
            iou_threshold=iou_threshold,
            max_det=max_det
        ))

    return detections


def init_metric_stats():
    return {
        "correct": [],
        "scores": [],
        "pred_labels": [],
        "target_labels": []
    }


def _first_indices_for_sorted_unique(values):
    return torch.stack([
        (values == value).nonzero(as_tuple=False)[0, 0]
        for value in values.unique(sorted=True)
    ])


def match_predictions(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresholds):
    num_pred = pred_boxes.shape[0]
    correct = torch.zeros((num_pred, iou_thresholds.shape[0]), dtype=torch.bool, device=pred_boxes.device)

    if num_pred == 0 or gt_boxes.shape[0] == 0:
        return correct

    del pred_scores
    iou = bbox_iou_matrix(gt_boxes, pred_boxes)
    correct_class = gt_labels[:, None] == pred_labels[None, :]
    iou = iou * correct_class

    for t, iou_threshold in enumerate(iou_thresholds):
        matches = (iou >= iou_threshold).nonzero(as_tuple=False)
        if matches.numel() == 0:
            continue

        match_iou = iou[matches[:, 0], matches[:, 1]]
        matches = matches[match_iou.argsort(descending=True)]
        if matches.shape[0] > 1:
            matches = matches[_first_indices_for_sorted_unique(matches[:, 1])]
            matches = matches[_first_indices_for_sorted_unique(matches[:, 0])]

        correct[matches[:, 1], t] = True

    return correct


def update_metric_stats(metric_stats, preds, gt_labels, gt_boxes, mask_gt, imgsize, iou_thresholds,
                        confidence_threshold=0.001, nms_iou_threshold=0.7, max_det=300):
    scores, _, pred_boxes = decode_predictions(preds, imgsize)
    detections = batch_nms(
        pred_boxes,
        scores,
        confidence_threshold=confidence_threshold,
        iou_threshold=nms_iou_threshold,
        max_det=max_det
    )

    for b, (pred_boxes_i, pred_scores_i, pred_labels_i) in enumerate(detections):
        valid_gt = mask_gt[b].bool()
        target_boxes = gt_boxes[b][valid_gt]
        target_labels = gt_labels[b][valid_gt].long()
        correct = match_predictions(
            pred_boxes_i,
            pred_scores_i,
            pred_labels_i,
            target_boxes,
            target_labels,
            iou_thresholds
        )

        metric_stats["correct"].append(correct.detach().cpu())
        metric_stats["scores"].append(pred_scores_i.detach().cpu())
        metric_stats["pred_labels"].append(pred_labels_i.detach().cpu())
        metric_stats["target_labels"].append(target_labels.detach().cpu())


def compute_ap(recall, precision, eps=1e-16):
    device = recall.device
    dtype = recall.dtype
    last_recall = recall[-1:] if recall.numel() else torch.ones(1, device=device, dtype=dtype)
    mrec = torch.cat((
        torch.zeros(1, device=device, dtype=dtype),
        recall,
        last_recall,
        torch.ones(1, device=device, dtype=dtype)
    ))
    mpre = torch.cat((
        torch.ones(1, device=device, dtype=dtype),
        precision,
        torch.zeros(2, device=device, dtype=dtype)
    ))
    mpre = torch.flip(torch.cummax(torch.flip(mpre, dims=[0]), dim=0).values, dims=[0])

    x = torch.linspace(0, 1, 101, device=device, dtype=dtype)
    idx = torch.searchsorted(mrec, x, right=True).clamp(1, mrec.numel() - 1)
    x0, x1 = mrec[idx - 1], mrec[idx]
    y0, y1 = mpre[idx - 1], mpre[idx]
    interp = y0 + (x - x0) * (y1 - y0) / (x1 - x0).clamp(min=eps)
    return torch.trapz(interp, x)


def _interp_confidence(x, confidence, values, left):
    confidence = confidence.to(device=x.device, dtype=x.dtype)
    values = values.to(device=x.device, dtype=x.dtype)
    xp = -confidence
    query = -x
    idx = torch.searchsorted(xp, query, right=True)
    result = torch.empty_like(query)

    left_mask = idx == 0
    right_mask = idx == xp.numel()
    middle_mask = ~(left_mask | right_mask)
    result[left_mask] = left
    result[right_mask] = values[-1]

    middle_idx = idx[middle_mask]
    x0 = xp[middle_idx - 1]
    x1 = xp[middle_idx]
    y0 = values[middle_idx - 1]
    y1 = values[middle_idx]
    weight = (query[middle_mask] - x0) / (x1 - x0).clamp(min=torch.finfo(x.dtype).eps)
    result[middle_mask] = y0 + weight * (y1 - y0)
    return result


def _smooth_curve(curve, fraction=0.05):
    num_filter = round(curve.numel() * fraction * 2) // 2 + 1
    padding = num_filter // 2
    padded = torch.cat((curve[0].repeat(padding), curve, curve[-1].repeat(padding)))
    return padded.unfold(0, num_filter, 1).mean(dim=-1)


def compute_detection_metrics(metric_stats, num_classes, iou_thresholds, eps=1e-16):
    empty_metrics = {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}

    if len(metric_stats["target_labels"]) == 0:
        return empty_metrics

    target_labels = torch.cat(metric_stats["target_labels"], dim=0)
    if target_labels.numel() == 0:
        return empty_metrics

    if len(metric_stats["scores"]) == 0 or sum(x.numel() for x in metric_stats["scores"]) == 0:
        return empty_metrics

    correct = torch.cat(metric_stats["correct"], dim=0)
    scores = torch.cat(metric_stats["scores"], dim=0)
    pred_labels = torch.cat(metric_stats["pred_labels"], dim=0)

    order = scores.argsort(descending=True)
    correct = correct[order]
    scores = scores[order]
    pred_labels = pred_labels[order]

    num_iou_thresholds = iou_thresholds.shape[0]
    ap = torch.zeros((num_classes, num_iou_thresholds), dtype=torch.float32)
    confidence_axis = torch.linspace(0, 1, 1000)
    precision_curve = torch.zeros((num_classes, confidence_axis.numel()), dtype=torch.float32)
    recall_curve = torch.zeros_like(precision_curve)
    valid_classes = []

    for cls_id in range(num_classes):
        n_gt = (target_labels == cls_id).sum()
        n_pred = (pred_labels == cls_id).sum()

        if n_gt == 0:
            continue

        valid_classes.append(cls_id)
        if n_pred == 0:
            continue

        cls_mask = pred_labels == cls_id
        correct_cls = correct[cls_mask]
        scores_cls = scores[cls_mask]
        tp = correct_cls.float().cumsum(dim=0)
        fp = (~correct_cls).float().cumsum(dim=0)
        recall = tp / (n_gt.float() + eps)
        precision = tp / (tp + fp + eps)

        for t in range(num_iou_thresholds):
            ap[cls_id, t] = compute_ap(recall[:, t], precision[:, t])

        recall_curve[cls_id] = _interp_confidence(confidence_axis, scores_cls, recall[:, 0], left=0.0)
        precision_curve[cls_id] = _interp_confidence(confidence_axis, scores_cls, precision[:, 0], left=1.0)

    if len(valid_classes) == 0:
        return empty_metrics

    valid_classes = torch.tensor(valid_classes, dtype=torch.long)
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + eps)
    best_idx = _smooth_curve(f1_curve[valid_classes].mean(dim=0), fraction=0.1).argmax()
    return {
        "precision": precision_curve[valid_classes, best_idx].mean().item(),
        "recall": recall_curve[valid_classes, best_idx].mean().item(),
        "map50": ap[valid_classes, 0].mean().item(),
        "map50_95": ap[valid_classes].mean().item()
    }
