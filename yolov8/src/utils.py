import json
import torch
import os
import time
from tqdm import tqdm
from functools import wraps
from .metrics import (bbox_ciou_matrix, compute_detection_metrics, init_metric_stats,
                      update_metric_stats)


class SimpleTaskAlignedAssigner:
    """YOLOv8 tạo target bằng TaskAlignedAssigner.

Đầu tiên nó lấy pred_scores, pred_bboxes và ground truth.
Với mỗi GT box, nó chỉ xét các anchor point nằm trong GT box.
Sau đó nó tính align_metric = class_score^alpha * IoU^beta.
Mỗi GT chọn top-k anchor có align_metric cao nhất làm positive.
Các anchor positive sẽ được gán:
- target_bboxes = GT box tương ứng
- target_scores = one-hot class nhưng giá trị là soft score
- fg_mask = True

Các anchor còn lại là background."""
    def __init__(self, num_classes=80, topk=10, alpha=0.5, beta=6.0, eps=1e-9):
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        B, N, C = pred_scores.shape
        device = pred_scores.device
        target_scores = torch.zeros_like(pred_scores)
        target_bboxes = torch.zeros_like(pred_bboxes)
        fg_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        target_gt_idx = torch.full((B, N), -1, dtype=torch.long, device=device)
        pred_probs = pred_scores.sigmoid()

        for b in range(B):
            valid = mask_gt[b].bool()
            labels = gt_labels[b][valid].long()
            boxes = gt_bboxes[b][valid]
            num_gt = boxes.shape[0]
            if num_gt == 0:
                continue
            lt = anchor_points[:, None, :] - boxes[None, :, :2]
            rb = boxes[None, :, 2:] - anchor_points[:, None, :]
            in_gt = torch.cat((lt, rb), dim=-1).amin(dim=-1) > self.eps
            # class score của class GT
            # pred_probs[b]: [N, C]
            # labels: [num_gt]
            # cls_score: [N, num_gt]
            cls_score = pred_probs[b][:, labels]

            # CIoU giữa mọi pred box và mọi GT box
            # [N, num_gt]
            overlaps = bbox_ciou_matrix(pred_bboxes[b], boxes).clamp(min=0)

            # task aligned metric
            align_metric = (cls_score ** self.alpha) * (overlaps ** self.beta)

            # chỉ giữ anchor nằm trong GT box
            align_metric = align_metric * in_gt.float()

            # mỗi GT chọn top-k anchor tốt nhất
            k = min(self.topk, N)

            topk_idx = torch.topk(align_metric, k=k,dim=0).indices # [topk, num_gt]
            pos_mask = torch.zeros((N, num_gt), dtype=torch.bool, device=device)
            gt_ids = torch.arange(num_gt, device=device)
            gt_ids = gt_ids.view(1, num_gt).expand(k, num_gt)
            pos_mask[topk_idx.reshape(-1), gt_ids.reshape(-1)] = True
            # loại anchor có metric = 0
            pos_mask = pos_mask & (align_metric > 0)
            # Nếu một anchor thuộc nhiều GT, Ultralytics chọn GT có overlap cao nhất.
            multi_gt = pos_mask.sum(dim=1) > 1
            if multi_gt.any():
                best_overlap_gt = overlaps.argmax(dim=1)
                pos_mask[multi_gt] = False
                pos_mask[multi_gt, best_overlap_gt[multi_gt]] = True

            fg = pos_mask.any(dim=1)
            if fg.any():
                pos_idx = fg.nonzero(as_tuple=False).squeeze(1)
                assigned_gt = pos_mask[pos_idx].long().argmax(dim=1)
                assigned_boxes = boxes[assigned_gt]
                assigned_labels = labels[assigned_gt]
                fg_mask[b, pos_idx] = True
                target_gt_idx[b, pos_idx] = assigned_gt
                target_bboxes[b, pos_idx] = assigned_boxes
                masked_align_metric = align_metric * pos_mask
                per_gt_max_metric = masked_align_metric.amax(dim=0).clamp(min=self.eps)
                per_gt_max_overlap = (overlaps * pos_mask).amax(dim=0)
                normalized_metric = masked_align_metric * (
                    per_gt_max_overlap / per_gt_max_metric
                ).unsqueeze(0)
                soft_scores = normalized_metric.amax(dim=1)[pos_idx].clamp(0, 1)
                target_scores[b, pos_idx, assigned_labels] = soft_scores

        return target_bboxes, target_scores, fg_mask, target_gt_idx



def get_boxes_and_dist_multiscale(fms, imgsize, offset=0.5):

    preds_dist_raw_all = []
    preds_dist_all = []
    boxes_all = []
    anchor_points_all = []
    stride_tensor_all = []
    strides = [imgsize//fm.shape[-1] for fm in fms]

    for fm, s in zip(fms, strides):
        B, C, H, W = fm.shape
        device, dtype = fm.device, fm.dtype
        reg_max = C // 4
        reg_preds = fm.permute(0, 2, 3, 1).contiguous()
        reg_preds = reg_preds.reshape(B, H*W, 4, reg_max)
        probs = torch.softmax(reg_preds, -1)
        proj = torch.arange(reg_max, device=reg_preds.device, dtype=reg_preds.dtype)
        dist = (proj * probs).sum(-1)
        sx = torch.arange(W, device=reg_preds.device, dtype=reg_preds.dtype)
        sy = torch.arange(H, device=reg_preds.device, dtype=reg_preds.dtype)
        y, x = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points = torch.stack((x+offset, y+offset), dim=-1).reshape(-1, 2)
        stride_tensor = torch.full((H*W, 1), s, device=device, dtype=dtype)
        anchor_points_all.append(anchor_points)
        stride_tensor_all.append(stride_tensor)
        x, y = (x.reshape(-1)).unsqueeze(0), (y.reshape(-1)).unsqueeze(0)
        l, t, r, b = dist.unbind(-1)
        x1 = x + offset - l
        y1 = y + offset - t
        x2 = x + offset + r
        y2 = y + offset + b
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        preds_dist_raw_all.append(reg_preds)
        preds_dist_all.append(dist)
        boxes_all.append(boxes)
    anchor_points_all = torch.cat(anchor_points_all, 0)
    stride_tensor_all = torch.cat(stride_tensor_all, 0)
    preds_dist_raw_all = torch.cat(preds_dist_raw_all, 1)
    preds_dist_all = torch.cat(preds_dist_all, 1)
    boxes_all = torch.cat(boxes_all, 1)
    return preds_dist_raw_all, preds_dist_all, boxes_all, anchor_points_all, stride_tensor_all



def bbox2dist(anchor_points, target_bboxes, reg_max):
    """
    anchor_points: [N, 2]
    target_bboxes: [B, N, 4]  # x1,y1,x2,y2 cùng scale với anchor_points

    return:
        target_dist: [B, N, 4]  # l,t,r,b
    """
    x1y1 = target_bboxes[..., :2]
    x2y2 = target_bboxes[..., 2:]
    lt = anchor_points.unsqueeze(0) - x1y1
    rb = x2y2 - anchor_points.unsqueeze(0)
    target_dist = torch.cat([lt, rb], dim=-1)
    # Vì DFL có bins 0 -> reg_max-1
    target_dist = target_dist.clamp(0, reg_max - 1 - 0.01)
    return target_dist

# def log_time(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         results = func(*args, **kwargs)
#         end = time.time()
#         print(f"Time: {(end-start)/60}m.")
#         return results
#     return wrapper

def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, sp, scheduler=None,
          metric_conf_threshold=0.001, metric_nms_iou_threshold=0.7, metric_max_det=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(sp, exist_ok=True)
    model = model.to(device)
    best_sp = os.path.join(sp, "best.pth")
    last_sp = os.path.join(sp, "last.pth")
    history_sp = os.path.join(sp, "history.json")
    best_map5095 = 0.0
    total_time = 0.0
    history = {"tr_loss": [], "val_loss": [], "precision": [], "recall": [], "map50": [], "map50_95": []}

    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        tr_running_loss = 0.0
        tr_box_loss = 0.0
        tr_cls_loss = 0.0
        tr_dfl_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAINING]", leave=False)
        for images, gt_labels, gt_boxes, mask_gt in train_pbar:
            images = images.to(device)
            gt_labels = gt_labels.to(device)
            gt_boxes = gt_boxes.to(device)
            mask_gt = mask_gt.to(device)

            preds = model(images)
            total_losses, losses = loss_fn(preds, gt_labels, gt_boxes, mask_gt)


            optimizer.zero_grad()
            total_losses.backward()
            optimizer.step()

            tr_running_loss += total_losses.item()
            tr_box_loss += losses["box_loss"].item()
            tr_cls_loss += losses["cls_loss"].item()
            tr_dfl_loss += losses["dfl_loss"].item()

        tr_epoch_loss = tr_running_loss / len(train_loader)
        tr_box_loss = tr_box_loss / len(train_loader)
        tr_cls_loss = tr_cls_loss / len(train_loader)
        tr_dfl_loss = tr_dfl_loss / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        val_box_loss = 0.0
        val_cls_loss = 0.0
        val_dfl_loss = 0.0
        metric_stats = init_metric_stats()
        metric_iou_thresholds = torch.arange(0.5, 0.96, 0.05, device=device)
        imgsize = loss_fn.imgsize
        num_classes = model.num_classes
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [VALIDATING]", leave=False)
        with torch.no_grad():
            for images, gt_labels, gt_boxes, mask_gt in val_pbar:
                images = images.to(device)
                gt_labels = gt_labels.to(device)
                gt_boxes = gt_boxes.to(device)
                mask_gt = mask_gt.to(device)

                preds = model(images)
                total_losses, losses = loss_fn(preds, gt_labels, gt_boxes, mask_gt)

                val_running_loss += total_losses.item()
                val_box_loss += losses["box_loss"].item()
                val_cls_loss += losses["cls_loss"].item()
                val_dfl_loss += losses["dfl_loss"].item()
                update_metric_stats(metric_stats, preds, gt_labels, gt_boxes, mask_gt,
                                    imgsize if imgsize is not None else images.shape[-1],
                                    metric_iou_thresholds,
                                    confidence_threshold=metric_conf_threshold,
                                    nms_iou_threshold=metric_nms_iou_threshold,
                                    max_det=metric_max_det)

            val_epoch_loss = val_running_loss / len(val_loader)
            val_box_loss = val_box_loss / len(val_loader)
            val_cls_loss = val_cls_loss / len(val_loader)
            val_dfl_loss = val_dfl_loss / len(val_loader)
            val_metrics = compute_detection_metrics(metric_stats, num_classes, metric_iou_thresholds.cpu())
            p, r, map50, map5095 = val_metrics["precision"], val_metrics["recall"], val_metrics["map50"], val_metrics["map50_95"]

        end = time.perf_counter()
        epoch_time = (end - start) / 60
        total_time += epoch_time

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['map50_95'])
            else:
                scheduler.step()

        history["tr_loss"].append(tr_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["precision"].append(p)
        history["recall"].append(r)
        history["map50"].append(map50)
        history["map50_95"].append(map5095)

        print(f"Epoch {epoch + 1}/{epochs} - Time={epoch_time:.2f}m")
        print(f"tr_loss: {tr_epoch_loss:.4f}    |   "
              f"box_loss: {tr_box_loss:.4f}     |   "
              f"cls_loss: {tr_cls_loss:.4f}     |   "
              f"dfl_loss: {tr_dfl_loss:.4f}")
        print(f"val_loss: {val_epoch_loss:.4f}  |   "
              f"box_loss: {val_box_loss:.4f}    |   "
              f"cls_loss: {val_cls_loss:.4f}    |   "
              f"dfl_loss: {val_dfl_loss:.4f}")
        print(f"val_metrics: "
              f"P: {p:.4f}  |   "
              f"R: {r:.4f}  |   "
              f"mAP50: {map50:.4f}  |   "
              f"mAP50-95: {map5095:.4f}")

        checkpoint = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "loss_fn": loss_fn.state_dict(),
                      "epoch": epoch}
        if map5095 > best_map5095:
            best_map5095 = map5095
            torch.save(checkpoint, best_sp)
            print(f"Best model is saved at {best_sp} epoch {epoch+1}")
        torch.save(checkpoint, last_sp)
    history["total_train_time"] = total_time
    with open(history_sp, "w") as f:
        json.dump(history, f)


