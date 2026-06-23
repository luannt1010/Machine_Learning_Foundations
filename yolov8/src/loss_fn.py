import torch.nn as nn
import torch.nn.functional as F
from .metrics import ciou, flatten_cls_fm
from .utils import bbox2dist, SimpleTaskAlignedAssigner, get_boxes_and_dist_multiscale

def dfl_loss(pred_dist_raw, target_bboxes, target_scores, anchor_points, fg_mask, target_scores_sum):
    """
    pred_dist_raw: [B, N, 4, reg_max]  # logits
    target_bboxes: [B, N, 4]           # feature coordinate
    anchor_points: [N, 2]              # feature coordinate
    target_scores: [B, N, C]
    fg_mask:       [B, N]
    """
    reg_max = pred_dist_raw.shape[-1]
    if fg_mask.sum() == 0:
        return pred_dist_raw.sum() * 0.0
    # [B, N, 4]
    target_dist = bbox2dist(anchor_points, target_bboxes, reg_max)
    # Chỉ lấy positive anchors
    pred = pred_dist_raw[fg_mask]      # [num_pos, 4, reg_max]
    target = target_dist[fg_mask]      # [num_pos, 4]
    # [num_pos * 4, reg_max]
    pred = pred.reshape(-1, reg_max)
    # [num_pos * 4]
    target = target.reshape(-1)
    target_left = target.long()
    target_right = target_left + 1
    weight_left = target_right.float() - target
    weight_right = target - target_left.float()
    loss_left = F.cross_entropy(pred, target_left, reduction="none")
    loss_right = F.cross_entropy(pred, target_right, reduction="none")
    loss = loss_left * weight_left + loss_right * weight_right
    # [num_pos * 4] -> [num_pos, 4] -> [num_pos]
    loss = loss.reshape(-1, 4).mean(dim=-1)
    # weight theo target score của positive anchor
    weight = target_scores.sum(-1)[fg_mask]  # [num_pos]
    loss = (loss * weight).sum() / target_scores_sum
    return loss

def cls_loss(target_scores, pred_scores, target_scores_sum):
    """
        pred_scores:   [B, N, num_classes] logits
        target_scores: [B, N, num_classes] soft target
        """
    loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="sum")
    return loss / target_scores_sum

def bbox_loss(target_scores, target_bboxes, pred_bboxes, fg_mask, target_scores_sum):
    """
    target_scores: [B, N, num_classes]
    fg_mask:       [B, N]
    pred_bboxes:   [B, N, 4]
    target_bboxes: [B, N, 4]
    """
    weight = target_scores[fg_mask].sum(-1)
    iou = ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
    loss_iou = ((1.0-iou) * weight).sum() / target_scores_sum
    return loss_iou



class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5, topk=10, alpha=0.5, beta=6.0, imgsize=640):
        super().__init__()

        self.num_classes = num_classes
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.imgsize = imgsize
        self.assigner = SimpleTaskAlignedAssigner(num_classes=num_classes, topk=topk, alpha=alpha, beta=beta)

    def forward(self, preds, gt_labels, gt_bboxes, mask_gt):
        """
        preds:
            {
                "p3": [cls_p3, reg_p3],
                "p4": [cls_p4, reg_p4],
                "p5": [cls_p5, reg_p5],
            }

        cls_pi: [B, num_classes, H, W]
        reg_pi: [B, 4 * reg_max, H, W]

        gt_labels: [B, max_gt] hoặc [B, max_gt, 1]
        gt_bboxes: [B, max_gt, 4] pixel coordinate
        mask_gt:   [B, max_gt] hoặc [B, max_gt, 1]
        """
        cls_p3, reg_p3 = preds["p3"]
        cls_p4, reg_p4 = preds["p4"]
        cls_p5, reg_p5 = preds["p5"]
        pred_scores = flatten_cls_fm([cls_p3, cls_p4, cls_p5])  # [B, N, num_classes]
        pred_dist_raw, pred_dist, pred_bboxes, anchor_points, stride_tensor = get_boxes_and_dist_multiscale([reg_p3, reg_p4, reg_p5], self.imgsize)
        # pred_dist_raw: [B, N, 4, reg_max]
        # pred_dist:     [B, N, 4]
        # pred_bboxes:   [B, N, 4] feature coordinate
        # anchor_points: [N, 2] feature coordinate
        # stride_tensor: [N, 1]
        pred_bboxes_pixel = pred_bboxes.detach() * stride_tensor.unsqueeze(0)
        anchor_points_pixel = anchor_points * stride_tensor

        target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(pred_scores.detach(), pred_bboxes_pixel,
                                                                             anchor_points_pixel, gt_labels, gt_bboxes, mask_gt)
        # target_bboxes: [B, N, 4] pixel coordinate
        # target_scores: [B, N, num_classes]
        # fg_mask:       [B, N]
        target_scores_sum = target_scores.sum().clamp(min=1.0)
        loss_cls = cls_loss(target_scores, pred_scores, target_scores_sum)
        target_bboxes_feature = target_bboxes / stride_tensor.unsqueeze(0)
        loss_box = bbox_loss(target_scores, target_bboxes_feature, pred_bboxes, fg_mask, target_scores_sum)
        loss_dfl = dfl_loss(pred_dist_raw, target_bboxes_feature, target_scores, anchor_points, fg_mask, target_scores_sum)
        total_loss = self.box_gain * loss_box + self.cls_gain * loss_cls + self.dfl_gain * loss_dfl

        loss_items = {"loss": total_loss.detach(), "box_loss": loss_box.detach(),
                      "cls_loss": loss_cls.detach(), "dfl_loss": loss_dfl.detach()}
        return total_loss, loss_items
