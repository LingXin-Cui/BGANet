import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision



def iou_score(output, target, threshold: float = 0.5, eps: float = 1e-5, return_hd95: bool = False,
              hd95_empty_penalty: float = 300.0):
    if torch.is_tensor(output):
        prob = torch.sigmoid(output.detach())
        target_t = target.detach()

        if prob.ndim == 2:
            prob = prob.unsqueeze(0)
        if target_t.ndim == 2:
            target_t = target_t.unsqueeze(0)

        pred = prob > float(threshold)
        target_b = target_t > 0.5

        batch_size = int(pred.shape[0]) if pred.ndim > 2 else 1
        pred_f = pred.reshape(batch_size, -1)
        target_f = target_b.reshape(batch_size, -1)

        intersection = torch.logical_and(pred_f, target_f).sum(dim=1).float()
        union = torch.logical_or(pred_f, target_f).sum(dim=1).float()
        pred_sum = pred_f.sum(dim=1).float()
        target_sum = target_f.sum(dim=1).float()

        iou = (intersection + eps) / (union + eps)
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

        if not return_hd95:
            return float(iou.mean().item()), float(dice.mean().item())

        pred_np = pred[:, 0].cpu().numpy().astype(np.uint8) if pred.ndim >= 4 else pred.cpu().numpy().astype(np.uint8)
        target_np = target_b[:, 0].cpu().numpy().astype(np.uint8) if target_b.ndim >= 4 else target_b.cpu().numpy().astype(np.uint8)
        hd95_vals = []
        for p, t in zip(pred_np, target_np):
            if p.sum() == 0 or t.sum() == 0:
                hd95_vals.append(float(hd95_empty_penalty))
            else:
                hd95_vals.append(float(hd95(p, t)))
        hd95_mean = float(np.mean(hd95_vals)) if hd95_vals else float(hd95_empty_penalty)
        return float(iou.mean().item()), float(dice.mean().item()), hd95_mean

    output_np = np.asarray(output)
    target_np = np.asarray(target)
    if output_np.ndim == 2:
        output_np = output_np[None, ...]
        target_np = target_np[None, ...]

    output_b = output_np > float(threshold)
    target_b = target_np > 0.5

    batch_size = int(output_b.shape[0]) if output_b.ndim > 0 else 1
    output_f = output_b.reshape(batch_size, -1)
    target_f = target_b.reshape(batch_size, -1)

    intersection = np.logical_and(output_f, target_f).sum(axis=1).astype(np.float64)
    union = np.logical_or(output_f, target_f).sum(axis=1).astype(np.float64)
    pred_sum = output_f.sum(axis=1).astype(np.float64)
    target_sum = target_f.sum(axis=1).astype(np.float64)

    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    if not return_hd95:
        return float(iou.mean()), float(dice.mean())

    hd95_vals = []
    for p, t in zip(output_b.astype(np.uint8), target_b.astype(np.uint8)):
        if p.sum() == 0 or t.sum() == 0:
            hd95_vals.append(float(hd95_empty_penalty))
        else:
            hd95_vals.append(float(hd95(p, t)))
    hd95_mean = float(np.mean(hd95_vals)) if hd95_vals else float(hd95_empty_penalty)
    return float(iou.mean()), float(dice.mean()), hd95_mean


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
