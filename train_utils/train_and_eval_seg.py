import torch
import train_utils.distributed_utils as utils
import torch.nn.functional as F

def calculate_clsloss(inputs, target, dice_weight=1.0, epsilon=1e-8):
    """
    Combined BCE (Binary Cross Entropy) and Dice Loss.

    Args:
    - inputs (torch.Tensor): Model predictions (logits).
    - target (torch.Tensor): Ground truth labels.
    - dice_weight (float): Weight for the Dice Loss.
    - epsilon (float): Small constant to avoid division by zero.

    Returns:
    - torch.Tensor: Combined BCE + Dice Loss.
    """
    binarized_target = (target > 0).float()   # binarized_labels

    # Binary Cross Entropy Loss
    bce_loss = F.binary_cross_entropy_with_logits(inputs, binarized_target)

    # Dice Loss
    intersection = (inputs * binarized_target).sum(dim=(1, 2, 3))
    union = inputs.sum(dim=(1, 2, 3)) + binarized_target.sum(dim=(1, 2, 3))

    dice_loss = 1.0 - (2.0 * intersection + epsilon) / (union + epsilon)

    # Combine BCE and Dice Loss
    combined_loss = bce_loss + dice_weight * dice_loss.mean()
    return combined_loss

def calculate_recall(predicted_probs, target, threshold=0.5):
    """
    Calculate Recall for binary segmentation task.

    Args:
    - predicted_probs (torch.Tensor): Model predictions (single-channel probability map after sigmoid).
    - target (torch.Tensor): Ground truth labels.
    - threshold (float): Threshold for binarizing predictions.

    Returns:
    - float: Recall score.
    """
    # Binarize predictions based on threshold
    binarized_predicted = (predicted_probs > threshold).float()

    # Binarize target labels
    binarized_target = (target > 0).float()

    # True Positives (TP) and False Negatives (FN)
    true_positive = (binarized_predicted * binarized_target).sum()
    false_negative = binarized_target.sum() - true_positive

    # Calculate Recall
    recall = true_positive / (true_positive + false_negative + 1e-8)  # Adding a small constant to avoid division by zero

    return recall.item()

def calculate_f1_score(predicted_probs, target, threshold=0.5):
    """
    Calculate F1 Score for binary segmentation task.

    Args:
    - predicted_probs (torch.Tensor): Model predictions (single-channel probability map after sigmoid).
    - target (torch.Tensor): Ground truth labels.
    - threshold (float): Threshold for binarizing predictions.

    Returns:
    - float: F1 Score.
    """
    # Binarize predictions based on threshold
    binarized_predicted = (predicted_probs > threshold).float()

    # Binarize target labels
    binarized_target = (target > 0).float()

    # True Positives (TP), False Positives (FP), and False Negatives (FN)
    true_positive = (binarized_predicted * binarized_target).sum()
    false_positive = binarized_predicted.sum() - true_positive
    false_negative = binarized_target.sum() - true_positive

    # Calculate Precision and Recall
    precision = true_positive / (true_positive + false_positive + 1e-8)  # Adding a small constant to avoid division by zero
    recall = true_positive / (true_positive + false_negative + 1e-8)  # Adding a small constant to avoid division by zero

    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)  # Adding a small constant to avoid division by zero

    return f1_score.item()

def calculate_iou(predicted_probs, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary segmentation task.

    Args:
    - predicted_probs (torch.Tensor): Model predictions (single-channel probability map after sigmoid).
    - target (torch.Tensor): Ground truth labels.
    - threshold (float): Threshold for binarizing predictions.

    Returns:
    - float: Intersection over Union (IoU) score.
    """
    # Binarize predictions based on threshold
    binarized_predicted = (predicted_probs > threshold).float()

    # Binarize target labels
    binarized_target = (target > 0).float()

    # True Positives (TP), False Positives (FP), False Negatives (FN)
    intersection = (binarized_predicted * binarized_target).sum()
    union = binarized_predicted.sum() + binarized_target.sum() - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-8)  # Adding a small constant to avoid division by zero

    return iou.item()

def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=5, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def evaluate_seg(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0
    total_recall = 0.0
    total_iou = 0.0
    num_samples = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():   #上下文管理器,不需要计算梯度,关闭autograd引擎
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            out = model(image)
            loss = calculate_clsloss(out, target)
            f1_score_value = calculate_f1_score(out, target)
            recall_value = calculate_recall(out, target)
            iou = calculate_iou(out, target)

            total_loss += loss * image.size(0)
            total_f1 += f1_score_value * image.size(0)
            total_recall += recall_value * image.size(0)
            total_iou += iou * image.size(0)

            num_samples += image.size(0)

    avg_loss = total_loss / num_samples
    avg_f1 = total_f1 / num_samples
    avg_recall = total_recall / num_samples
    avg_iou = total_iou / num_samples

    print('Validation Loss: {:.4f}'.format(avg_loss))    #print(f"Combined Loss (BCE + Dice): {loss:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"IoU: {avg_iou:.4f}")
    return avg_loss, avg_f1, avg_recall, avg_iou

def train_one_epoch_seg(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):   #将管理器内部的代码块中所有计算转换为使用低精度浮点数

            out = model(image)
            loss = calculate_clsloss(out, target)

        optimizer.zero_grad()   #将优化器中的所有梯度清零
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr
