import numpy as np
import torch
from torch import nn
import train_utils.distributed_utils as utils
import torch.nn.functional as F


def CustomLoss(y_true, y_pred):
    # 计算MAE
    mae = F.l1_loss(y_true, y_pred)

    # 计算指数项的和
    exp_sum = torch.sum(torch.exp(torch.abs(y_true - y_pred)) - 1)

    # 总损失为MAE加上指数项的和的平均值
    loss = mae + exp_sum / y_true.size(0)

    return loss

def L1_smooth_loss(predicted, target, reduction='mean', beta=1.0):
    diff = torch.abs(predicted - target)
    if beta > 0:
        smooth = torch.sign(diff) * (1 - torch.exp(-beta * diff))
        loss = smooth * diff
    else:
        loss = diff
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    return loss

def reg_criterion(inputs, target):
    mae = F.l1_loss(inputs, target)
    mse = nn.MSELoss()(inputs, target)
    return mse + mae


def calculate_clsloss(inputs, target, probas_threshold=0.5, dice_weight=1.0, epsilon=1e-8):
    """
    Combined BCE (Binary Cross Entropy) and Dice Loss.

    Args:
    - inputs (torch.Tensor): Model predictions (probs from sigmoid function).
    - target (torch.Tensor): Ground truth labels.
    - dice_weight (float): Weight for the Dice Loss.
    - epsilon (float): Small constant to avoid division by zero.

    Returns:
    - torch.Tensor: Combined BCE + Dice Loss.
    """

    binarized_target = (target > 0).float()   # binarized_labels

    if inputs.shape[1] == 2:  # 如果是两个通道的logits，要将target拆分为两通道
        # 创建两个通道的张量
        channel_0 = (binarized_target == 0).float()  # 将原始张量中的0映射到第一个通道
        channel_1 = (binarized_target == 1).float()  # 将原始张量中的1映射到第二个通道
        binarized_target = torch.concat([channel_0, channel_1], dim=1)
        inputs_probabilities = torch.nn.functional.softmax(inputs, dim=1)
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy(inputs_probabilities, binarized_target)   #F.binary_cross_entropy_with_logits：适用于二分类问题, 输入参数是未经过softmax的logits
        binarized_inputs = (inputs_probabilities > probas_threshold).float()

    # Dice Loss
    intersection = (binarized_inputs * binarized_target).sum(dim=(1, 2, 3))
    union = binarized_inputs.sum(dim=(1, 2, 3)) + binarized_target.sum(dim=(1, 2, 3))

    dice_loss = 1.0 - (2.0 * intersection + epsilon) / (union + epsilon)

    # 综合交叉熵损失和 Dice Loss
    lambda_dice = 1.0  # 可以调整的权重
    # Combine BCE and Dice Loss
    combined_loss = bce_loss + dice_weight * dice_loss.mean()
    return combined_loss



def calculate_clsloss_ori(inputs, target):
    y_truebi = torch.where(target > 0, torch.ones_like(target), torch.zeros_like(target))
    y_truebi = y_truebi.long().view(-1)
    inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
    #二值交叉熵损失
    BCElosses = F.cross_entropy(inputs, y_truebi, reduction='mean')

    # # 计算Dice系数和Dice Loss
    # intersection = torch.sum(y_truebi.unsqueeze(1) * inputs)  # 重叠区域
    # union = torch.sum(y_truebi) + torch.sum(inputs)  # 总体区域
    # dice_coefficient = (2.0 * intersection) / (union + 1e-7)  # 加上一个很小的常数以避免除以0
    # dice_loss = 1 - dice_coefficient

     # 计算 Dice Loss
    eps = 1e-7
    probas = F.softmax(inputs, dim=1)
    preds = torch.argmax(probas, dim=1)

    intersection = torch.sum(preds * y_truebi)
    union = torch.sum(preds) + torch.sum(y_truebi)

    dice_loss = 1 - (2 * intersection + eps) / (union + eps)

    # 综合交叉熵损失和 Dice Loss
    lambda_dice = 1.0  # 可以调整的权重
    total_loss = BCElosses + lambda_dice * dice_loss

    return total_loss

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

def calculate_building_mse(out, y_true):  # calculate the mse of building area
    y_true_nonzero_mask = (y_true > 0).float()

    # Filter out predictions and ground truth for only non-zero pixels
    out_nonzero = out[y_true_nonzero_mask.bool()]
    y_true_nonzero = y_true[y_true_nonzero_mask.bool()]

    building_mse = nn.MSELoss()(out_nonzero, y_true_nonzero)

    return building_mse

def T36_criterion(Aout, Bout, Cout, out, y_true, log_vars):
  # seg_loss = calculate_clsloss(Bout, y_true, probas_threshold=0.5)
  seg_loss = calculate_clsloss_ori(Bout, y_true)

  precision0 = torch.exp(-log_vars[0])
  reg_lossA = reg_criterion(Aout, y_true)
  loss1 = reg_lossA * precision0 + log_vars[0]

  precision2 = torch.exp(-log_vars[2])
  reg_lossC = reg_criterion(Cout, y_true)
  loss4 = reg_lossC * precision2 + log_vars[2]

  precision3 = torch.exp(-log_vars[3])
  reg_loss = reg_criterion(out, y_true)
  loss5 = reg_loss * precision3 + log_vars[3]

  return loss1 + seg_loss + loss4 + loss5

def T36_criterion1(Aout, Bout, Cout, out, y_true, log_vars):
  seg_loss1 = calculate_clsloss_ori(Bout, y_true)
  seg_loss2 = calculate_clsloss_ori(Cout, y_true)

  precision0 = torch.exp(-log_vars[0])
  reg_lossA = reg_criterion(Aout, y_true)
  loss1 = reg_lossA * precision0 + log_vars[0]

  precision3 = torch.exp(-log_vars[3])
  reg_loss = reg_criterion(out, y_true)
  loss5 = reg_loss * precision3 + log_vars[3]

  return loss1 + seg_loss1 + seg_loss2 + loss5

def T36_criterion3(Aout, Bout, Cout, out, y_true, log_vars):
  # seg_loss = calculate_clsloss(Bout, y_true, probas_threshold=0.5)
  seg_loss = calculate_clsloss_ori(Bout, y_true)

  precision0 = torch.exp(-log_vars[0])
  reg_lossA = reg_criterion(Aout, y_true)
  loss1 = reg_lossA * precision0 + log_vars[0]

  precision2 = torch.exp(-log_vars[2])
  reg_lossC = reg_criterion(Cout, y_true)
  loss4 = reg_lossC * precision2 + log_vars[2]

  precision3 = torch.exp(-log_vars[3])
  reg_loss = reg_criterion(out, y_true)
  loss5 = reg_loss * precision3 + log_vars[3]

  return loss1 + seg_loss + loss4 + loss5


def evaluate(model, data_loader, device, num_classes, log_var_paras=[]):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    total_loss = 0.0
    val_rmse = 0.
    val_building_rmse = 0.
    sample_count = 0

    # total_Bout_f1 = 0.0
    # total_Bout_recall = 0.0
    # total_Bout_iou = 0.0
    #
    # total_f1 = 0.0
    # total_recall = 0.0
    # total_iou = 0.0

    with torch.no_grad():   #上下文管理器,不需要计算梯度,关闭autograd引擎
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            Aout, Bout, Cout, out = model(image)
            loss = T36_criterion(Aout, Bout, Cout, out, target, log_var_paras)

            # calculate the accuracy of reg
            mse = nn.MSELoss()(out, target).cpu().detach().numpy()
            val_rmse += (np.sqrt(mse) * image.shape[0])
            building_mse = calculate_building_mse(out, target).cpu().detach().numpy()
            val_building_rmse += (np.sqrt(building_mse) * image.shape[0])

            # # calculate the accuracy of seg
            # # B branch
            # Bf1_score_value = calculate_f1_score(Bout, target)
            # Brecall_value = calculate_recall(Bout, target)
            # Biou = calculate_iou(Bout, target)
            #
            # total_Bout_f1 += Bf1_score_value * image.shape[0]
            # total_Bout_recall += Brecall_value * image.shape[0]
            # total_Bout_iou += Biou * image.shape[0]

            # # entire
            # #二值化结果
            # min_value = torch.min(out) + 0.001
            # zero = torch.zeros_like(out)
            # out = torch.where(out >= min_value, out, zero)
            #
            # f1_score_value = calculate_f1_score(out, target, threshold=min_value)
            # recall_value = calculate_recall(out, target, threshold=min_value)
            # iou = calculate_iou(out, target, threshold=min_value)
            #
            # total_f1 += f1_score_value * image.shape[0]
            # total_recall += recall_value * image.shape[0]
            # total_iou += iou * image.shape[0]

            total_loss += loss.item() * image.shape[0]
            sample_count += image.shape[0]

    # avg_Bout_f1 = total_Bout_f1 / sample_count
    # avg_Bout_recall = total_Bout_recall / sample_count
    # avg_Bout_iou = total_Bout_iou / sample_count
    #
    # avg_f1 = total_f1 / sample_count
    # avg_recall = total_recall / sample_count
    # avg_iou = total_iou / sample_count

    avg_loss = total_loss / sample_count
    val_rmse = val_rmse/sample_count
    val_building_rmse = val_building_rmse/sample_count

    print('Validation Loss: {:.4f}'.format(avg_loss))

    # return avg_loss, sample_count, val_rmse, val_building_rmse, [avg_Bout_f1 ,avg_Bout_recall, avg_Bout_iou], [avg_f1, avg_recall, avg_iou]
    return avg_loss, sample_count, val_rmse, val_building_rmse


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None, log_var_paras=[]):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    train_rmse = 0.
    train_building_rmse = 0.
    sample_count = 0

    # total_Bout_f1 = 0.0
    # total_Bout_recall = 0.0
    # total_Bout_iou = 0.0
    #
    # total_f1 = 0.0
    # total_recall = 0.0
    # total_iou = 0.0

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):   #将管理器内部的代码块中所有计算转换为使用低精度浮点数

            Aout, Bout, Cout, out = model(image)
            loss = T36_criterion(Aout, Bout, Cout, out, target, log_var_paras)

            # calculate the accuracy of reg
            mse = nn.MSELoss()(out, target).cpu().detach().numpy()
            train_rmse += (np.sqrt(mse) * image.shape[0])
            building_mse = calculate_building_mse(out, target).cpu().detach().numpy()
            train_building_rmse += (np.sqrt(building_mse) * image.shape[0])

            # # calculate the accuracy of seg
            # # B branch
            # Bf1_score_value = calculate_f1_score(Bout, target)
            # Brecall_value = calculate_recall(Bout, target)
            # Biou = calculate_iou(Bout, target)

            # total_Bout_f1 += Bf1_score_value * image.shape[0]
            # total_Bout_recall += Brecall_value * image.shape[0]
            # total_Bout_iou += Biou * image.shape[0]

            # # entire
            # #二值化结果
            # min_value = torch.min(out) + 0.001
            # zero = torch.zeros_like(out)
            # out = torch.where(out >= min_value, out, zero)
            #
            # f1_score_value = calculate_f1_score(out, target, threshold=min_value)
            # recall_value = calculate_recall(out, target, threshold=min_value)
            # iou = calculate_iou(out, target, threshold=min_value)

            # total_f1 += f1_score_value * image.shape[0]
            # total_recall += recall_value * image.shape[0]
            # total_iou += iou * image.shape[0]

            sample_count += image.shape[0]

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

    # avg_Bout_f1 = total_Bout_f1 / sample_count
    # avg_Bout_recall = total_Bout_recall / sample_count
    # avg_Bout_iou = total_Bout_iou / sample_count

    # avg_f1 = total_f1 / sample_count
    # avg_recall = total_recall / sample_count
    # avg_iou = total_iou / sample_count

    train_rmse = train_rmse/sample_count
    train_building_rmse = train_building_rmse/sample_count

    # return metric_logger.meters["loss"].global_avg, lr, sample_count, train_rmse, train_building_rmse, [avg_Bout_f1 ,avg_Bout_recall, avg_Bout_iou], [avg_f1, avg_recall, avg_iou]
    return metric_logger.meters["loss"].global_avg, lr, sample_count, train_rmse, train_building_rmse


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

