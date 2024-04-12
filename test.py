import os
import time
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
from dataset import diydataset
import pandas as pd
from sklearn.metrics import r2_score
from tifffile import imwrite
from osgeo import gdal
import csv
from src.TriUnet3 import TriUnet36, TriUnet36t, TriUnet36tC, TriUnet36t2, TriUnet36t3, TriUnet36t1, TriUnet36t_sar, TriUnet36t_VIIRS, TriUnet36t_optical


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def save_plot2csv(tensor1, tensor2, filepath):
    # 将两个 tensor 变形为 1 维数组
    x = tensor1.reshape(-1)
    y = tensor2.reshape(-1)

    # 将这两个数组保存到 CSV 文件中
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pred', 'gt'])  # 写入表头
        for i in range(len(y)):
            writer.writerow([y[i], x[i]])  # 逐行写入数据

def calculate_delta(tensor1, tensor2):
    x = tensor1.reshape(-1)
    y = tensor2.reshape(-1)

    max_ratio1 = np.maximum(x / y, y / x)
    max_ratio = max_ratio1[~np.isinf(max_ratio1)]

    delta1 = np.mean(max_ratio < 1.25)
    delta2 = np.mean(max_ratio < 1.25 ** 2)
    delta3 = np.mean(max_ratio < 1.25 ** 3)
    return delta1, delta2, delta3

def calculate_recall(predicted_probs, target, threshold=2.9):
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

def calculate_f1_score(predicted_probs, target, threshold=2.9):
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

def calculate_iou(predicted_probs, target, threshold=2.9):
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

def main():
    num_classes = 1
    in_channel = 7
    weights_path = '45T36t_025pre_02_39_weights/model_0.pth'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = TriUnet36t(2, 4, 1, ori_filter=16)
    # model = TriUnet36t_optical(2, 4, 1, ori_filter=16)
    # model = TriUnet36t1(2, 4, 1, ori_filter=16)
    # model = TriUnet36t_VIIRS(2, 4, 1, ori_filter=16)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

   # Load dataset
    train_data = diydataset(root="C:\\Users\\lenovo\\Desktop\\test\\ImageSets", txt_name="train.txt")
    tes_data = diydataset(root="C:\\Users\\lenovo\\Desktop\\test\\ImageSets", txt_name="test.txt", transforms=train_data.transforms)

    # 提取均值和标准差
    normalized_trans = train_data.transforms
    mean = normalized_trans.transforms[-1].mean[0].numpy()
    std = normalized_trans.transforms[-1].std[0].numpy()

    # 定义一个空的 DataFrame
    results_df = pd.DataFrame(columns=['Sample', 'RMSE', 'MAE', 'building_RMSE', 'Bf1', 'Brecall', 'Biou', 'f1', 'recall', 'iou'])

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = tes_data.get_samples().shape[-2:]
        init_img = torch.zeros((1, in_channel, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()

        with open(os.path.join("C:\\Users\\lenovo\\Desktop\\test\\ImageSets", "test.txt"), "r") as f:
            pred_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        pred_tensorlist = []
        all_predB_tensor = []
        gt_tensorlist = []

        RMSE_list = []
        MAE_list = []
        building_RMSE_list = []

        Bf1_list = []
        Brecall_list = []
        Biou_list = []

        f1_list = []
        recall_list = []
        iou_list = []

        for test_num in range(len(tes_data)):
            temp, true = tes_data[test_num]
            img = temp.unsqueeze(0)

            _, Bout, _,  output = model(img.to(device))

            # 反均一化
            unnorm_true = true * std + mean     # [1, 256, 256]
            unnorm_prediction = (output.squeeze(0).to("cpu")) * std + mean

            # calculate and record metric
                # postprocess
            min_value = torch.min(unnorm_prediction) + 0.01
            zero = torch.zeros_like(unnorm_prediction)
            unnorm_prediction = torch.where(unnorm_prediction >= min_value, unnorm_prediction, zero)

                # calculate the accuracy of reg
            mse = (nn.MSELoss()(unnorm_prediction, unnorm_true)).numpy()
            building_mse = (calculate_building_mse(unnorm_prediction, unnorm_true)).numpy()
            mae = (F.l1_loss(unnorm_prediction, unnorm_true)).numpy()

                # B branch
            Bf1 = calculate_f1_score(Bout.squeeze(0).to("cpu"), true, threshold=min_value)
            Brecall = calculate_recall(Bout.squeeze(0).to("cpu"), true, threshold=min_value)
            Biou = calculate_iou(Bout.squeeze(0).to("cpu"), true, threshold=min_value)

                # entire
            f1 = calculate_f1_score(output.squeeze(0).to("cpu"), true, threshold=min_value)
            recall = calculate_recall(output.squeeze(0).to("cpu"), true, threshold=min_value)
            iou = calculate_iou(output.squeeze(0).to("cpu"), true, threshold=min_value)

                # 将结果保存到 CSV 文件
            results_df.loc[test_num] = [os.path.basename(pred_list[test_num]), np.sqrt(mse), mae, np.sqrt(building_mse), Bf1, Brecall, Biou, f1, recall, iou]
            results_df.to_csv(os.path.join('45T36t_025pre_02_39_weights_test_result', 'metric.csv'), index=False)

            RMSE_list.append(np.sqrt(mse))
            MAE_list.append(mae)
            building_RMSE_list.append(np.sqrt(building_mse))
            Bf1_list.append(Bf1)
            Brecall_list.append(Brecall)
            Biou_list.append(Biou)
            f1_list.append(f1)
            recall_list.append(recall)
            iou_list.append(iou)

            pred_tensorlist.append(unnorm_prediction)
            # all_predB_tensor.append(Bout.squeeze(0))
            gt_tensorlist.append(unnorm_true)

            # 保存test结果
            prediction = unnorm_prediction.numpy().astype(np.float32).reshape(256, 256)

                # get GetGeoTransform() and GetProjection() of sample
            geo_dataset = gdal.Open(pred_list[test_num])
            geotransform = geo_dataset.GetGeoTransform()
            projection = geo_dataset.GetProjection()

                # 创建TIFF文件
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(os.path.join('45T36t_025pre_02_39_weights_test_result', os.path.basename(pred_list[test_num])), prediction.shape[1], prediction.shape[0], 1, gdal.GDT_Float32)
            dataset.SetGeoTransform(geotransform)
            dataset.SetProjection(projection)

                # 将预测结果写入TIFF文件
            band = dataset.GetRasterBand(1)
            band.WriteArray(prediction)

                # 关闭数据集
            dataset.FlushCache()
            dataset = None

        print('样本数：', len(RMSE_list), '各样本平均值：')
        print("RMSE:", sum(RMSE_list) / len(RMSE_list), "MAE:", sum(MAE_list) / len(MAE_list), "building_rmse:", sum(building_RMSE_list) / len(building_RMSE_list))
        # print("Bf1:", sum(Bf1_list) / len(Bf1_list), "Brecall:", sum(Brecall_list) / len(Brecall_list), "Biou:", sum(Biou_list) / len(Biou_list))
        print("f1:", sum(f1_list) / len(f1_list), "recall:", sum(recall_list) / len(recall_list), "iou:", sum(iou_list) / len(iou_list))

        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        all_pred_tensor = (torch.cat(pred_tensorlist, dim=0)).numpy()
        all_predB_tensor = (torch.cat(pred_tensorlist, dim=0)).numpy()
        all_gt_tensor = (torch.cat(gt_tensorlist, dim=0)).numpy()

        # mse = nn.MSELoss()(all_pred_tensor, all_gt_tensor)
        # building_mse = calculate_building_mse(all_pred_tensor, all_gt_tensor)
        # mae = F.l1_loss(all_pred_tensor, all_gt_tensor)
        # print("RMSE:", np.sqrt(mse), "MAE:", mae, "building_rmse:", np.sqrt(building_mse))
        #
        # Bf1 = calculate_f1_score(all_predB_tensor, all_gt_tensor)
        # Brecall = calculate_recall(all_predB_tensor, all_gt_tensor)
        # Biou = calculate_iou(all_predB_tensor, all_gt_tensor)
        # print("Bf1:", Bf1, "Brecall:", Brecall, "Biou:", Biou)
        #
        # f1 = calculate_f1_score(all_pred_tensor, all_gt_tensor, threshold=2.5)
        # recall = calculate_recall(all_pred_tensor, all_gt_tensor, threshold=2.5)
        # iou = calculate_iou(all_pred_tensor, all_gt_tensor, threshold=2.5)
        # print("f1:", f1, "recall:", recall, "iou:", iou)


        save_plot2csv(all_gt_tensor, all_pred_tensor, filepath=os.path.join('45T36t_025pre_02_39_weights_test_result', 'scatterplot.csv'))




if __name__ == '__main__':
    if not os.path.exists("./45T36t_025pre_02_39_weights_test_result"):
        os.mkdir("./45T36t_025pre_02_39_weights_test_result")
    torch.cuda.empty_cache()

    main()
