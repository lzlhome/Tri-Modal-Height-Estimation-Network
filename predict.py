import os
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from dataset import diydataset
import pandas as pd
from osgeo import gdal
from src.TriUnet3 import TriUnet36, TriUnet36t, TriUnet36tC, TriUnet36t2, TriUnet36t3, TriUnet36t1, TriUnet36t_sar, TriUnet36t_VIIRS, TriUnet36t_optical


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def save_tensor_to_txt(tensor1, tensor2, file_path):
    # 将两个张量按照（x y）格式堆叠在一起
    stacked_tensor = np.column_stack((tensor1.reshape(-1), tensor2.reshape(-1)))

    # 将堆叠后的张量保存到txt文件中
    np.savetxt(file_path, stacked_tensor, fmt='%.18e', delimiter=' ')



def main():
    aux = False  # inference time not need aux_classifier
    num_classes = 1
    in_channel = 7
    weights_path = 'T36NEWseg_025pre_70_02weights/model_0.pth'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = TriUnet36t_optical(2, 4, 1, ori_filter=16)
    # model = TriUnet36t1(2, 4, 1, ori_filter=16)
    model = TriUnet36t(2, 4, 1, ori_filter=16)

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
    # 提取均值和标准差
    mean = train_data.transforms.transforms[-1].mean[0].numpy()
    std = train_data.transforms.transforms[-1].std[0].numpy()

    tes_data = diydataset(root="C:\\Users\\lenovo\\Desktop\\test\\ImageSets\\predM1234", txt_name="beijing.txt", transforms=train_data.transforms, is_pred=True)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = tes_data.get_samples().shape[-2:]
        init_img = torch.zeros((1, in_channel, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        pred = []


        with open(os.path.join("C:\\Users\\lenovo\\Desktop\\test\\ImageSets\\predM1234", "beijing.txt"), "r") as f:
            pred_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]


        print('num of pred_block : {}'.format(len(tes_data)))

        for test_num in range(len(tes_data)):
            temp, _, geo_information = tes_data[test_num]
            img = temp.unsqueeze(0)

            _, _, _, output = model(img.to(device))
            prediction = output.squeeze(0).squeeze(0)
            prediction = prediction.to("cpu")#.numpy().astype(np.float64)

            # 反均一化操作
            unnormalized_prediction = prediction * std + mean

            # postprocess
            min_value = torch.min(unnormalized_prediction) + 0.01
            zero = torch.zeros_like(unnormalized_prediction)
            unnormalized_prediction = torch.where(unnormalized_prediction >= min_value, unnormalized_prediction, zero)

            prediction = unnormalized_prediction.numpy().astype(np.float64)

            pred.append(prediction)

            # 假设prediction是一个256x256的单通道ndarray，类型为float32

            # 创建TIFF文件
            driver = gdal.GetDriverByName("GTiff")

            dataset = driver.Create(os.path.join('H:/M4', os.path.basename(pred_list[test_num])), prediction.shape[1], prediction.shape[0], 1, gdal.GDT_Float32)

            dataset.SetGeoTransform(geo_information[0])
            dataset.SetProjection(geo_information[1])
            # 将预测结果写入TIFF文件
            band = dataset.GetRasterBand(1)
            band.WriteArray(prediction)

            # 关闭数据集
            dataset.FlushCache()
            dataset = None

            print('finish: {}'.format(test_num))

        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))



if __name__ == '__main__':
    torch.cuda.empty_cache()

    main()
