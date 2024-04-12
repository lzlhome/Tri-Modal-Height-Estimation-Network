import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataset import diydataset
import time
import datetime
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from src.TriUnet3 import M1, M2, M3, M4

def create_model(pretrain=False):
    model = M4(2, 4, 1, ori_filter=16)

    if pretrain:
        weights_dict = torch.load('para.pth', map_location='cpu')['model']
        model_dict = model.state_dict()
        weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
        model_dict.update(weights_dict)
        model.load_state_dict(weights_dict)
        print('pre-trained paras load finish!')

    return model



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes
    in_channel = args.in_channel
    data_path = args.data_path

    # 用来保存训练以及验证过程中信息
    results_file = "results_weights{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    # Load dataset
    train_data = diydataset(txt_name="train.txt")
    val_data = diydataset(txt_name="val.txt", transforms=train_data.transforms)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    model = create_model(pretrain=False).to(device)

    normalized_trans = train_data.transforms
    std = normalized_trans.transforms[-1].std[0].numpy()

    # define task-dependent log_variance
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)
    log_var_c = torch.zeros((1,), requires_grad=True)
    log_var_d = torch.zeros((1,), requires_grad=True)

    params_to_optimize = ([p for p in model.parameters() if p.requires_grad] + [log_var_a] + [log_var_b] + [log_var_c] + [log_var_d])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam([
        {'params': params_to_optimize, 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'weight_decay': args.weight_decay}
    ])

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    print('begin train!')
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        print('============new epoch===========')

        mean_loss, lr, train_sample_count, train_rmse, train_building_rmse = train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler, log_var_paras=[log_var_a.to(device), log_var_b.to(device), log_var_c.to(device), log_var_d.to(device)])

        val_loss, val_sample_count, val_rmse, val_building_rmse = evaluate(model, val_loader, device=device, num_classes=num_classes, log_var_paras=[log_var_a.to(device), log_var_b.to(device), log_var_c.to(device), log_var_d.to(device)])

        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                             f"lr: {lr:.6f}\n" \
                            f"train_sample_count: {train_sample_count//16:.6f}\n" \
                             f"train_loss: {mean_loss:.4f}\n"\
                            f"train_rmse: {train_rmse:.6f}\n"\
                            # f"train_building_rmse: {train_building_rmse:.6f}\n"\
                            # f"train_B_f1: {train_B_metric[0]:.6f}\n"\
                            # f"train_B_recall: {train_B_metric[1]:.6f}\n"\
                            # f"train_B_iou: {train_B_metric[2]:.6f}\n"\
                            # f"train_f1: {train_metric[0]:.6f}\n"\
                            # f"train_recall: {train_metric[1]:.6f}\n"\
                            # f"train_iou: {train_metric[2]:.6f}\n"\

            val_info = f"val_sample_count: {val_sample_count//16:.6f}\n" \
                           f"val_loss: {val_loss:.4f}\n"\
                            f"val_rmse: {val_rmse:.6f}\n"\
                            # f"val_building_rmse: {val_building_rmse:.6f}\n"\
                            # f"val_B_f1: {val_B_metric[0]:.6f}\n"\
                            # f"val_B_recall: {val_B_metric[1]:.6f}\n"\
                            # f"val_B_iou: {val_B_metric[2]:.6f}\n"\
                            # f"val_f1: {val_metric[0]:.6f}\n"\
                            # f"val_recall: {val_metric[1]:.6f}\n"\
                            # f"val_iou: {val_metric[2]:.6f}\n"\

            f.write(train_info + val_info + "\n\n")

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args,
                         "log_var_a": log_var_a.data.item(),
                         "log_var_b": log_var_b.data.item(),
                         "log_var_c": log_var_c.data.item(),
                         "log_var_d": log_var_d.data.item()
                         }

        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="C:\\Users\\lenovo\\Desktop\\test\\ImageSets", help="sample root")  # "./data/ImageSets"
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--in_channel", default=7, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--encoder-lr', default=0.001, type=float, help='initial encoder part learning rate')
    parser.add_argument('--decoder-lr', default=0.01, type=float, help='initial decoder part learning rate')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--beta1', default=0.9, type=float, metavar='M1',
                        help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='M2',
                        help='beta2')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool, help="Use toh.rccuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("weights"):
        os.mkdir("weights")
    torch.cuda.empty_cache()

    main(args)
