from __future__ import division
import torch
from torch.utils.data import DataLoader
import argparse
import sys
sys.path.insert(0, "/home/mengdi/yuxiang.ye/kaggle")
import os
from RSNA.model.darknet import Darknet
from RSNA.lib.dataloader_c3 import yolov3_dataset, yolov3_config
from RSNA.lib.utils import nmi_supression
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import numpy as np
import pickle as pkl
import random
import torch.nn as nn
colors = pkl.load(open("pallete", "rb"))

os.chdir("/home/mengdi/yuxiang.ye/kaggle")

# .5 val:0.1809 .4 val:0.1581 .6 val: 0.1102
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_1013/1013_focalloss/model_best.pth.tar"

# .5 val:0.1363
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_finetune/yolov3_finetune/model_best.pth.tar"

# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_finetune/yolov3_finetune_focalloss_b333/model_best.pth.tar"

weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_1013/1014_focalloss_clf/model_best.pth.tar"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parse = argparse.ArgumentParser()
parse.add_argument("--model_cfg_path", type=str, default='RSNA/config/yolov3.cfg', help="model config path")
parse.add_argument("--confidence", type=float, default=0.4, help="model confidence path")
parse.add_argument("--weights", type=str, default=weights_path, help="model weights path")

args = parse.parse_args()

darknet = Darknet(args.model_cfg_path, channel=3, clf=True)


def calc_metrics(output, target, class_num=3):
    metrics_dict = {}
    pred = torch.max(torch.nn.Softmax(dim=1)(output), dim=1)[1]
    acc = torch.sum(torch.eq(pred, target)).item() / target.size(0)
    for i in range(class_num):
        recall = torch.sum((pred == i) * (target == i)).item() / (torch.sum(target == i).item() + 1e-8)
        precison = torch.sum((pred == i) * (target == i)).item() / (torch.sum(pred == i).item() + 1e-8)
        metrics_dict["recall%d" % i] = recall
        metrics_dict["precision%d" % i] = precison
    metrics_dict['acc'] = acc
    return metrics_dict


class my_model(nn.Module):
    def __init__(self, darknet, filter_num=1024, num_class=3):
        super(my_model, self).__init__()
        self.darknet = darknet
        self.loss = nn.CrossEntropyLoss()
        self.clf_net = nn.Sequential()
        self.clf_net.add_module("clf_con1", nn.Conv2d(filter_num, filter_num, kernel_size=3, stride=2))
        self.clf_net.add_module("clf_bn1", nn.BatchNorm2d(filter_num))
        self.clf_net.add_module("clf_active1", nn.LeakyReLU(0.1, inplace=True))
        self.clf_net.add_module("clf_con2", nn.Conv2d(filter_num, filter_num, kernel_size=3, stride=2))
        self.clf_net.add_module("clf_bn2", nn.BatchNorm2d(filter_num))
        self.clf_net.add_module("clf_active2", nn.LeakyReLU(0.1, inplace=True))
        self.clf_net.add_module("ap", nn.AdaptiveAvgPool2d(1))
        self.Linear = nn.Linear(filter_num, num_class)

    def forward(self, x, device, target=None, clf_target=None, num_class=3):
        self.loss = self.loss.to(device)
        x, dark_loss = self.darknet(x, device, target)
        b_num = x.size(0)
        x = self.clf_net(x)
        x = x.view(b_num, -1)
        x = self.Linear(x)
        if clf_target is not None:
            clf_target = clf_target.to(device)
            loss_clf = self.loss(x, clf_target)
            metrics_dict = calc_metrics(x, clf_target)
            return metrics_dict, loss_clf, dark_loss
        else:
            return torch.max(torch.nn.Softmax(dim=1)(x), dim=1)[1].item(),  dark_loss


model = my_model(darknet.to(device))
print("Network successfully loaded")

"""
solve bug, if saved model in gpu and the gpu is not avaliable, when load model, may cause out of memory
"""
model.load_state_dict(torch.load(args.weights, map_location='cpu')['state_dict'])
model.to(device)
print("Network successfully loaded")

# create dataloader
val_datset = yolov3_dataset(yolov3_config['info_path'], subset=[4], clf=True)
val_dataloader = DataLoader(val_datset, batch_size=1, shuffle=False)
val_df = pd.read_csv("/home/mengdi/yuxiang.ye/kaggle/RSNA/csv/info.csv")
val_df = val_df.loc[val_df.subset.map(lambda x: x in [4])]


data_path_root = "/home/mengdi/DataShare/kaggle/RSNA_Pneumonia_Dectection_Challenge/all"
train_data_path = os.path.join(data_path_root, "stage_1_train_images")

# val_datset = yolov3_dataset(yolov3_config['info_path'], subset=[0,1,2,3])
# val_df = pd.read_csv("/home/mengdi/yuxiang.ye/kaggle/RSNA/csv/info.csv")
# val_df = val_df.loc[val_df.subset.map(lambda x: x in [0, 1, 2, 3])]
# val_dataloader = DataLoader(val_datset, batch_size=1, shuffle=False)

uids = val_df.patientId.unique()
model.eval()
val_list = []
verbose = 1

with torch.no_grad():
    for uid_num, ((imgs, targets, targets_label), uid) in enumerate(zip(val_dataloader, uids)):
        print(uid_num, uid)
        imgs = imgs.to(device)
        pred_label, (heatmap_list, pred_bbox_list) = model(imgs, device)
        if verbose:
            # check hitmap
            plt.figure(figsize=(10, 10))
            for index, (feature, title) in enumerate(zip(heatmap_list, ['low', 'mid', 'high'])):
                plt.subplot(3, 3, 3 * index + 1)
                plt.title(title + "%.4f" % (torch.max(feature[0, :, :, 0, 4]).item()))
                plt.imshow(feature[0, :, :, 0, 4])

                plt.subplot(3, 3, 3 * index + 2)
                plt.title(title + "%.4f" % (torch.max(feature[0, :, :, 1, 4]).item()))
                plt.imshow(feature[0, :, :, 1, 4])

                plt.subplot(3, 3, 3 * index + 3)
                plt.title(title + "%.4f" % (torch.max(feature[0, :, :, 2, 4]).item()))
                plt.imshow(feature[0, :, :, 2, 4])

        if verbose:
            image = sitk.ReadImage(os.path.join(train_data_path, uid + ".dcm"))
            arr = sitk.GetArrayFromImage(image)[0]
            bboxes = val_df.loc[val_df.patientId == uid, ['x', 'y', 'width', 'height']].values
            if not np.isnan(bboxes).sum():
                for x, y, w, h in bboxes:
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    arr_gt = cv2.rectangle(arr, (x, y), (x+w, y+h), 255, 1)
                # plt.figure(figsize=(10, 10))
                # plt.title("gt")
                # plt.imshow(arr_gt, plt.cm.gray)


        # nmi
        output = nmi_supression(pred_bbox_list.cpu(), args.confidence)

        if output.size(0):
            # x1, y1, w, h, remove batch_id
            output = output[:, 1:].numpy()
            # => x1, y1, center_x, center_y
            # print(output)
            xywh = output.copy()

            xywh[:, 2] = (xywh[:, 2] - xywh[:, 0]) * 2
            xywh[:, 3] = (xywh[:, 3] - xywh[:, 1]) * 2
            xywh[:, 0] = xywh[:, 0] * 2
            xywh[:, 1] = xywh[:, 1] * 2
            xywh = xywh[:, [4, 0, 1, 2, 3]]
            xywh_squeeze = xywh.flatten()
            xywh_str = ""
            for i in xywh_squeeze:
                xywh_str += str(i) + " "
            print(xywh_str)

            val_list.append([uid, xywh_str, pred_label, targets_label.item()])

            if verbose:
                for x1, y1, x2, y2, p in output:
                    x1 = int(x1 * 2)
                    y1 = int(y1 * 2)
                    x2 = int(x2 * 2)
                    y2 = int(y2 * 2)
                    color = random.choice(colors)
                    arr_pd = cv2.rectangle(arr, (x1, y1), (x2, y2), color, 1)
                    t_size = cv2.getTextSize("{0:.5f}".format(p), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(arr, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
                    cv2.putText(arr, "{0:.5f}".format(p), (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

                plt.figure(figsize=(10, 10))
                plt.title("gt255_pred0_{0:}_{1:}".format(pred_label, targets_label.item()))
                plt.imshow(arr_pd, plt.cm.gray)
        else:
            val_list.append([uid, "", pred_label, targets_label.item()])
            print(output.shape)
        if verbose:
            plt.show()
    val = pd.DataFrame(val_list, columns=['patientId', 'PredictionString', 'label', 'gt'])
    val.to_csv("/home/mengdi/yuxiang.ye/kaggle/evaluate/val_submission_clf.csv", index=None)



