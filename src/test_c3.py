from __future__ import division
import torch
from torch.utils.data import DataLoader
import argparse
import os
import sys
sys.path.insert(0, "/home/mengdi/yuxiang.ye/kaggle")
from RSNA.model.darknet import Darknet
from RSNA.lib.utils import nmi_supression
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import numpy as np
from glob import glob
from skimage import exposure, img_as_float
import torchvision.transforms as transforms
import torch.nn as nn

os.chdir("/home/mengdi/yuxiang.ye/kaggle")
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_1013/1014_focalloss_clf/model_best.pth.tar"
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_1015/1015_focalloss_clf_focal_0.1_bce/model_best.pth.tar"
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_finetune/yolov3_finetune/model_best.pth.tar"
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/1018weights/1017_finetune_best_fmeasure_focal_nohist.tar"


# 1000 0.3 0.2+
weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_stage2/" \
               "test1_bce/checkpoint_5_0.49051_0.7792837834762214.pth.tar"

weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_stage2" \
               "/test1_bce_lr_schedule1_from_scratch_dp1_wd1e5/checkpoint_21_0.55704_0.766275421610362.pth.tar"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parse = argparse.ArgumentParser()
parse.add_argument("--model_cfg_path", type=str, default='RSNA/config/yolov3.cfg', help="model config path")
parse.add_argument("--confidence", type=float, default=0.1, help="model confidence path")
parse.add_argument("--weights", type=str, default=weights_path, help="model weights path")

args = parse.parse_args()

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
        self.extract_feature = None

    def forward(self, x, device, target=None, clf_target=None, num_class=3):
        self.loss = self.loss.to(device)
        x, dark_loss = self.darknet(x, device, target)
        b_num = x.size(0)
        x = self.clf_net(x)
        self.extract_feature = x.view(b_num, -1)

        x = self.Linear(self.extract_feature)
        if clf_target is not None:
            clf_target = clf_target.to(device)
            loss_clf = self.loss(x, clf_target)
            metrics_dict = calc_metrics(x, clf_target)
            return metrics_dict, loss_clf, dark_loss
        else:
            return torch.max(torch.nn.Softmax(dim=1)(x), dim=1)[
                       1].item(), dark_loss, self.extract_feature.squeeze().cpu().numpy()


darknet = Darknet(args.model_cfg_path, channel=3, clf=True, droprate=None)
# darknet.load_state_dict(torch.load(args.weights, map_location='cpu')['state_dict'])
model = my_model(darknet.to(device))
model.load_state_dict(torch.load(args.weights, map_location='cpu')['state_dict'])
print("Network successfully loaded")
model.to(device)
test_data_path = "/home/mengdi/DataShare/kaggle/RSNA_Pneumonia_Dectection_Challenge/all/all_stage2/stage_2_test_images"
# test_data_path = "/home/mengdi/DataShare/kaggle/RSNA_Pneumonia_Dectection_Challenge/all//stage_1_test_images"

test_files = glob(test_data_path + "/*.dcm")

model.eval()
test_list = []
verbose = 1


class to_tensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return torch.from_numpy(data.copy()).float()


normalize = transforms.Normalize(mean=[0.49, 0.49, 0.49],
                                     std=[0.2294, 0.2294, 0.2294])

transform = transforms.Compose([to_tensor(), normalize])

with torch.no_grad():
    for uid_num, dcmfile in enumerate(test_files):
        uid = os.path.basename(dcmfile).replace(".dcm", "")

        print(uid_num, dcmfile)
        image = sitk.ReadImage(dcmfile)
        arr = sitk.GetArrayFromImage(image)[0]
        raw_resized = cv2.resize(arr, (512, 512))
        raw_resized = img_as_float(raw_resized)
        raw_resized = np.expand_dims(raw_resized, axis=0)
        raw_resized = np.repeat(raw_resized, 3, axis=0)
        imgs = transform(raw_resized)
        imgs = imgs.view(1, 3, 512, 512)
        imgs = imgs.to(device)

        pred_label, (heatmap_list, pred_bbox_list), f1024 = model(imgs, device)

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
            xywh_str = xywh_str.strip()
            print(xywh_str)

            if verbose:
                arr_pd = arr.copy()
                for x1, y1, x2, y2, _ in output:
                    x1 = int(x1 * 2)
                    y1 = int(y1 * 2)
                    x2 = int(x2 * 2)
                    y2 = int(y2 * 2)
                    arr_pd = cv2.rectangle(arr_pd, (x1, y1), (x2, y2), 255, 1)
                plt.figure(figsize=(10, 10))
                plt.title("pred{0:}".format(pred_label))
                plt.imshow(arr_pd, plt.cm.gray)

            test_list.append([uid, xywh_str])  # + f1024.tolist())

        else:
            test_list.append([uid, ""])  # + f1024.tolist())
            print(output.shape)
        if verbose:
            plt.plot()
    test = pd.DataFrame(test_list, columns=['patientId', 'PredictionString'])  #  + ["f{0:}".format(i) for i in range(1024)])
    test.to_csv("/home/mengdi/yuxiang.ye/kaggle/evaluate/step2/test_28_from_scratch.csv", index=None)
    # test.to_csv("/home/mengdi/yuxiang.ye/kaggle/step2/test_1028_from_scratch.csv", index=None)

