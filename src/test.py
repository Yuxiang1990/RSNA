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

os.chdir("/home/mengdi/yuxiang.ye/kaggle")
weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/3_smooth_l1_1001_agument_0.108532.pth"  # 0.1128
# 0.1218 conf 0.4;  0.0613  0.1 ;  0.1307 0.5 ;
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/sgd_1002_finetune/9000__0.462225_best.pth"

weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/18000_adam_1002_0.103746_best.pth"
weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/23000_adam_1002_0.101644_best.pth"  # test 0.145

# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/sgd_1002_finetune/final.pth" # test 0.147

# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/adam_finetune/13000_0.097647_best.pth"

# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/8_smooth_l1_1001_agument_0.103552.pth"

# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/adam_new/26000_0.100196_best.pth"
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/adam_1004_solve_bug_finetune/22000_0.370154_best.pth"
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_hard/adam_1010_313_va/model_best.pth.tar"

# test 0.4(0.164) 0.3(0.164)
weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_finetune/yolov3_finetune/model_best.pth.tar"

# test 0.3(0.131)
# weights_path = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_finetune/yolov3_finetune_focalloss_b333/model_best.pth.tar"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parse = argparse.ArgumentParser()
parse.add_argument("--model_cfg_path", type=str, default='RSNA/config/yolov3.cfg', help="model config path")
parse.add_argument("--confidence", type=float, default=0.2, help="model confidence path")
parse.add_argument("--weights", type=str, default=weights_path, help="model weights path")

args = parse.parse_args()

model = Darknet(args.model_cfg_path, channel=3)
model.load_state_dict(torch.load(args.weights)['state_dict'])
print(model.module_list)
model.to(device)
print("Network successfully loaded")

test_data_path = "/home/mengdi/DataShare/kaggle/RSNA_Pneumonia_Dectection_Challenge/all/all_stage2/stage_2_test_images"
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

        # clg model
        # if uid not in preds_uids:
        #     test_list.append([uid, ""])
        #     continue

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
        heatmap_list, pred_bbox_list = model(imgs, device)

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
                plt.title("pred")
                plt.imshow(arr_pd, plt.cm.gray)

            test_list.append([uid, xywh_str])

        else:
            test_list.append([uid, ""])
            print(output.shape)
        if verbose:
            plt.plot()
    test = pd.DataFrame(test_list, columns=['patientId', 'PredictionString'])
    test.to_csv("/home/mengdi/yuxiang.ye/kaggle/evaluate/step2/test_0.2.csv", index=None)



