from torch.utils.data import Dataset
# from RSNA.lib.img2anchor_atten import Anchor
from RSNA.lib.img2anchor_atten import Anchor

import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from skimage import exposure, img_as_float
import matplotlib.pyplot as plt
import torch
import random
import torchvision.transforms as transforms

yolov3_config = {
                "anchors_list": [100, 200, 100, 100, 100, 50,
                                    200, 400, 200, 200, 200, 100,
                                        400, 800, 400, 400, 400, 200],
                # "anchors_list": [120, 240, 100, 100, 120, 80,
                #                     220, 440, 220, 220, 220, 146,
                #                         320, 640, 320, 320, 320, 214],
                "info_path": "/home/mengdi/yuxiang.ye/kaggle/RSNA/csv/info_stage2_new.csv",
                "feature_map": [64, 32, 16],
                "feature_name": ['high_reso', 'mid_reso', 'low_reso'],
                "anchor_num": 3  # each layer has (anchor_num) anchor
                 }


class2index = {"No Lung Opacity / Not Normal": 2,
               "Normal": 1,
               "Lung Opacity": 0
               }


class to_tensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return torch.from_numpy(data.copy()).float()


normalize = transforms.Normalize(mean=[0.49, 0.49, 0.49],
                                     std=[0.2294, 0.2294, 0.2294])

transform = transforms.Compose([to_tensor(), normalize])


class yolov3_dataset(Dataset):
    def __init__(self, info_path, subset=[0, 1, 2, 3], img_dim=512, origin_dim=1024, verbose=False, agu=False, clf=False):
        info_df = pd.read_csv(info_path)
        assert("subset" in info_df.columns)
        self.INFO = info_df.loc[info_df.subset.map(lambda x: x in subset)]
        tmp = self.INFO[['patientId', 'class']].drop_duplicates()
        self.uid_item_list = tmp.patientId.unique().tolist()
        self.class_list = tmp['class'].values.tolist()
        self.img_dim = img_dim
        self.ori_dim = origin_dim
        self.verbose = verbose
        self.agu = agu
        self.clf = clf

    def __getitem__(self, uid_item, wrap=20):
        dcm_path = self.INFO.loc[self.INFO.patientId == self.uid_item_list[uid_item], "path"].values[0]
        bboxs = self.INFO.loc[self.INFO.patientId == self.uid_item_list[uid_item],
                              ['x', 'y', 'width', 'height']].values
        clf = self.INFO.loc[self.INFO.patientId == self.uid_item_list[uid_item], 'class'].values[0]
        clf_index = np.array(class2index[clf])
        raw = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))[0]
        raw_resized = cv2.resize(raw, (self.img_dim, self.img_dim))
        random_agu = 1  # np.random.randint(2)
        random_hist = 0

        if self.agu and random_agu:
            # warpAffine
            rows, cols = raw_resized.shape
            offxyz = np.random.randint(-wrap, wrap, size=6)
            pts1 = np.float32([[0, 0], [0, cols-1], [rows-1, 0]])
            pts2 = np.float32([[0 + offxyz[0], 0 + offxyz[1]],
                               [0 + offxyz[2], cols-1 + offxyz[3]],
                               [rows-1 + offxyz[4], 0 + offxyz[5]]])

            M = cv2.getAffineTransform(pts1, pts2)
            AB = M[:, :2]
            C = M[:, 2]
            raw_resized_warp = cv2.warpAffine(raw_resized, M, (cols, rows))
            if random_hist:
                arr_hist = exposure.equalize_hist(raw_resized_warp)
            else:
                arr_hist = img_as_float(raw_resized_warp)
        else:
            arr_hist = img_as_float(raw_resized)
            # arr_hist = exposure.equalize_hist(raw_resized)

        anchor_dict = {}
        anchor_num = yolov3_config['anchor_num']
        for index, (f_map, f_name) in enumerate(zip(yolov3_config['feature_map'], yolov3_config['feature_name'])):
            if np.isnan(bboxs).sum():
                anchor_dict[f_name] = torch.from_numpy(np.zeros((anchor_num, f_map, f_map, 6))).float()
            else:
                if self.agu and random_agu:
                    # because of cv2.warpAffine
                    bboxs_wrap = bboxs.copy()
                    # tranform xy
                    bboxs_wrap[:, :2] = np.dot(bboxs_wrap[:, :2], AB.T)
                    bboxs_wrap[:, 0] += C[0]
                    bboxs_wrap[:, 1] += C[1]
                    # tranform wh
                    bboxs_wrap[:, 2] = AB[0, 0] * bboxs_wrap[:, 2] + AB[0, 1] * bboxs_wrap[:, 3]
                    bboxs_wrap[:, 3] = AB[1, 0] * bboxs_wrap[:, 2] + AB[1, 1] * bboxs_wrap[:, 3]
                else:
                    bboxs_wrap = bboxs

                anchors_list = yolov3_config['anchors_list'][index * 2 * anchor_num: (index + 1) * 2 * anchor_num]
                anchor = Anchor(f_map, self.ori_dim, bboxs_wrap, anchors_list=anchors_list)
                anchor_dict[f_name] = torch.from_numpy(anchor.img2anchor).float()

        if self.verbose:
            if not np.isnan(bboxs).sum():
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                for x, y, w, h in bboxs:
                    x1 = int(x / 2)
                    y1 = int(y / 2)
                    x2 = int((x + w) / 2)
                    y2 = int((y + h) / 2)
                    raw_resized = cv2.rectangle(raw_resized, (x1, y1), (x2, y2), 1, 1)
                plt.title("min:{0:} max:{1:}".format(raw_resized.min(), raw_resized.max()))
                plt.imshow(raw_resized, plt.cm.gray)
                plt.subplot(1, 2, 2)
                for x, y, w, h in bboxs_wrap:
                    x1 = int(x / 2)
                    y1 = int(y / 2)
                    x2 = int((x + w) / 2)
                    y2 = int((y + h) / 2)
                    arr_hist = cv2.rectangle(arr_hist, (x1, y1), (x2, y2), 1, 1)
                plt.title("min:{0:} max:{1:}".format(arr_hist.min(), arr_hist.max()))
                plt.imshow(arr_hist, plt.cm.gray)
                plt.plot()

        arr_hist = np.expand_dims(arr_hist, axis=0)
        arr_hist = np.repeat(arr_hist, 3, axis=0)
        if self.clf:
            return transform(arr_hist).float(), anchor_dict, torch.LongTensor(clf_index)
        else:
            return transform(arr_hist).float(), anchor_dict

    def __len__(self):
        return len(self.uid_item_list)


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)


class yolov3_batchsample(object):
    def __init__(self, class_list, batch_size=[4, 4, 4]):
        self.class_list = class_list
        self.batch_size = batch_size
        self.Lung_Opacity_list = []
        self.Normal_list = []
        self.Unnormal_list = []
        for index, cls in enumerate(self.class_list):
            if cls == "Normal":
                self.Normal_list.append(index)
            elif cls == "Lung Opacity":
                self.Lung_Opacity_list.append(index)
            else:
                self.Unnormal_list.append(index)
        self.iter_list = []
        self.len = len(self.Lung_Opacity_list) // (batch_size[0])
        self.iter_list.append(shuffle_iterator(self.Lung_Opacity_list))
        self.iter_list.append(shuffle_iterator(self.Normal_list))
        self.iter_list.append(shuffle_iterator(self.Unnormal_list))

    def __iter__(self):
        batch = []
        for _ in range(self.len):
            for index, bz in enumerate(self.batch_size):
                iterator = self.iter_list[index]
                for _ in range(bz):
                    iterator_index = next(iterator)
                    batch.append(iterator_index)
            random.shuffle(batch)
            yield batch
            batch = []

    def __len__(self):
        return self.len


if __name__ == "__main__":
    train_datset = yolov3_dataset(yolov3_config['info_path'], subset=[0, 1, 2, 3], verbose=True, agu=True)
    for i in range(100):
        arr, anchor_dict = train_datset[i]
        print(arr.size(), anchor_dict['high_reso'].size())
    # batch_sample = yolov3_batchsample(train_datset.class_list)
    # print(list(batch_sample))
    # print(len(list(batch_sample)))
