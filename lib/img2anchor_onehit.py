from RSNA.lib.utils import bbox_iou
import numpy as np
import math
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import pandas as pd
anchors_list = [100, 200, 100, 100, 100, 50,
          200, 400, 200, 200, 200, 100,
          400, 800, 400, 400, 400, 200]
# anchors_list = [120, 240, 100, 100, 120, 80,
#                 220, 440, 220, 220, 220, 146,
#                 320, 640, 320, 320, 320, 214]

class Anchor(object):
    def __init__(self, feature_dim, img_dim, bbox_arr, anchors_list=anchors_list):
        """
        :param feature_dim:
        :param img_dim:
        :param bbox_arr: format x, y, w, h
        :param anchors_list: list w1, h1, w2, h2 ...
        """
        self.feature_dim = feature_dim
        self.stride = img_dim / feature_dim
        anchors_list = np.array(anchors_list) / self.stride
        self.anchors_scaled = [(anchors_list[i], anchors_list[i + 1]) for i in np.arange(0, len(anchors_list), 2)]
        self.bbox_list = bbox_arr / self.stride
        # x, y, w, h, conf, conf_mask, xywh_mask
        self.xywhp_arr = np.zeros((len(self.anchors_scaled), self.feature_dim, self.feature_dim, 7))
        self.xywhp_arr[..., 5] = 1

    @property
    def img2anchor(self, iou_th=0.4):
        x, y = np.meshgrid(range(self.feature_dim), range(self.feature_dim))
        x_squeeze = x.reshape(-1, 1)
        y_squeeze = y.reshape(-1, 1)
        xy_center_coords = np.hstack([x_squeeze, y_squeeze])
        anchors_arr = np.array(self.anchors_scaled)
        for bx, by, bw, bh in self.bbox_list:
            bbox_gt = (bx, by, bx+bw, by+bh)
            center_x = bx + bw / 2
            center_y = by + bh / 2

            anchor_arr_xywh = np.zeros((anchors_arr.shape[0], 4))
            anchor_arr_xywh[:, 0] = int(center_x) - anchors_arr[:, 0] / 2
            anchor_arr_xywh[:, 1] = int(center_y) - anchors_arr[:, 1] / 2
            anchor_arr_xywh[:, 2] = int(center_x) + anchors_arr[:, 0] / 2
            anchor_arr_xywh[:, 3] = int(center_y) + anchors_arr[:, 1] / 2

            iou_map = bbox_iou(anchor_arr_xywh, bbox_gt)
            anchor_ignore_index = np.nonzero(iou_map > iou_th)[0].tolist()
            anchor_max_index = np.argmax(iou_map)
            # if anchor_max_index in anchor_ignore_index:
            #     anchor_ignore_index.remove(anchor_max_index)
            # remove anchor_ignore_index according max

            for anchor_index, (anchor_w, anchor_h) in enumerate(anchors_arr):
                xy_left_coords = xy_center_coords - (anchor_w / 2, anchor_h / 2)
                xy_right_coords = xy_center_coords + (anchor_w / 2, anchor_h / 2)
                bbox_anchor = np.hstack([xy_left_coords, xy_right_coords])
                iou_map_tmp = bbox_iou(bbox_anchor, bbox_gt)
                iou_map_arr = iou_map_tmp.reshape(self.feature_dim, -1)
                # conf
                self.xywhp_arr[anchor_index, :, :, 4] = np.maximum(iou_map_arr,
                                                                   self.xywhp_arr[anchor_index, :, :, 4])
                if (anchor_index == anchor_max_index) and (iou_map[anchor_index] > iou_th):
                    # xywh mask
                    self.xywhp_arr[anchor_index, int(center_y), int(center_x), 6] = 1
                    continue
                if anchor_index in anchor_ignore_index:
                    # conf mask
                    self.xywhp_arr[anchor_index, :, :, 5] = np.minimum(self.xywhp_arr[anchor_index, :, :, 5],
                                                                       iou_map_arr < iou_th)

            # dx
            self.xywhp_arr[:, int(center_y), int(center_x), 0] = center_x - int(center_x)
            # dy
            self.xywhp_arr[:, int(center_y), int(center_x), 1] = center_y - int(center_y)
            # dw
            self.xywhp_arr[:, int(center_y), int(center_x), 2] = list(map(lambda x:
                                                                          math.log(bw / x + 1e-16), anchors_arr[:, 0]))
            # dh
            self.xywhp_arr[:, int(center_y), int(center_x), 3] = list(map(lambda x:
                                                                          math.log(bw / x + 1e-16), anchors_arr[:, 1]))

        return self.xywhp_arr

    def plot_bbox(self, xywhp_arr, dcm_path, title):
        plt.figure(figsize=(15, 10))
        anchors_num = xywhp_arr.shape[0] * 2
        raw = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))[0]
        arr_anchor = cv2.resize(raw, (self.feature_dim, self.feature_dim))
        for x, y, w, h in self.bbox_list:
            arr_anchor = cv2.rectangle(arr_anchor, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 1)
        for index, xywhp in enumerate(xywhp_arr):
            plt.subplot(anchors_num // 3, 3, index + 1)
            y, x = np.where(xywhp[:, :, 2])
            info = np.round(np.squeeze(xywhp[y, x]), 3)
            plt.title("{0:}_{1:}".format(info, title))
            plt.imshow(arr_anchor, plt.cm.gray)
            plt.imshow(xywhp[:, :, 4], alpha=0.15)
            plt.subplot(anchors_num // 3, 3, index + 4)
            plt.imshow(arr_anchor, plt.cm.gray)
            plt.title("{0:}_{1:}".format(np.max(xywhp[:, :, 5]), np.min(xywhp[:, :, 5])))
            plt.imshow(xywhp[:, :, 5], alpha=0.15)
        plt.show()


if __name__ == "__main__":
    info = pd.read_csv('/home/mengdi/yuxiang.ye/kaggle/RSNA/csv/info.csv')
    info_pos = info.loc[info.Target == 1]
    uids = info_pos.patientId.unique()
    # uid = np.random.choice(uids)
    # uid = uids[0]
    # print(uid)
    for uid in uids:
        print(uid)
        bboxs = info_pos.loc[info_pos.patientId == uid, ['x', 'y', 'width', 'height']].values
        dcmpath = info_pos.loc[info_pos.patientId == uid, 'path'].values[0]
        for index, fsize, reso in zip(list(range(3)), [64, 32, 16], ['high', 'mid', 'low']):
            anchor = Anchor(fsize, 1024, bboxs, anchors_list=anchors_list[index*6: (index+1)*6])
            anchor.plot_bbox(anchor.img2anchor, dcmpath, reso)
