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
        self.xywhp_arr = np.zeros((len(self.anchors_scaled), self.feature_dim, self.feature_dim, 6))

    @property
    def img2anchor(self):
        x, y = np.meshgrid(range(self.feature_dim), range(self.feature_dim))
        x_squeeze = x.reshape(-1, 1)
        y_squeeze = y.reshape(-1, 1)
        xy_center_coords = np.hstack([x_squeeze, y_squeeze])
        for index, anchor in enumerate(self.anchors_scaled):
            iou_map_arr_bak = np.zeros((self.feature_dim, self.feature_dim))
            for bx, by, bw, bh in self.bbox_list:

                # calc iou
                anchor_w, anchor_h = anchor
                # print("*" * 20)
                # print("anchor w,h", anchor_w, anchor_h)
                # print("bbox w,h", bw, bh, "dw", math.log(bw / anchor_w), "dh", math.log(bh / anchor_h))
                xy_left_coords = np.maximum(xy_center_coords - (anchor_w / 2, anchor_h / 2), 0)
                xy_right_coords = np.minimum(xy_center_coords + (anchor_w / 2, anchor_h / 2), self.feature_dim)
                bbox_anchor = np.hstack([xy_left_coords, xy_right_coords])
                bbox_gt = (bx, by, bx+bw, by+bh)
                iou_map = bbox_iou(bbox_anchor, bbox_gt)
                iou_map_arr = iou_map.reshape(self.feature_dim, -1)
                # calc dx, dy, dw, dh (0,1)
                center_x = bx + bw / 2
                center_y = by + bh / 2
                # dx
                self.xywhp_arr[index, int(center_y), int(center_x), 0] = center_x - int(center_x)
                # dy
                self.xywhp_arr[index, int(center_y), int(center_x), 1] = center_y - int(center_y)
                # dw
                self.xywhp_arr[index, int(center_y), int(center_x), 2] = math.log(bw / anchor_w + 1e-16)
                # dh
                self.xywhp_arr[index, int(center_y), int(center_x), 3] = math.log(bh / anchor_h + 1e-16)
                self.xywhp_arr[index, :, :, 4] = np.maximum(iou_map_arr, iou_map_arr_bak)
                self.xywhp_arr[index, int(center_y), int(center_x), 5] = 1
                iou_map_arr_bak = iou_map_arr

        return self.xywhp_arr

    def plot_bbox(self, xywhp_arr, dcm_path, title):
        plt.figure(figsize=(15, 10))
        anchors_num = xywhp_arr.shape[0]
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
        plt.show()


if __name__ == "__main__":
    info = pd.read_csv('/home/mengdi/yuxiang.ye/kaggle/RSNA/csv/info.csv')
    info_pos = info.loc[info.Target == 1]
    uids = info_pos.patientId.unique()
    # uid = np.random.choice(uids)
    # uid = uids[0]
    # print(uid)
    for uid in uids:
        bboxs = info_pos.loc[info_pos.patientId == uid, ['x', 'y', 'width', 'height']].values
        dcmpath = info_pos.loc[info_pos.patientId == np.random.choice(uids), 'path'].values[0]
        for index, fsize, reso in zip(list(range(3)), [64, 32, 16], ['high', 'mid', 'low']):
            anchor = Anchor(fsize, 1024, bboxs, anchors_list=anchors_list[index*6: (index+1)*6])
            anchor.plot_bbox(anchor.img2anchor, dcmpath, reso)
