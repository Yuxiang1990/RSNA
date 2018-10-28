import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from skimage.transform import resize
import random
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import Sampler

info = "/home/mengdi/yuxiang.ye/kaggle/RSNA/csv/info.csv"
data_path_root = "/home/mengdi/DataShare/kaggle/RSNA_Pneumonia_Dectection_Challenge/all"
train_data_path = os.path.join(data_path_root, "stage_1_train_images")

train_gan_path = "/home/mengdi/yuxiang.ye/kaggle/step2/GAN_train.csv"
val_gan_path = "/home/mengdi/yuxiang.ye/kaggle/step2/GAN_val.csv"


class random_rot_flip(object):
    def __init__(self):
        pass

    def __call__(self, data):
        flip = np.random.randint(2, size=2)
        if flip[0]:
            data = np.flip(data, axis=0)
        if flip[1]:
            data = np.flip(data, axis=1)
        random_rot = np.random.randint(4)
        data = np.rot90(data, random_rot)
        return data


class to_tensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return torch.from_numpy(data.copy()).float()


normalize = transforms.Normalize(mean=[0.49, 0.49, 0.49],
                                     std=[0.2294, 0.2294, 0.2294])

transform = transforms.Compose([to_tensor(), normalize])


class gan_loader(Dataset):
    def __init__(self, yolov3_candidates_p, img_dim=(224, 224), debug=False, warp=False, rot_flip=False):
        self.cand_df = pd.read_csv(yolov3_candidates_p)[:100]
        self.img_dim = img_dim
        self.labels = self.cand_df['GAN_label'].values
        self.debug = debug
        self.warp = warp
        self.rotflip = rot_flip

    def __getitem__(self, index):
        seriesuid = self.cand_df.loc[index, 'seriesuid']
        label = int(self.cand_df.loc[index, 'GAN_label'])
        x, y, w, h = self.cand_df.loc[index, ['x', 'y', 'w', 'h']].values

        # get raw arr from dcm path
        image = sitk.ReadImage(os.path.join(train_data_path, seriesuid + ".dcm"))
        arr = sitk.GetArrayFromImage(image)[0]

        # double w and h, generate left-up coord and right-down coord
        x1 = max(x - w // 2, 0)
        y1 = max(y - h // 2, 0)
        x2 = min(x + w + w // 2, arr.shape[0])
        y2 = min(y + h + h // 2, arr.shape[0])
        if self.warp:
            warp_size = 10
            rows, cols = arr.shape
            offxyz = np.random.randint(-warp_size, warp_size, size=6)
            pts1 = np.float32([[0, 0], [0, cols - 1], [rows - 1, 0]])
            pts2 = np.float32([[0 + offxyz[0], 0 + offxyz[1]],
                               [0 + offxyz[2], cols - 1 + offxyz[3]],
                               [rows - 1 + offxyz[4], 0 + offxyz[5]]])

            M = cv2.getAffineTransform(pts1, pts2)
            AB = M[:, :2]
            C = M[:, 2]
            arr = cv2.warpAffine(arr, M, (cols, rows))

        arr_crop = arr[y1:y2, x1:x2]
        cropw, croph = arr_crop.shape
        dim_diff = np.abs(cropw - croph)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        pad = ((pad1, pad2), (0, 0)) if h <= w else ((0, 0), (pad1, pad2))
        # Add padding
        arr_crop_pad = np.pad(arr_crop, pad, 'constant', constant_values=0)
        # resized
        arr_crop_resized = resize(arr_crop_pad, self.img_dim, mode='reflect')
        if self.rotflip:
            arr_crop_resized = random_rot_flip()(arr_crop_resized)

        # print(seriesuid, arr_crop_resized.max(), arr_crop_resized.min())
        if self.debug:
            return seriesuid, arr_crop_resized, label, arr
        else:
            arr_crop_resized = np.expand_dims(arr_crop_resized, axis=0)
            arr_crop_resized = np.repeat(arr_crop_resized, 3, axis=0)
            return transform(arr_crop_resized), label

    @classmethod
    def GAN_plot(cls, info, *args, **kwargs):
        assert(kwargs['debug'])
        info_df = pd.read_csv(info)
        dataset = cls(*args, **kwargs)
        randon_index = np.random.choice(range(len(dataset)))
        seriesuid, arr_crop, label, arr = dataset[randon_index]

        plt.figure(figsize=(8, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(arr_crop, plt.cm.gray)
        plt.title(seriesuid + " " + str(label))

        plt.subplot(1, 2, 2)
        for i, row in info_df.loc[info_df.patientId == seriesuid].iterrows():
            x, y, w, h = row[["x", "y", "width", "height"]]
            if x > 0:
                x, y, w, h = row[["x", "y", "width", "height"]].astype(int)
                arr = cv2.rectangle(arr, (x, y), (x + w, y + h), 255, 2)
        plt.imshow(arr, plt.cm.gray)
        plt.title("gt")
        plt.show()

    def __len__(self):
        return self.cand_df.shape[0]


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


class gan_batchsample(object):
    def __init__(self, class_list, batch_size=[4, 4]):
        self.class_list = class_list
        self.batch_size = batch_size
        Lung_Opacity_list = []
        unLung_Opacity_list = []
        for index, cls in enumerate(self.class_list):
            if cls:
                Lung_Opacity_list.append(index)
            else:
                unLung_Opacity_list.append(index)
        self.iter_list = []
        self.iter_list.append(shuffle_iterator(Lung_Opacity_list))
        self.iter_list.append(shuffle_iterator(unLung_Opacity_list))
        self.len = len(Lung_Opacity_list) // (batch_size[0])

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


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices, batch_indices):
        self.indices = indices
        self.batch_indices = batch_indices

    def __iter__(self):
        return (self.indices[i] for i in self.batch_indices)

    def __len__(self):
        return len(self.indices)


# define train dataloader
train_dataset = gan_loader(train_gan_path, warp=True, rot_flip=True, img_dim=(224, 224))
train_sample = gan_batchsample(train_dataset.labels)
train_dataloader = DataLoader(train_dataset, batch_sampler=train_sample)

# define val dataloader
val_dataset = gan_loader(val_gan_path, img_dim=(224, 224))
val_sample = gan_batchsample(val_dataset.labels)
val_dataloader = DataLoader(val_dataset, batch_sampler=val_sample)


# define magnet dataloader
class magnet_dataloader:
    def __init__(self, batch_size, dataset):
        print("dataset length", len(dataset))
        self.dataset = dataset
        self.batch_size = batch_size
        self.sample = SubsetSequentialSampler(range(len(dataset)), range(batch_size))

    @property
    def magnet_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sample)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    # train_dataset = gan_loader(train_gan_path)
    # train_sample = gan_batchsample(train_dataset.labels)
    # train_dataloader = DataLoader(train_dataset, batch_sampler=train_sample)
    # for img, label in train_dataloader:
    #     print(img.shape, label)

    # for _ in range(100):
    #     gan_loader.GAN_plot(info, "/home/mengdi/yuxiang.ye/kaggle/step2/GAN_train.csv",
    #                         debug=True, warp=True, rot_flip=True)

    train_loader = magnet_dataloader(8, train_dataset)
    for img, label in train_loader.magnet_loader:
        print(img.size(), label)
