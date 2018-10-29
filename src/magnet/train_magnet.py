import torch.nn as nn
import torch.optim as optim
import argparse
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch
import shutil
from RSNA.lib.gan_dataloader import magnet_dataloader  # ,  train_dataset, val_dataset
import time
import os
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from RSNA.lib.magnet_utils.cfg import magnet_setting
from RSNA.lib.magnet_utils.magnet_loss import MagnetLoss
from RSNA.lib.magnet_utils.magnet_tools import ClusterBatchBuilder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from RSNA.model.lenet import LeNet


cudnn.benchmark = True
os.chdir("/home/mengdi/yuxiang.ye/kaggle")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description="pytorch densenet169 finetune")
parser.add_argument("--epochs", type=int,  default=500, help="number of epoch")
parser.add_argument('--arch', '-a', default="densenet169", choices=model_names)
parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('--bs', '--batch_size', type=int, default=64)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--n_cpu', type=int, default=8)
parser.add_argument('--feature_n', type=int, default=2)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument("--checkpoint_dir", type=str, default="RSNA/checkpoints_magnet",
                    help="directory where model weights save")
parser.add_argument("--tensorboardX_log", type=str, default="RSNA/Tensorboardx_magnet",
                    help="directory where model log save")
args = parser.parse_args()
print(args.__dict__)

# define model
pretrained_model = models.__dict__[args.arch](pretrained=True, drop_rate=0)


detail_log = "densenet169_magnet_loss_f4"
os.makedirs(args.tensorboardX_log, exist_ok=True)
os.makedirs(os.path.join(args.checkpoint_dir, detail_log))
writer = SummaryWriter(os.path.join(args.tensorboardX_log, detail_log))


class my_model(nn.Module):
    def __init__(self, pretrain_model):
        super(my_model, self).__init__()
        self.pretrain_model = pretrain_model
        self.clf_layer = nn.Linear(in_features=pretrain_model.classifier.out_features, out_features=args.feature_n)

    def forward(self, x):
        x = self.pretrain_model(x)
        clf = self.clf_layer(x)
        return clf


model = my_model(pretrained_model)
# model.load_state_dict(torch.load("/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_GAN"
#                                  "/densenet169_adam_dp0_agu/model_best.pth.tar")['state_dict'])
model = LeNet(2)
model.to(device)
# model = torch.nn.DataParallel(model).to(device)
# criterion = nn.CrossEntropyLoss().to(device)
criterion = MagnetLoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
lr_schedule = ReduceLROnPlateau(optimizer, 'max', factor=0.3, patience=20)

# define dataloader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


def load_dataset():
    # MNIST dataset
    trans_img = transforms.Compose([
            transforms.ToTensor()
        ])

    print("Downloading MNIST data...")
    trainset = MNIST('/home/mengdi/yuxiang.ye/siamese-triplet-master/data/MNIST',
                     train=True, transform=trans_img, download=False)
    valset = MNIST('/home/mengdi/yuxiang.ye/siamese-triplet-master/data/MNIST',
                    train=False, transform=trans_img, download=False)
    return trainset, valset


train_dataset, val_dataset = load_dataset()
train_dataloader = magnet_dataloader(args.bs, train_dataset).magnet_loader
val_dataloader = magnet_dataloader(args.bs, val_dataset).magnet_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint_%d_%.5f.pth.tar'):
    epoch = state['epoch']
    acc = state['acc_best']
    checkpath = os.path.join(args.checkpoint_dir, detail_log, filename % (epoch, acc))
    bestpath = os.path.join(args.checkpoint_dir, detail_log, "model_best.pth.tar")
    if is_best:
        torch.save(state, checkpath)
        shutil.copyfile(checkpath, bestpath)


def compute_reps(model, dataset, chunk_size=16):
    reps = []
    loader = DataLoader(dataset, batch_size=chunk_size, shuffle=False, num_workers=32)
    for batch_idx, (img, target) in tqdm(enumerate(loader)):
        img = img.to(device)
        output, _ = model(img)
        embeddings = output.data
        reps.append(embeddings.cpu().numpy())
    return np.vstack(reps)


def calc_metrics(features, centroids, targ_class=None, labels=None, batch=True):
    """
    :param features: model generate
    :param centroids: K-means generate cluster
    :param batch_class: according K-means batch index generate
    :return: metrics
    """
    metrics_dict = {}
    if not isinstance(features, np.ndarray):
        features = features.data.cpu().numpy()
    if (not isinstance(labels, np.ndarray)) and labels is not None:
        labels = labels.numpy()
    features_arr = np.expand_dims(features, axis=1)
    pred_labels = np.argmin(np.linalg.norm((features_arr - centroids), axis=-1), axis=-1)
    pred_labels //= magnet_setting['k']
    if batch:
        assert targ_class is not None
        num_classes = len(np.unique(targ_class))
        gt_labels = np.repeat(targ_class, magnet_setting['d'])
    else:
        num_classes = len(np.unique(labels))
        assert labels is not None
        gt_labels = labels
    acc = np.sum(np.equal(pred_labels, gt_labels)) / len(pred_labels)
    for i in range(num_classes):
        recall = np.sum((pred_labels == i) * (gt_labels == i)) / (np.sum(gt_labels == i) + 1e-8)
        precison = np.sum((pred_labels == i) * (gt_labels == i)) / (np.sum(pred_labels == i) + 1e-8)
        metrics_dict["recall%d" % i] = recall
        metrics_dict["precision%d" % i] = precison
        if i == 1:
            fmeasure = (2 * recall * precison) / (recall + precison + 1e-8)
            metrics_dict["fmeasure%d" % i] = fmeasure
    metrics_dict['acc'] = acc

    return metrics_dict


def train(train_loader, batch_builder, model, criterion, optimizer, epoch, device):
    """
    1. initial get representations
    2. kmeans to get k cluster, each cluster has m cluster center
    3. each batch select d index from per cluster center
    """
    m = magnet_setting['m']
    d = magnet_setting['d']
    k = magnet_setting['k']
    n_steps = len(train_loader)
    model.eval()
    initial_reps = compute_reps(model, train_dataset)
    if initial_reps is None:
        raise NotImplementedError
    alpha = magnet_setting['alpha']
    batch_builder.update_clusters(initial_reps)
    batch_example_inds, batch_class_inds, batch_class = batch_builder.gen_batch()
    train_loader.sampler.batch_indices = batch_example_inds

    model.train()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    recall0 = AverageMeter()
    recall1 = AverageMeter()
    precision0 = AverageMeter()
    precision1 = AverageMeter()
    F1 = AverageMeter()
    acc = AverageMeter()

    start = time.time()
    for i in range(n_steps):
        # print(batch_class)
        niter = epoch * n_steps + i
        for _, (input, target) in enumerate(train_loader):
            """
            only once iteration because of the dataloader batch_sampler 
            """
            input = input.to(device)
            # target = target.to(device)
            data_time.update(time.time() - start)
            output, _ = model(input)
            loss, batch_example_losses = criterion(output, batch_class_inds, m, d, alpha, device)
            losses.update(loss.item(), input.size(0))
            metrics = calc_metrics(output, batch_builder.centroids, batch_class)
            recall0.update(metrics['recall0'], input.size(0))
            recall1.update(metrics['recall1'], input.size(0))

            precision0.update(metrics['precision0'], input.size(0))
            precision1.update(metrics['precision1'], input.size(0))

            F1.update(metrics['fmeasure1'], input.size(0))
            acc.update(metrics['acc'], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - start)
            start = time.time()

        batch_builder.update_losses(batch_example_inds, batch_example_losses)
        batch_example_inds, batch_class_inds, batch_class = batch_builder.gen_batch()
        train_loader.sampler.batch_indices = batch_example_inds

        if i % args.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}({loss.avg:.4f})  '
                  'R0 {recall0.val:.4f}({recall0.avg:.4f}) '
                  'R1 {recall1.val:.4f}({recall1.avg:.4f}) '
                  'P0 {precision0.val:.4f}({precision0.avg:.4f}) '
                  'P1 {precision1.val:.4f}({precision1.avg:.4f}) '
                  'F1 {F1.val:.4f} '
                  'Acc {acc.val:.4f}({acc.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, recall0=recall0, recall1=recall1,
                precision0=precision0, precision1=precision1, acc=acc, F1=F1))

    writer.add_scalar('Train/loss', losses.avg, niter)
    writer.add_scalar('Train/recall0', recall0.avg, niter)
    writer.add_scalar('Train/recall1', recall1.avg, niter)
    writer.add_scalar('Train/precision0', precision0.avg, niter)
    writer.add_scalar('Train/precision1', precision1.avg, niter)
    writer.add_scalar('Train/acc', acc.avg, niter)
    writer.add_scalar('Train/F1', F1.avg, niter)
    writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], niter)

    return batch_builder.centroids


def val(model, epoch, centroids):
    niter = (epoch + 1) * len(train_dataloader)
    model.eval()
    val_reps = compute_reps(model, val_dataset)
    if val_reps is None:
        raise NotImplementedError
    # val_labels = val_dataset.labels
    val_labels = getattr(val_dataset, 'test_labels')
    metrics = calc_metrics(val_reps, centroids, labels=val_labels, batch=False)
    print('Val R0 {0:.4f} '
          'R1 {1:.4f} '
          'P0 {2:.4f} '
          'P1 {3:.4f} '
          'F1 {4:.4f} '
          'Acc {5:.4f}'.format(metrics['recall0'], metrics['recall1'], metrics['precision0'],
                                         metrics['precision1'], metrics['fmeasure1'], metrics['acc']))

    writer.add_scalar('Val/recall0', metrics['recall0'], niter)
    writer.add_scalar('Val/recall1', metrics['recall1'], niter)
    writer.add_scalar('Val/precision0', metrics['precision0'], niter)
    writer.add_scalar('Val/precision1', metrics['precision1'], niter)
    writer.add_scalar('Val/F1', metrics['fmeasure1'], niter)
    writer.add_scalar('Val/acc', metrics['acc'], niter)
    # plot debug
    if args.feature_n == 2:
        plot_embedding(val_reps, val_labels, epoch)
    # schedule learning rate
    lr_schedule.step(metrics['fmeasure1'])
    return metrics['fmeasure1']


magnet_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


def plot_embedding(embeddings, targets, epoch):
    plt.figure(figsize=(10,10))
    cls_num = len(np.unique(targets))
    for i in range(cls_num):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    xlim = (np.min(embeddings[:, 0]), np.max(embeddings[:, 0]))
    ylim = (np.min(embeddings[:, 1]), np.max(embeddings[:, 1]))
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.legend(magnet_classes)
    plt.savefig(os.path.join(args.checkpoint_dir, detail_log, str(epoch) + '.png'))


def main():
    F1_best = 0
    m = magnet_setting['m']
    d = magnet_setting['d']
    k = magnet_setting['k']
    # train_labels = train_dataset.labels
    train_labels = getattr(train_dataset, 'train_labels')
    batch_builder = ClusterBatchBuilder(train_labels, k, m, d)

    for epoch in range(args.epochs):
        centroids = train(train_dataloader, batch_builder, model, criterion, optimizer, epoch, device)
        F1 = val(model, epoch, centroids)
        is_best = F1 > F1_best
        F1_best = max(F1, F1_best)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'acc_best': F1_best,
        }, is_best)


if __name__ == "__main__":
    main()

