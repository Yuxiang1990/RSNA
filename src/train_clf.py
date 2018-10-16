from __future__ import division
import torch
import shutil
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
from RSNA.model.darknet import Darknet
from RSNA.lib.dataloader_c3 import yolov3_dataset, yolov3_config, yolov3_batchsample
from tensorboardX import SummaryWriter
from RSNA.lib.utils import AverageMeter
import torch.nn as nn
from RSNA.model.focalloss import FocalLoss_clf

os.chdir("/home/mengdi/yuxiang.ye/kaggle")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parse = argparse.ArgumentParser()
parse.add_argument("--epochs", type=int,  default=100, help="number of epoch")
parse.add_argument("--batchsize", type=int, default=8, help="size of each image batch")
parse.add_argument("--model_cfg_path", type=str, default='RSNA/config/yolov3.cfg', help="model config path")
parse.add_argument("--weights_path", type=str, default="RSNA/config/yolov3.weights", help="init weights file")
parse.add_argument("--n_cpu", type=int, default=16, help="number of cpu during batch generation")
parse.add_argument("--origin_dim", type=int, default=1024)
parse.add_argument("--input_dim", type=int, default=512)
parse.add_argument("--checkpoint_dir", type=str, default="RSNA/checkpoints_1015", help="directory where model weights save")
parse.add_argument("--tensorboardX_log", type=str, default="RSNA/Tensorboardx_1015", help="directory where model log save")
parse.add_argument("--checkpoint_interval", type=int, default=1)
parse.add_argument("--anchor_num", type=int, default=9)

"""
need you define!!
"""
detail_log = "1016_dp0.1_gamma2_alpha212_loss1_2"

args = parse.parse_args()
os.makedirs(os.path.join(args.checkpoint_dir, detail_log), exist_ok=True)
os.makedirs(args.tensorboardX_log, exist_ok=True)
writer = SummaryWriter(os.path.join(args.tensorboardX_log, detail_log))

# model defination, and load model
darknet = Darknet(args.model_cfg_path, channel=3, clf=True, droprate=0.1)
print(darknet.module_list)
darknet.load_weights(args.weights_path)


def calc_metrics(output, target, class_num=3):
    metrics_dict = {}
    pred = torch.max(torch.nn.Softmax(dim=1)(output), dim=1)[1]
    acc = torch.sum(torch.eq(pred, target)).item() / target.size(0)
    for i in range(class_num):
        recall = torch.sum((pred == i) * (target == i)).item() / (torch.sum(target == i).item() + 1e-8)
        precison = torch.sum((pred == i) * (target == i)).item() / (torch.sum(pred == i).item() + 1e-8)
        metrics_dict["recall%d" % i] = recall
        metrics_dict["precision%d" % i] = precison
        if i == 0:
            fmeasure = (2 * recall * precison) / (recall + precison + 1e-8)
            metrics_dict["fmeasure%d" % i] = fmeasure
    metrics_dict['acc'] = acc
    return metrics_dict


class my_model(nn.Module):
    def __init__(self, darknet, filter_num=1024, num_class=3):
        super(my_model, self).__init__()
        self.darknet = darknet
        # self.loss = nn.CrossEntropyLoss()
        self.loss = FocalLoss_clf(gamma=2, alpha=[1.0, 0.5, 1.0])
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
        clf_target = clf_target.to(device)
        x, dark_loss = self.darknet(x, device, target)
        b_num = x.size(0)
        x = self.clf_net(x)
        x = x.view(b_num, -1)
        x = self.Linear(x)
        loss_clf = self.loss(x, clf_target)
        metrics_dict = calc_metrics(x, clf_target)
        return metrics_dict, loss_clf, dark_loss


model = my_model(darknet.to(device))
best_weights = "/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints_1015/" \
               "1015_focalloss_clf_focal_0.1_bce/model_best.pth.tar"
model.load_state_dict(torch.load(best_weights, map_location='cpu')['state_dict'])
model.to(device)
print("Network successfully loaded")

# net_info = model.blocks[0]
# learning_rate = float(net_info['learning_rate'])
# finetune
learning_rate = 3.e-5

# momentum = float(net_info['momentum'])
# decay = float(net_info['decay'])
# print("`Net Info` lr:{0:} momentum:{1:} decay:{2:}".format(learning_rate, momentum, decay))

# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensorBase

# create dataloader
train_datset = yolov3_dataset(yolov3_config['info_path'], subset=[0, 1, 2, 3], agu=True, clf=True)
val_datset = yolov3_dataset(yolov3_config['info_path'], subset=[4], agu=True, clf=True)
train_sample = yolov3_batchsample(train_datset.class_list)
train_dataloader = DataLoader(train_datset, batch_sampler=train_sample, num_workers=args.n_cpu)

val_sample = yolov3_batchsample(val_datset.class_list)
val_dataloader = DataLoader(val_datset, batch_sampler=val_sample, num_workers=args.n_cpu)


def agjust_learning_rate(optimizer, decay=0.3):
    lr = optimizer.param_groups[0]['lr'] * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("modify learning rate !!!, now is", lr)


loss_best = 1000.0
vfmeasure_best = 0
try_time = 0
early_stop_time = 0


def save_checkpoint(state, is_best, filename='checkpoint_{0:}_{1:.5f}_{2:}.pth.tar'):
    epoch = state['epoch']
    checkpath = os.path.join(args.checkpoint_dir, detail_log, filename.format(epoch, state['loss'], state['fmeasure']))
    bestpath = os.path.join(args.checkpoint_dir, detail_log, "model_best.pth.tar")
    if is_best:
        torch.save(state, checkpath)
        shutil.copyfile(checkpath, bestpath)


for epoch in range(args.epochs):
    model.train()

    # init record
    total_loss = AverageMeter()
    loss_x = AverageMeter()
    loss_y = AverageMeter()
    loss_w = AverageMeter()
    loss_h = AverageMeter()
    loss_conf = AverageMeter()
    recall = AverageMeter()

    clf = AverageMeter()
    recall0 = AverageMeter()
    recall1 = AverageMeter()
    recall2 = AverageMeter()
    precision0 = AverageMeter()
    precision1 = AverageMeter()
    precision2 = AverageMeter()
    acc = AverageMeter()
    fmeasure0 = AverageMeter()

    for batch_t, (imgs, targets, clf_targets) in enumerate(train_dataloader):
        # ['high_reso', 'mid_reso', 'low_reso']
        imgs = imgs.to(device)
        optimizer.zero_grad()
        metrics, loss_clf, loss_dark = model(imgs, device, targets, clf_targets)
        loss = loss_clf + 0.5 * loss_dark
        loss.backward()
        per_recall = model.darknet.losses['pred_num'] / (model.darknet.losses['gt_num'] / args.anchor_num)

        total_loss.update(loss.item(), imgs.size(0))
        clf.update(loss_clf.item(), imgs.size(0))
        loss_x.update(model.darknet.losses['x'], imgs.size(0))
        loss_y.update(model.darknet.losses['y'], imgs.size(0))
        loss_w.update(model.darknet.losses['w'], imgs.size(0))
        loss_h.update(model.darknet.losses['h'], imgs.size(0))
        loss_conf.update(model.darknet.losses['conf'], imgs.size(0))
        recall.update(per_recall, imgs.size(0))

        recall0.update(metrics['recall0'], imgs.size(0))
        recall1.update(metrics['recall1'], imgs.size(0))
        recall2.update(metrics['recall2'], imgs.size(0))
        precision0.update(metrics['precision0'], imgs.size(0))
        precision1.update(metrics['precision1'], imgs.size(0))
        precision2.update(metrics['precision2'], imgs.size(0))
        acc.update(metrics['acc'], imgs.size(0))
        fmeasure0.update(metrics['fmeasure0'], imgs.size(0))

        niter = epoch * len(train_dataloader) + batch_t

        print("[Train: Epoch {0:}/{1:} Batch {2:}/{3:}] [Losses: "
              "x {loss_x.val:.5f}({loss_x.avg:.5f}), "
              "y {loss_y.val:.5f}({loss_y.avg:.5f}), "
              "w {loss_w.val:.5f}({loss_w.avg:.5f}), "
              "h {loss_h.val:.5f}({loss_h.avg:.5f}), "
              "conf {loss_conf.val:.5f}({loss_conf.avg:.5f}), "
              "total_loss {total_loss.val:.5f}({total_loss.avg:.5f}), "
              "recall {recall.val:.5f}({recall.avg:.5f}), "
              "clf {clf.val:.5f}({clf.avg:.5f})"
              "R0 {recall0.val:.4f}({recall0.avg:.4f}) "
              "R1 {recall1.val:.4f}({recall1.avg:.4f}) "
              "R2 {recall2.val:.4f}({recall2.avg:.4f}) "
              "P0 {precision0.val:.4f}({precision0.avg:.4f}) "
              "P1 {precision1.val:.4f}({precision1.avg:.4f}) "
              "P2 {precision2.val:.4f}({precision2.avg:.4f}) "
              "Acc {acc.val:.4f}({acc.avg:.4f}]".format(epoch, args.epochs, batch_t, len(train_dataloader),
                                           loss_x=loss_x, loss_y=loss_y, loss_w=loss_w, loss_h=loss_h,
                                           loss_conf=loss_conf, total_loss=total_loss, recall=recall, clf=clf,
                                                            recall0=recall0, recall1=recall1,
                                                            recall2=recall2, precision0=precision0,
                                                            precision1=precision1,
                                                            precision2=precision2, acc=acc))
        optimizer.step()

        # if (niter % 100 == 0) and (niter != 0):
    writer.add_scalar('Train/total_loss', total_loss.avg, niter)
    writer.add_scalar('Train/loss_x', loss_x.avg, niter)
    writer.add_scalar('Train/loss_y', loss_y.avg, niter)
    writer.add_scalar('Train/loss_w', loss_w.avg, niter)
    writer.add_scalar('Train/loss_h', loss_h.avg, niter)
    writer.add_scalar('Train/loss_conf', loss_conf.avg, niter)
    writer.add_scalar('Train/recall', recall.avg, niter)
    writer.add_scalar('Train/clf', clf.avg, niter)
    writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], niter)
    writer.add_scalar('Train/recall0', recall0.avg, niter)
    writer.add_scalar('Train/recall1', recall1.avg, niter)
    writer.add_scalar('Train/recall2', recall2.avg, niter)
    writer.add_scalar('Train/precision0', precision0.avg, niter)
    writer.add_scalar('Train/precision1', precision1.avg, niter)
    writer.add_scalar('Train/precision2', precision2.avg, niter)
    writer.add_scalar('Train/acc', acc.avg, niter)
    writer.add_scalar('Train/fmeasure0', fmeasure0.avg, niter)



        # if (niter % 1000 == 0) and (niter != 0):
    model.eval()
    vtotal_loss = AverageMeter()
    vloss_x = AverageMeter()
    vloss_y = AverageMeter()
    vloss_w = AverageMeter()
    vloss_h = AverageMeter()
    vloss_conf = AverageMeter()
    vrecall = AverageMeter()
    vclf = AverageMeter()
    vrecall0 = AverageMeter()
    vrecall1 = AverageMeter()
    vrecall2 = AverageMeter()
    vprecision0 = AverageMeter()
    vprecision1 = AverageMeter()
    vprecision2 = AverageMeter()
    vacc = AverageMeter()
    vfmeasure0 = AverageMeter()

    with torch.no_grad():
        val_len = len(val_dataloader)
        for batch_v, (imgs, targets, clf_targets) in enumerate(val_dataloader):
            imgs = imgs.to(device)
            metrics, loss_clf, loss_dark = model(imgs, device, targets, clf_targets)
            loss = loss_clf + loss_dark
            vtotal_loss.update(loss.item(), imgs.size(0))
            vloss_x.update(model.darknet.losses['x'], imgs.size(0))
            vloss_y.update(model.darknet.losses['y'], imgs.size(0))
            vloss_w.update(model.darknet.losses['w'], imgs.size(0))
            vloss_h.update(model.darknet.losses['h'], imgs.size(0))
            vloss_conf.update(model.darknet.losses['conf'], imgs.size(0))
            vrecall.update(per_recall, imgs.size(0))

            vclf.update(loss_clf.item(), imgs.size(0))
            vrecall0.update(metrics['recall0'], imgs.size(0))
            vrecall1.update(metrics['recall1'], imgs.size(0))
            vrecall2.update(metrics['recall2'], imgs.size(0))

            vprecision0.update(metrics['precision0'], imgs.size(0))
            vprecision1.update(metrics['precision1'], imgs.size(0))
            vprecision2.update(metrics['precision2'], imgs.size(0))
            vacc.update(metrics['acc'], imgs.size(0))
            vfmeasure0.update(metrics['fmeasure0'], imgs.size(0))

            print("[Val: Epoch {0:}/{1:} Batch {2:}/{3:}] [Losses: "
                  "x {loss_x.val:.5f}({loss_x.avg:.5f}), "
                  "y {loss_y.val:.5f}({loss_y.avg:.5f}), "
                  "w {loss_w.val:.5f}({loss_w.avg:.5f}), "
                  "h {loss_h.val:.5f}({loss_h.avg:.5f}), "
                  "conf {loss_conf.val:.5f}({loss_conf.avg:.5f}), "
                  "total_loss {total_loss.val:.5f}({total_loss.avg:.5f}), "
                  "recall {recall.val:.5f}({recall.avg:.5f}), "
                  "clf {clf.val:.5f}({clf.avg:.5f}) "
                  "R0 {recall0.val:.4f}({recall0.avg:.4f}) "
                  "R1 {recall1.val:.4f}({recall1.avg:.4f}) "
                  "R2 {recall2.val:.4f}({recall2.avg:.4f}) "
                  "P0 {precision0.val:.4f}({precision0.avg:.4f}) "
                  "P1 {precision1.val:.4f}({precision1.avg:.4f}) "
                  "P2 {precision2.val:.4f}({precision2.avg:.4f}) "
                  "Acc {acc.val:.4f}({acc.avg:.4f}]]".format(epoch, args.epochs, batch_v,
                                                                      len(val_dataloader),
                                                                      loss_x=vloss_x, loss_y=vloss_y, loss_w=vloss_w,
                                                                      loss_h=vloss_h,
                                                                      loss_conf=vloss_conf, total_loss=vtotal_loss,
                                                                      recall=vrecall, clf=vclf, recall0=vrecall0,
                                                             recall1=vrecall1, recall2=vrecall2,
                                                             precision0=vprecision0, precision1=vprecision1,
                                                             precision2=vprecision2, acc=vacc))

        writer.add_scalar('Val/total_loss', vtotal_loss.avg, niter)
        writer.add_scalar('Val/clf_loss', vclf.avg, niter)
        writer.add_scalar('Val/loss_x', vloss_x.avg, niter)
        writer.add_scalar('Val/loss_y', vloss_y.avg, niter)
        writer.add_scalar('Val/loss_w', vloss_w.avg, niter)
        writer.add_scalar('Val/loss_h', vloss_h.avg, niter)
        writer.add_scalar('Val/loss_conf', vloss_conf.avg, niter)
        writer.add_scalar('Val/recall', vrecall.avg, niter)
        writer.add_scalar('Val/lr', optimizer.param_groups[0]['lr'], niter)
        writer.add_scalar('Val/recall0', vrecall0.avg, niter)
        writer.add_scalar('Val/recall1', vrecall1.avg, niter)
        writer.add_scalar('Val/recall2', vrecall2.avg, niter)
        writer.add_scalar('Val/precision0', vprecision0.avg, niter)
        writer.add_scalar('Val/precision1', vprecision1.avg, niter)
        writer.add_scalar('Val/precision2', vprecision2.avg, niter)
        writer.add_scalar('Val/acc', vacc.avg, niter)
        writer.add_scalar('Val/fmeasure0', vfmeasure0.avg, niter)

        is_loss_best = vtotal_loss.avg < loss_best
        loss_best = min(vtotal_loss.avg, loss_best)

        is_f1_best = vfmeasure0.avg > vfmeasure_best
        vfmeasure_best = max(vfmeasure0.avg, vfmeasure_best)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': loss_best,
            'fmeasure': vfmeasure_best
            # 'optimizer': optimizer.state_dict(),
        }, is_loss_best | is_f1_best)

        if is_loss_best | is_f1_best:
            try_time = 0
        else:
            try_time += 1
            if try_time == 10:
                agjust_learning_rate(optimizer)
                try_time = 0
torch.cuda.empty_cache()





