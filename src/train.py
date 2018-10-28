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

os.chdir("/home/mengdi/yuxiang.ye/kaggle")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parse = argparse.ArgumentParser()
parse.add_argument("--epochs", type=int,  default=50, help="number of epoch")
parse.add_argument("--batchsize", type=int, default=8, help="size of each image batch")
parse.add_argument("--model_cfg_path", type=str, default='RSNA/config/yolov3.cfg', help="model config path")
parse.add_argument("--weights_path", type=str, default="RSNA/config/yolov3.weights", help="init weights file")
parse.add_argument("--n_cpu", type=int, default=16, help="number of cpu during batch generation")
parse.add_argument("--origin_dim", type=int, default=1024)
parse.add_argument("--input_dim", type=int, default=512)
parse.add_argument("--checkpoint_dir", type=str, default="RSNA/checkpoints_1013", help="directory where model weights save")
parse.add_argument("--tensorboardX_log", type=str, default="RSNA/Tensorboardx_1013", help="directory where model log save")
parse.add_argument("--checkpoint_interval", type=int, default=1)
parse.add_argument("--anchor_num", type=int, default=9)

"""
need you define!!
"""
detail_log = "1013_focalloss"

args = parse.parse_args()
os.makedirs(os.path.join(args.checkpoint_dir, detail_log), exist_ok=True)
os.makedirs(args.tensorboardX_log, exist_ok=True)
writer = SummaryWriter(os.path.join(args.tensorboardX_log, detail_log))

# model defination, and load model
model = Darknet(args.model_cfg_path, channel=3, droprate=0.4)
# model.load_state_dict(torch.load("/home/mengdi/yuxiang.ye/kaggle/RSNA/checkpoints/"
#                                  "adam_1004_solve_bug_finetune/22000_0.370154_best.pth"))
print(model.module_list)
model.load_weights(args.weights_path)
model.to(device)
print("Network successfully loaded")

net_info = model.blocks[0]
learning_rate = float(net_info['learning_rate'])
# finetune
learning_rate = 1.e-4

# momentum = float(net_info['momentum'])
# decay = float(net_info['decay'])
# print("`Net Info` lr:{0:} momentum:{1:} decay:{2:}".format(learning_rate, momentum, decay))

# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensorBase

# create dataloader
train_datset = yolov3_dataset(yolov3_config['info_path'], subset=[0, 1, 2, 4], agu=True)
val_datset = yolov3_dataset(yolov3_config['info_path'], subset=[3], agu=True)
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
try_time = 0
early_stop_time = 0


def save_checkpoint(state, is_best, filename='checkpoint_%d_%.5f.pth.tar'):
    epoch = state['epoch']
    acc = state['loss_best']
    checkpath = os.path.join(args.checkpoint_dir, detail_log, filename % (epoch, acc))
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

    for batch_t, (imgs, targets) in enumerate(train_dataloader):
        # ['high_reso', 'mid_reso', 'low_reso']
        imgs = imgs.to(device)
        optimizer.zero_grad()
        loss = model(imgs, device, targets)
        loss.backward()
        per_recall = model.losses['pred_num'] / (model.losses['gt_num'] / args.anchor_num)

        total_loss.update(loss.item(), imgs.size(0))
        loss_x.update(model.losses['x'], imgs.size(0))
        loss_y.update(model.losses['y'], imgs.size(0))
        loss_w.update(model.losses['w'], imgs.size(0))
        loss_h.update(model.losses['h'], imgs.size(0))
        loss_conf.update(model.losses['conf'], imgs.size(0))
        recall.update(per_recall, imgs.size(0))

        niter = epoch * len(train_dataloader) + batch_t

        print("[Train: Epoch {0:}/{1:} Batch {2:}/{3:}] [Losses: "
              "x {loss_x.val:.5f}({loss_x.avg:.5f}), "
              "y {loss_y.val:.5f}({loss_y.avg:.5f}), "
              "w {loss_w.val:.5f}({loss_w.avg:.5f}), "
              "h {loss_h.val:.5f}({loss_h.avg:.5f}), "
              "conf {loss_conf.val:.5f}({loss_conf.avg:.5f}), "
              "total_loss {total_loss.val:.5f}({total_loss.avg:.5f}), "
              "recall {recall.val:.5f}({recall.avg:.5f})]".format(epoch, args.epochs, batch_t, len(train_dataloader),
                                           loss_x=loss_x, loss_y=loss_y, loss_w=loss_w, loss_h=loss_h,
                                           loss_conf=loss_conf, total_loss=total_loss, recall=recall))

        if (niter % 100 == 0) and (niter != 0):
            writer.add_scalar('Train/total_loss', total_loss.avg, niter)
            writer.add_scalar('Train/loss_x', loss_x.avg, niter)
            writer.add_scalar('Train/loss_y', loss_y.avg, niter)
            writer.add_scalar('Train/loss_w', loss_w.avg, niter)
            writer.add_scalar('Train/loss_h', loss_h.avg, niter)
            writer.add_scalar('Train/loss_conf', loss_conf.avg, niter)
            writer.add_scalar('Train/recall', recall.avg, niter)
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], niter)

        optimizer.step()

        if (niter % 1000 == 0) and (niter != 0):
            model.eval()

            vtotal_loss = AverageMeter()
            vloss_x = AverageMeter()
            vloss_y = AverageMeter()
            vloss_w = AverageMeter()
            vloss_h = AverageMeter()
            vloss_conf = AverageMeter()
            vrecall = AverageMeter()

            with torch.no_grad():
                val_len = len(val_dataloader)
                for batch_v, (imgs, targets) in enumerate(val_dataloader):
                    imgs = imgs.to(device)
                    loss = model(imgs, device, targets)

                    vtotal_loss.update(loss.item(), imgs.size(0))
                    vloss_x.update(model.losses['x'], imgs.size(0))
                    vloss_y.update(model.losses['y'], imgs.size(0))
                    vloss_w.update(model.losses['w'], imgs.size(0))
                    vloss_h.update(model.losses['h'], imgs.size(0))
                    vloss_conf.update(model.losses['conf'], imgs.size(0))
                    vrecall.update(per_recall, imgs.size(0))

                    print("[Val: Epoch {0:}/{1:} Batch {2:}/{3:}] [Losses: "
                          "x {loss_x.val:.5f}({loss_x.avg:.5f}), "
                          "y {loss_y.val:.5f}({loss_y.avg:.5f}), "
                          "w {loss_w.val:.5f}({loss_w.avg:.5f}), "
                          "h {loss_h.val:.5f}({loss_h.avg:.5f}), "
                          "conf {loss_conf.val:.5f}({loss_conf.avg:.5f}), "
                          "total_loss {total_loss.val:.5f}({total_loss.avg:.5f}), "
                          "recall {recall.val:.5f}({recall.avg:.5f})]".format(epoch, args.epochs, batch_v,
                                                                              len(val_dataloader),
                                                                              loss_x=vloss_x, loss_y=vloss_y, loss_w=vloss_w,
                                                                              loss_h=vloss_h,
                                                                              loss_conf=vloss_conf, total_loss=vtotal_loss,
                                                                              recall=vrecall))

                writer.add_scalar('Val/total_loss', vtotal_loss.avg, niter)
                writer.add_scalar('Val/loss_x', vloss_x.avg, niter)
                writer.add_scalar('Val/loss_y', vloss_y.avg, niter)
                writer.add_scalar('Val/loss_w', vloss_w.avg, niter)
                writer.add_scalar('Val/loss_h', vloss_h.avg, niter)
                writer.add_scalar('Val/loss_conf', vloss_conf.avg, niter)
                writer.add_scalar('Val/recall', vrecall.avg, niter)
                writer.add_scalar('Val/lr', optimizer.param_groups[0]['lr'], niter)

                is_best = vtotal_loss.avg < loss_best
                loss_best = min(vtotal_loss.avg, loss_best)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss_best': loss_best,
                    'optimizer': optimizer.state_dict(),
                }, is_best)

                if is_best:
                    try_time = 0
                else:
                    try_time += 1
                    if try_time == 10:
                        agjust_learning_rate(optimizer)
                        try_time = 0
torch.cuda.empty_cache()





