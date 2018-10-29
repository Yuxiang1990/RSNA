from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from RSNA.model.focalloss import FocalLoss
import torch.nn.functional as F


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, inp_dim, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.inp_dim = inp_dim
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.sml_loss = nn.SmoothL1Loss()
        self.focalLoss = FocalLoss(gamma=2)

    def forward(self, x, device, anchors_index, target=None, iou_thre=0.4):
        is_training = target is not None
        if is_training:
            if anchors_index == 0:
                target = target['low_reso']
            elif anchors_index == 1:
                target = target['mid_reso']
            elif anchors_index == 2:
                target = target['high_reso']

        batch_size = x.size(0)
        stride = self.inp_dim // x.size(2)
        grid_size = self.inp_dim // stride
        bbox_attrs = 5
        num_anchors = len(self.anchors)
        # print(stride, grid_size, x.size(2), self.inp_dim)

        x = x.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
        anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors]
        # print(anchors)
        # Sigmoid the  centre_X, centre_Y. and object confidencce
        x[:, :, 0] = torch.sigmoid(x[:, :, 0])
        x[:, :, 1] = torch.sigmoid(x[:, :, 1])
        x[:, :, 4] = torch.sigmoid(x[:, :, 4])
        if is_training:
            # calc loss
            # (batchsize, anchor_num, feature_size, feature_size, bbox_attrs)
            target = target.permute(0, 2, 3, 1, 4).contiguous()
            target = target.view(-1, grid_size * grid_size * num_anchors, 6)
            self.mse_loss = self.mse_loss.to(device)
            self.bce_loss = self.bce_loss.to(device)
            self.sml_loss = self.sml_loss.to(device)
            self.focalLoss = self.focalLoss.to(device)
            target = target.to(device)

            # select anchor center index
            target_squeeze = target.view(-1, 6)
            target_index = torch.nonzero(target_squeeze[..., 5]).squeeze()
            x_squeeze = x.view(-1, bbox_attrs)
            x_squeeze = x_squeeze.index_select(0, target_index)
            target_squeeze = target_squeeze.index_select(0, target_index)

            # calc
            if target_squeeze.size(0):
                n_correct = (x_squeeze[..., 4] > iou_thre).sum().item()
                n_gt = target_squeeze[..., 5].sum().item()
            else:
                n_gt = 0
                n_correct = 0
            # print("n_correct:", n_correct, "n_gt", n_gt)

            if target_squeeze.size(0):
                loss_x = self.sml_loss(x_squeeze[..., 0], target_squeeze[..., 0])
                loss_y = self.sml_loss(x_squeeze[..., 1], target_squeeze[..., 1])
                loss_w = self.sml_loss(x_squeeze[..., 2], target_squeeze[..., 2]) / 2
                loss_h = self.sml_loss(x_squeeze[..., 3], target_squeeze[..., 3]) / 2
            else:
                loss_x = torch.from_numpy(np.array(0)).type(torch.FloatTensor).to(device)
                loss_y = torch.from_numpy(np.array(0)).type(torch.FloatTensor).to(device)
                loss_w = torch.from_numpy(np.array(0)).type(torch.FloatTensor).to(device)
                loss_h = torch.from_numpy(np.array(0)).type(torch.FloatTensor).to(device)

            # loss_x = self.sml_loss(x[..., 0] * target[..., 5], target[..., 0] * target[..., 5])
            # loss_y = self.sml_loss(x[..., 1] * target[..., 5], target[..., 1] * target[..., 5])
            # loss_w = self.sml_loss(x[..., 2] * target[..., 5], target[..., 2] * target[..., 5]) / 2
            # loss_h = self.sml_loss(x[..., 3] * target[..., 5], target[..., 3] * target[..., 5]) / 2

            loss_conf = self.bce_loss(x[..., 4], (target[..., 4] > iou_thre).type(torch.FloatTensor).to(device))
            # loss_conf = self.bce_loss(x[..., 4], target[..., 4]).type(torch.FloatTensor).to(device)
            loss = 1.0 * (loss_x + loss_y + loss_w + loss_h) + loss_conf
            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), n_gt, n_correct
        else:
            heatmap = x.view(batch_size, grid_size, grid_size, num_anchors, bbox_attrs)
            # Add the center offsets
            grid = np.arange(grid_size)
            a, b = np.meshgrid(grid, grid)

            x_offset = torch.FloatTensor(a).view(-1, 1)
            y_offset = torch.FloatTensor(b).view(-1, 1)

            x_offset = x_offset.to(device)
            y_offset = y_offset.to(device)

            x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
            x[:, :, :2] += x_y_offset

            # log space transform height and the width
            anchors = torch.FloatTensor(anchors).to(device)
            anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
            x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * anchors
            x[:, :, :4] *= stride

            return heatmap, x


def create_modules(blocks, channel=1, droprate=0.2):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    input_dim = net_info['height']
    module_list = nn.ModuleList()
    prev_filters = channel
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)
            if droprate is not None:
                module.add_module("dp_{0}".format(index), nn.Dropout2d(p=droprate))
            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

                # If it's an upsampling layer
                # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(int(input_dim), anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile, channel=1, clf=False, droprate=0.2):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks, channel=channel, droprate=droprate)
        self.loss_name = ['x', 'y', 'w', 'h', 'conf', 'gt_num', 'pred_num']
        self.yolo_detect_num = 0
        self.clf_support = clf
        self.droprate = droprate

    def forward(self, x, device, target=None):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer
        loss_sum_list = []
        is_training = target is not None
        self.losses = defaultdict(float)
        write = 0
        heatmap_list = []
        for i, module in enumerate(modules):
            # get clf_feature
            if (self.yolo_detect_num == 0) and self.clf_support and (x.size(1) == 1024):
                clf_feature = x

            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                if target is None:
                    # inference
                    # Transform
                    heatmap, x = self.module_list[i][0](x, device, self.yolo_detect_num, target=None)
                    # if not write:  # if no collector has been intialised.
                    #     detections = x
                    #     write = 1
                    # else:
                    #     detections = torch.cat((detections, x), 1)

                    # only high resolution only be used
                    if self.yolo_detect_num == 2:
                        detections = x
                        heatmap_list.append(heatmap)

                else:
                    # train
                    x, *losses = self.module_list[i][0](x, device, self.yolo_detect_num, target=target)

                    # only high resolution only be used
                    if self.yolo_detect_num == 2:
                        loss_sum_list.append(x)
                        for name, loss in zip(self.loss_name, losses):
                            self.losses[name] += loss

                self.yolo_detect_num += 1
                if self.yolo_detect_num == 3:
                    self.yolo_detect_num = 0
            outputs[i] = x

        if self.clf_support:
            return clf_feature, (sum(loss_sum_list) if is_training else (heatmap_list, detections))
        else:
            return sum(loss_sum_list) if is_training else (heatmap_list, detections)

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    if self.droprate is not None:
                        bn = model[2]

                    else:
                        bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


if __name__ == "__main__":
    darknet = Darknet("/home/mengdi/yuxiang.ye/YOLO_v3_tutorial_from_scratch/cfg/yolov3.cfg")
    darknet = darknet.cuda()
    pred = darknet(img.cuda(), torch.cuda.is_available())
    print(pred)