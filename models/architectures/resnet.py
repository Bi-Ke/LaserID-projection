"""
    Function: ResNet architecture.

    Date: August 28, 2022.
"""
import sys
sys.path.insert(0, '.')
import torch.nn as nn
import torch
from torch.nn import functional as F
from models.post_processing.post_processing_module import PostProcessingModule


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class FinalModel(nn.Module):
    def __init__(self, backbone_net, semantic_head):
        super(FinalModel, self).__init__()
        self.backend = backbone_net
        self.semantic_head = semantic_head

    def forward(self, x):
        middle_feature_maps = self.backend(x)
        semantic_output = self.semantic_head(middle_feature_maps)
        return semantic_output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU()

        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, nclasses, aux, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True,
                 norm_layer=None, groups=1, width_per_group=64, search=7, ppm_input_channels=32):
        super(ResNet34, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        # Input Module.
        self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        # Backbone.
        self.inplanes = 128
        self.layer1 = self._make_layer(block, 128, layers[0])  # 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 4
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)  # 6
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)  # 3

        # Classification Head.
        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)

        # (Added) Point-wise classification head.
        self.ppm_conv = nn.Conv2d(128, ppm_input_channels, kernel_size=(3, 3), padding=(1, 1))
        self.ppm = PostProcessingModule(nclasses=nclasses,
                                        search=search,
                                        ppm_input_channels=ppm_input_channels)

        if self.aux:
            self.semantic_output = nn.Conv2d(128, nclasses, 1)
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                           norm_layer(planes * block.expansion))
            else:
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                            if_BN=self.if_BN))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x, proj_range_xyz, unproj_range_xyz, p2ri_lut, num_valid_pts):
        """p2ri_lut: Look Up Table of mapping points to the corresponding range image.
           num_points: number of valid points.
        """
        # Input Module.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Backbone.
        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        # Classification Head.
        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)  # B x 128 x H x W.

        # (Added) Point-wise classification head.
        # x: [B, C, H, W], input semantic features.
        # proj_range_xyz, [B, 4, H, W], range, x, y, z.
        # unproj_range_xyz: [B, N, 4], range, x, y, z.
        # p2ri_lut: [B, N, 3], point_index, y_coord, x_coord.
        # num_valid_pt: [B, 1], means the number of valid points.
        ppm_out = self.ppm_conv(out)  # B x 32 x H x W.
        main_head_out = self.ppm(x=ppm_out,
                                 proj_range_xyz=proj_range_xyz,
                                 unproj_range_xyz=unproj_range_xyz,
                                 p2ri_lut=p2ri_lut,
                                 num_valid_pts=num_valid_pts)

        # Auxiliary Branches.
        if self.aux:
            head1_out = self.semantic_output(out)
            head2_out = self.aux_head1(res_2)
            head3_out = self.aux_head2(res_3)
            head4_out = self.aux_head3(res_4)
            return [main_head_out, head1_out, head2_out, head3_out, head4_out]
        else:
            return main_head_out  # [B, cls, N]


if __name__ == "__main__":
    import time
    # model = ResNet34(nclasses=19, aux=True).cuda()
    model = ResNet34(nclasses=19, aux=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    for i in range(20):
        # inputs = torch.randn(2, 5, 64, 2048).cuda()
        inputs = torch.randn(2, 5, 64, 2048)

        proj_range_xyz = torch.rand(2, 4, 64, 2048)  # [B, 4, H, W]
        unproj_range_xyz = torch.rand(2, 10, 4)  # [B, N, 4]

        p2ri_lut = torch.zeros(2, 10, 3)  # [B, N, 3], point indices, y_coords, and x_coords.
        p2ri_lut[0][0, :] = torch.LongTensor([0, 0, 1])
        p2ri_lut[0][1, :] = torch.LongTensor([1, 0, 2])
        p2ri_lut[1][0, :] = torch.LongTensor([0, 0, 1])
        p2ri_lut[1][1, :] = torch.LongTensor([1, 0, 2])
        p2ri_lut[1][2, :] = torch.LongTensor([2, 1, 3])
        p2ri_lut = p2ri_lut.long()

        num_valid_pt = torch.LongTensor([[2], [3]])  # [B,]
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = model(inputs, proj_range_xyz, unproj_range_xyz, p2ri_lut, num_valid_pt)
            # torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            main_head_out, head1_out, head2_out, head3_out, head4_out = outputs
            print("main_head_out: ", main_head_out.shape)
            print("head1_out: ", head1_out.shape)
            print("head2_out: ", head2_out.shape)
            print("head3_out: ", head3_out.shape)
            print("head4_out: ", head4_out.shape)
            # main_head_out:  torch.Size([2, 19, 10])
            # head1_out:  torch.Size([2, 19, 64, 2048])
            # head2_out:  torch.Size([2, 19, 64, 2048])
            # head3_out:  torch.Size([2, 19, 64, 2048])
            # head4_out:  torch.Size([2, 19, 64, 2048])
        fwt = time.time() - start_time
        time_train.append(fwt)
        print("Forward time per img: %.3f (Mean: %.3f)" % (fwt / 1, sum(time_train) / len(time_train) / 1))
        time.sleep(0.15)




