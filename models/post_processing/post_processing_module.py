"""
    Function: Attention Post-Processing Module.

    Date: September 5, 2022.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class Mlp(nn.Module):
    """two MLP layers."""
    def __init__(self, in_features=4, hidden_features=64, out_features=32):
        """input features: (xi-xj, yi-yj, xi-xj, ri-rj)"""
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)  # 4 -> 64.
        self.norm = nn.LayerNorm(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)  # 64 -> 32.
        # self.drop = nn.Dropout(drop)  # drop = 0.1
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """B x N x C"""
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        return x


class PostProcessingModule(nn.Module):
    def __init__(self, nclasses=19, search=7, ppm_input_channels=32):
        super(PostProcessingModule, self).__init__()
        self.search = search
        self.nclasses = nclasses
        self.ppm_input_channels = ppm_input_channels

        self.mlp = Mlp(in_features=4, hidden_features=64, out_features=ppm_input_channels)
        self.conv = nn.Conv2d(ppm_input_channels, nclasses, kernel_size=(self.search, self.search))  # Linear or Conv2d?

    def forward(self, x, proj_range_xyz, unproj_range_xyz, p2ri_lut, num_valid_pts):
        """ Warning! Only works for un-batched point clouds.
            If they come batched we need to iterate over the batch dimension or do
            something REALLY smart to handle unaligned number of points in memory

            Additionally, the goal of all of the following non parallel operations is to avoid out of memory !!!

            x: [B, C, H, W], input semantic features.
            proj_range_xyz, [B, 4, H, W], range, x, y, z.
            unproj_range_xyz: [B, N, 4], range, x, y, z.
            p2ri_lut: [B, N, 3], point_index, y_coord, x_coord.
            num_valid_pt: [B, 1], means the number of valid points.
        """
        # check if size of kernel is odd and even.
        if self.search % 2 == 0:
            raise ValueError("Nearest neighbor kernel must be odd number")

        # calculate padding
        pad = int((self.search - 1) / 2)  # padding: 3.
        # center = int(((self.search * self.search) - 1) / 2)  # 7*7 / 2 = 24.

        # sizes of projection scan
        B, C, H, W = proj_range_xyz.shape  # [B, 4, H, W]

        x = F.pad(x, pad=(pad, pad, pad, pad), mode='constant', value=0.0)  # [B, C, H+2P, W+2P]
        proj_range_xyz = F.pad(proj_range_xyz, pad=(pad, pad, pad, pad), mode='constant', value=0.0)  # [B, 4, H+2P, W+2P]

        max_num_pts = unproj_range_xyz.shape[1]  # the maximum number of points.
        out = torch.zeros(B, max_num_pts, self.nclasses).type_as(x)  # [B, N, cls]

        for b in range(B):
            b_num_vpts = num_valid_pts[b]
            b_proj_range_xyz = proj_range_xyz[b]  # [4, H+2P, W+2P]
            # b_unproj_range_xyz = unproj_range_xyz[b][0:b_num_vpts, :]  # [V, 4]
            b_unproj_range_xyz = unproj_range_xyz[b]  # [N, 4]
            b_p2ri_lut = p2ri_lut[b][0:b_num_vpts, :]  # [V, 3]
            b_x = x[b]  # [C, H+2P, W+2P]

            # WARNING: only for Velodyne HDL-64E
            for idx in range(64):
                top = idx
                bottom = idx + self.search
                idx_b_proj_range_xyz = b_proj_range_xyz[:, top:bottom, :]  # [4, 7, W+2P]

                # unfold neighborhood to get nearest neighbors for each pixel.
                idx_b_unfold_proj_range_xyz = F.unfold(idx_b_proj_range_xyz[None, ...],  # [1, 4, 7, W+2P]
                                                       kernel_size=(self.search, self.search))  # [1, 4*7*7, W]

                # index with px, py to get ALL the points in the point cloud
                idx_mask = b_p2ri_lut[:, 1] == idx
                idx_b_p2ri_lut = b_p2ri_lut[idx_mask, :]

                # idx_b_y_coords = idx_b_p2ri_lut[:, 1]  # idx = 0,1,2,...,64.
                idx_b_x_coords = idx_b_p2ri_lut[:, 2]
                # idx_b_idx_list = idx_b_y_coords * W + idx_b_x_coords
                idx_b_unfold_unproj_range_xyz = idx_b_unfold_proj_range_xyz[:, :, idx_b_x_coords]  # [1, 4*7*7, idxV]

                # WARNING, THIS IS A HACK
                # Make non valid (<0) range points extremely big so that there is no screwing
                # up the nn self.search
                # unproj_unfold_k_rang[unproj_unfold_k_rang == 0] = float("inf")  # revised.
                idx_b_idx_pt = idx_b_p2ri_lut[:, 0]  # point indices.
                idx_b_unproj_range_xyz = b_unproj_range_xyz[idx_b_idx_pt, :]  # [N, 4] -> [idxV, 4]
                idx_b_unproj_range_xyz = idx_b_unproj_range_xyz.t().contiguous()  # [4, idxV]

                # now the matrix is unfolded TOTALLY, replace the middle points with the actual points
                idx_b_unfold_unproj_range_xyz = idx_b_unfold_unproj_range_xyz.view(1, C, self.search*self.search, -1)
                # idx_b_unfold_unproj_range_xyz[:, :, center, :] = idx_b_unproj_range_xyz  # [1, 4, 7*7, idxV] ???

                # 1. relative coordinates.
                # compute (|ri-rj|, |xi-xj|, |yi-yj|, |zi-zj|)
                idx_b_unproj_range_xyz = idx_b_unproj_range_xyz.unsqueeze(dim=1)  # [4, 1, idxV]
                idx_b_range_xyz_distances = torch.abs(idx_b_unfold_unproj_range_xyz - idx_b_unproj_range_xyz) # [1, 4, 7*7, idxV]
                idx_b_range_xyz_distances = idx_b_range_xyz_distances.permute(0, 3, 2, 1).contiguous()  # [1, 4, 7*7, idxV] -> [1, idxV, 7*7, 4]

                # make weights.
                idx_b_range_xyz_distances = self.mlp(idx_b_range_xyz_distances)  # [1, idxV, 7*7, 32]
                idx_b_rxyz_weights = F.softmax(idx_b_range_xyz_distances, dim=-1)  # [1, idxV, 7*7, 32]

                # 2. do the same unfolding with the 'x'.
                idx_b_x = b_x[:, top:bottom, :]  # [C, 7, W+2P]
                idx_b_unfold_x = F.unfold(idx_b_x[None, ...],  # [1, C, 7, W+2P]
                                          kernel_size=(self.search, self.search))  # [1, C*7*7, W]
                idx_b_unfold_x = idx_b_unfold_x[:, :, idx_b_x_coords]  # [1, C*7*7, idxV]
                idx_b_unfold_x = idx_b_unfold_x.view(1, self.ppm_input_channels, self.search*self.search, -1)  # [1, C, 7*7, idxV]
                idx_b_unfold_x = idx_b_unfold_x.permute(0, 3, 2, 1).contiguous()  # [1, idxV, 7*7, C], C = 32.

                # 3. element-wise product.
                idx_b_unfold_x = idx_b_unfold_x * idx_b_rxyz_weights  # [1, idxV, 7*7, 32]
                idx_b_unfold_x = idx_b_unfold_x.view(-1, self.search, self.search, self.ppm_input_channels)  # [idxV, 7, 7, 32]
                idx_b_unfold_x = idx_b_unfold_x.permute(0, 3, 1, 2).contiguous()  # [idxV, 32, 7, 7]

                # 4. get the final probability.
                idx_b_unfold_x = self.conv(idx_b_unfold_x).view(-1, self.nclasses)  # [idxV, 19]

                # 5. save the final probability. # [B, N, 19] -> [N, 19]
                out[b][idx_b_idx_pt, :] = idx_b_unfold_x
        out = out.permute(0, 2, 1).contiguous()  # [B, N, cls] -> [B, cls, N]
        return out


# if __name__ == "__main__":
#     # x, proj_range_xyz, unproj_range_xyz, p2ri_lut, num_valid_pts
#     x = torch.rand(2, 32, 64, 20)  # [B, C, H, W]
#     proj_range_xyz = torch.rand(2, 4, 64, 20)  # [B, 4, H, W]
#     unproj_range_xyz = torch.rand(2, 10, 4)  # [B, N, 4]
#
#     p2ri_lut = torch.zeros(2, 10, 3)  # [B, N, 3], point indices, y_coords, and x_coords.
#     p2ri_lut[0][0, :] = torch.LongTensor([0, 0, 1])
#     p2ri_lut[0][1, :] = torch.LongTensor([1, 0, 2])
#     p2ri_lut[1][0, :] = torch.LongTensor([0, 0, 1])
#     p2ri_lut[1][1, :] = torch.LongTensor([1, 0, 2])
#     p2ri_lut[1][2, :] = torch.LongTensor([2, 1, 3])
#     p2ri_lut = p2ri_lut.long()
#
#     num_valid_pt = torch.LongTensor([[2], [3]])  # [B,]
#
#     ppm = PostProcessingModule(nclasses=19, search=7, ppm_input_channels=32)
#     out = ppm(x=x,
#               proj_range_xyz=proj_range_xyz,
#               unproj_range_xyz=unproj_range_xyz,
#               p2ri_lut=p2ri_lut,
#               num_valid_pts=num_valid_pt)
#     print(out.shape)

