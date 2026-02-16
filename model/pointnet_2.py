# model/pointnet_2.py

# Model architecture code

import torch
import torch.nn as nn


def square_distance(src, dst):
    """Pairwise squared Euclidean distance."""
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    """Gather points/features by index."""
    device = points.device
    B = points.shape[0]
    
    view_shape = list(idx.shape)
    view_shape.append(points.shape[-1])
    
    idx = idx.reshape(B, -1)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1)
    batch_indices = batch_indices.repeat(1, idx.shape[1])
    
    new_points = points[batch_indices, idx, :]
    new_points = new_points.view(*view_shape)
    
    return new_points


def farthest_point_sample(xyz, npoint):
    """FPS sampling."""
    device = xyz.device
    B, N, _ = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), device=device)
    
    batch_indices = torch.arange(B, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query grouping."""
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.argsort(dim=-1)[:, :, :nsample]
    
    group_dist = torch.gather(dist, 2, group_idx)
    mask = group_dist > radius ** 2
    
    group_idx[mask] = group_idx[:, :, 0:1].expand_as(group_idx)[mask]
    return group_idx


class PointNetSetAbstractionMSG(nn.Module):
    def __init__(self, npoint, radii, nsamples, in_channel, mlp_channels_list):
        super().__init__()
        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        
        self.mlp_convs = nn.ModuleList()
        
        for scale in range(len(radii)):
            layers = []
            last_channel = in_channel + 3
            
            for out_channel in mlp_channels_list[scale]:
                layers.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
                layers.append(nn.BatchNorm2d(out_channel))
                layers.append(nn.ReLU(inplace=True))
                last_channel = out_channel
            
            self.mlp_convs.append(nn.Sequential(*layers))
    
    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        S = self.npoint
        
        if S is not None and S < N:
            fps_idx = farthest_point_sample(xyz, S)
            new_xyz = index_points(xyz, fps_idx)
        else:
            new_xyz = xyz
        
        new_points_list = []
        
        for i, radius in enumerate(self.radii):
            nsample = self.nsamples[i]
            
            group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
            
            if points is not None:
                grouped_points = index_points(points.transpose(1, 2), group_idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
            
            grouped_points = grouped_points.permute(0, 3, 1, 2)
            new_points = self.mlp_convs[i](grouped_points)
            new_points = torch.max(new_points, dim=-1)[0]
            
            new_points_list.append(new_points)
        
        new_points_concat = torch.cat(new_points_list, dim=1)
        
        return new_xyz, new_points_concat


class PointNetPPEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.sa1 = PointNetSetAbstractionMSG(
            npoint=1024,
            radii=[0.1, 0.2],
            nsamples=[16, 32],
            in_channel=0,
            mlp_channels_list=[[32, 32, 64], [64, 64, 128]]
        )
        
        self.sa2 = PointNetSetAbstractionMSG(
            npoint=256,
            radii=[0.2, 0.4],
            nsamples=[32, 64],
            in_channel=64 + 128,
            mlp_channels_list=[[64, 64, 128], [128, 128, 256]]
        )
        
        self.sa3 = PointNetSetAbstractionMSG(
            npoint=64,
            radii=[0.4, 0.8],
            nsamples=[64, 128],
            in_channel=128 + 256,
            mlp_channels_list=[[128, 128, 256], [256, 256, 512]]
        )
        
        self.global_mlp = nn.Sequential(
            nn.Conv1d(256 + 512, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, latent_dim, 1)
        )
    
    def forward(self, xyz):
        points = None
        
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        x = self.global_mlp(l3_points)
        x = torch.max(x, 2, keepdim=False)[0]
        
        return x


class PointNetPPDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_points=2048):
        super().__init__()
        self.num_points = num_points
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )
    
    def forward(self, latent):
        x = self.mlp(latent)
        x = x.view(-1, self.num_points, 3)
        return x


class PointNetPPAutoEncoder(nn.Module):
    def __init__(self, latent_dim=128, num_points=2048):
        super().__init__()
        
        self.encoder = PointNetPPEncoder(latent_dim=latent_dim)
        self.decoder = PointNetPPDecoder(latent_dim=latent_dim, num_points=num_points)
    
    def forward(self, xyz):
        latent = self.encoder(xyz)
        recon = self.decoder(latent)
        return recon, latent