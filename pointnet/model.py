import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        x = pointcloud.transpose(2, 1)  # [B, 3, N]

        if self.input_transform:
            trans_input = self.stn3(x)          # [B, 3, 3]
            x = torch.bmm(trans_input, x)       # [B, 3, N]

        x = self.conv1(x)                       # [B, 64, N]

        feat_trans = None
        if self.feature_transform:
            feat_trans = self.stn64(x)          # [B, 64, 64]
            x = torch.bmm(feat_trans, x)        # [B, 64, N]

        local_feat = x                          # [B, 64, N] — segmentation에서 사용

        x = self.conv2(x)                       # [B, 128, N]
        x = self.conv3(x)                       # [B, 1024, N]

        global_feat = torch.max(x, 2)[0]        # [B, 1024]

        return global_feat, feat_trans, local_feat


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        global_feat, feat_trans, _ = self.pointnet_feat(pointcloud)
        logits = self.fc(global_feat)
        return logits, feat_trans


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.stn3 = STNKd(k=3)
        self.stn64 = STNKd(k=64)

        # Shared MLP 1: 3 → 64
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())

        # Shared MLP 2: 64 → 128 → 1024
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

        # Segmentation head: concat(local 64, global 1024) = 1088 → m
        self.seg_head = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, m, 1),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        N = pointcloud.shape[1]
        x = pointcloud.transpose(2, 1)          # [B, 3, N]

        trans_input = self.stn3(x)              # [B, 3, 3]
        x = torch.bmm(trans_input, x)           # [B, 3, N]

        x = self.conv1(x)                       # [B, 64, N]

        feat_trans = self.stn64(x)              # [B, 64, 64]
        x = torch.bmm(feat_trans, x)            # [B, 64, N]

        local_feat = x                          # [B, 64, N]

        x = self.conv2(x)                       # [B, 128, N]
        x = self.conv3(x)                       # [B, 1024, N]

        global_feat = torch.max(x, 2)[0]        # [B, 1024]
        global_feat = global_feat.unsqueeze(2).expand(-1, -1, N)  # [B, 1024, N]

        x = torch.cat([local_feat, global_feat], dim=1)  # [B, 1088, N]

        logits = self.seg_head(x)               # [B, 50, N]

        return logits, feat_trans



def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
