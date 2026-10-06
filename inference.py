"""AutoPerception — 3D BEV Object Detection (LiDAR-Camera Fusion)"""
import torch, torch.nn as nn, numpy as np

class PointPillarVoxelizer(nn.Module):
    """Simplified PointPillars voxelization for BEV detection."""
    def __init__(self, voxel_size=(0.16,0.16,4.0), pc_range=(-51.2,-51.2,-5,51.2,51.2,3)):
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        nx = int((pc_range[3]-pc_range[0])/voxel_size[0])
        ny = int((pc_range[4]-pc_range[1])/voxel_size[1])
        self.nx, self.ny = nx, ny
    def forward(self, points: torch.Tensor):
        # points: (N, 4) — x, y, z, intensity
        mask = ((points[:,0] >= self.pc_range[0]) & (points[:,0] < self.pc_range[3]) &
                (points[:,1] >= self.pc_range[1]) & (points[:,1] < self.pc_range[4]) &
                (points[:,2] >= self.pc_range[2]) & (points[:,2] < self.pc_range[5]))
        pts = points[mask]
        xi = ((pts[:,0]-self.pc_range[0])/self.voxel_size[0]).long().clamp(0, self.nx-1)
        yi = ((pts[:,1]-self.pc_range[1])/self.voxel_size[1]).long().clamp(0, self.ny-1)
        bev = torch.zeros(1, self.ny, self.nx, device=points.device)
        bev[0, yi, xi] = pts[:,3]  # intensity as feature
        return bev

class BEVBackbone(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class DetectionHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=10, num_anchors=2):
        super().__init__()
        self.cls = nn.Conv2d(in_ch, num_classes * num_anchors, 1)
        self.reg = nn.Conv2d(in_ch, 7 * num_anchors, 1)  # x,y,z,w,l,h,rot
    def forward(self, x): return self.cls(x), self.reg(x)

if __name__ == "__main__":
    voxelizer = PointPillarVoxelizer(); backbone = BEVBackbone(); head = DetectionHead()
    pts = torch.randn(50000, 4)
    bev = voxelizer(pts)
    print(f"BEV map: {bev.shape}")
    feats = backbone(bev.unsqueeze(0))
    cls_out, reg_out = head(feats)
    print(f"Cls: {cls_out.shape} | Reg: {reg_out.shape}")
