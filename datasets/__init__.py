from .dataset import SceneFlowDataset, KITTIDataset, DrivingStereoDataset, MobiDepthDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
    "mobidepth": MobiDepthDataset
}
