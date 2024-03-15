from ultralytics import YOLO, settings
from PIL import Image
import os

class PoseModel:
    def __init__(self, hg_root: str, cache_dir: str = None):
        self.hg_root = hg_root
        self.cache_dir = cache_dir
        settings.update({"weights_dir": hg_root})

    def load_pose_model(self):
        self.pose_model = YOLO(os.path.join(self.cache_dir if self.cache_dir is not None else "", "yolov8n-pose.pt"))

    def infer_keypoints(self, image: str | bytes | Image.Image):
        pose_results = self.pose_model(
            image,
        )

        return {"pose_keypoints_2d": pose_results[0].keypoints[0].xy.cpu().numpy()[0]}
