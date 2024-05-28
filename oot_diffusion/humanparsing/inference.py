import pdb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import load_atr_model, load_lip_model, inference
import torch

lip_model_path = "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth"
atr_model_path = "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908301523-atr.pth"

class BodyParsingModel:
    def __init__(self, gpu_id: int = 0, cache_dir: str = ""):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        models_path = Path(cache_dir) / "humanparsing"
        atr_model_local_path = models_path / "exp-schp-201908301523-atr.pth"
        lip_model_local_path = models_path / "exp-schp-201908261155-lip.pth"
        if not atr_model_local_path.exists():
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(atr_model_path, models_path, "exp-schp-201908301523-atr.pth")
        if not lip_model_local_path.exists():
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(lip_model_path, models_path, "exp-schp-201908261155-lip.pth")
        self.atr_model = load_atr_model(atr_model_local_path)
        self.lip_model = load_lip_model(lip_model_local_path)
        

    def infer_parse_model(self, input_image):
        torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = inference(self.atr_model, self.lip_model, input_image)
        return parsed_image, face_mask
