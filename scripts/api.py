from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

from modules.api.models import *
from modules.api import api
import time
import gradio as gr
from oot_diffusion.inference_segmentation import ClothesMaskModel
DEFAULT_HG_ROOT = Path(os.getcwd()) / "ootd_models"

try:
    from helper.logging import Logger
    logger = Logger("OOTD")
except Exception:
    import logging
    logger = logging.getLogger("OOTD")

def ootd_api(_: gr.Blocks, app: FastAPI):
    @app.post("/sdapi/v2/ootd/getmask")
    async def getmask(
        image: str = Body("", title='input image'),
    ):
        t = time.time()
        logger.info("/sdapi/v2/ootd/getmask start")
        cmm = ClothesMaskModel(
            hg_root=DEFAULT_HG_ROOT,
            cache_dir=None,
        )
        (masked_vton_img, mask, model_image, model_parse, face_mask) = cmm.generate(
            model_path=api.decode_base64_to_image(image)
        )

        logger.info(f"/sdapi/v2/ootd/getmask done in {(time.time() - t):.3f}")
        return [
            api.encode_pil_to_base64(mask),
            api.encode_pil_to_base64(model_parse),
        ]

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(ootd_api)
except:
    pass
