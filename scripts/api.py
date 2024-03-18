import os
from pathlib import Path
import time

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import gradio as gr
import numpy as np
from PIL import Image

from oot_diffusion.inference_segmentation import ClothesMaskModel

from modules.api.models import *
from modules.api import api


DEFAULT_HG_ROOT = Path(__file__).parent.parent / "ootd_models"
DEFAULT_CACHE = Path(__file__).parent.parent / ".cache"

class GetMaskRequest(BaseModel):
    image: str

try:
    from helper.logging import Logger
    logger = Logger("OOTD")
except Exception:
    import logging
    logger = logging.getLogger("OOTD")

def get_masked_image(input_image, mask_image):
    input_array = np.array(input_image)
    mask_array = np.array(mask_image)
    mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
    masked_array = np.zeros_like(input_array)
    for c in range(input_array.shape[2]):
        masked_array[:,:,c] = np.where(mask_array == 255, input_array[:,:,c], 0)
    mask_image = Image.fromarray(mask_array)
    masked_image = Image.fromarray(masked_array)
    return mask_image, masked_image

def ootd_api(_: gr.Blocks, app: FastAPI):
    @app.post("/sdapi/v2/ootd/getmask")
    async def getmask(
        data: GetMaskRequest
    ):
        t = time.time()
        logger.info("/sdapi/v2/ootd/getmask start")
        cmm = ClothesMaskModel(
            hg_root=str(DEFAULT_HG_ROOT),
            cache_dir=str(DEFAULT_CACHE),
        )
        try:
            (_, cloth_mask, model_image, model_parse, _) = cmm.generate(
                model_path=api.decode_base64_to_image(data.image)
            )
        except Exception as e:
            return HTTPException(status_code=500, detail=str("Get mask error: " + str(e)))
        # convert model_parse to model_mask
        model_np = np.array(model_parse.convert("RGB"))
        model_array = ~np.all(model_np == [0, 0, 0], axis=-1)
        model_mask = Image.fromarray(model_array.astype(np.uint8) * 255)
        
        cloth_mask, cloth_masked = get_masked_image(model_image, cloth_mask)
        model_mask, model_masked = get_masked_image(model_image, model_mask)

        logger.info(f"/sdapi/v2/ootd/getmask done in {(time.time() - t):.3f}")
        return [
            api.encode_pil_to_base64(cloth_mask),
            api.encode_pil_to_base64(cloth_masked),
            api.encode_pil_to_base64(model_mask),
            api.encode_pil_to_base64(model_masked),
        ]

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(ootd_api)
except:
    pass
