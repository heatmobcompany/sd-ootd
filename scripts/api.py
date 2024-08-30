import os
from pathlib import Path
import time

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import gradio as gr
import numpy as np
import cv2
from PIL import Image

from oot_diffusion.inference_segmentation import ClothesMaskModel
from oot_diffusion.inference_with_mask import OOTDiffusionWithMaskModel

from modules.api.models import *
from modules.api import api


DEFAULT_HG_ROOT = Path(__file__).parent.parent / "ootd_models"
DEFAULT_CACHE = Path(__file__).parent.parent / ".cache"


class GetMaskRequest(BaseModel):
    image: str


class TryOutfitRequest(BaseModel):
    cloth_image: str
    mask_image: str
    model_image: str

cmm = ClothesMaskModel(
    hg_root=str(DEFAULT_HG_ROOT),
    cache_dir=str(DEFAULT_CACHE),
)

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
        masked_array[:, :, c] = np.where(
            mask_array == 255, input_array[:, :, c], 0)
    mask_image = Image.fromarray(mask_array)
    masked_image = Image.fromarray(masked_array)
    return mask_image, masked_image

def calculate_body_ratio(image):
    image = np.array(image)    
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    image_height = gray_image.shape[0]
    white_pixels = np.any(gray_image == 255, axis=1)
    white_height = np.sum(white_pixels)
    body_ratio = white_height / image_height
    return body_ratio

def calculate_blur(image):
    image = np.array(image)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    blur_value = laplacian.var()
    return blur_value

def ootd_api(_: gr.Blocks, app: FastAPI):
    @app.post("/sdapi/v2/ootd/getmask")
    async def getmask(
        data: GetMaskRequest
    ):
        t = time.time()
        logger.info("/sdapi/v2/ootd/getmask start")
        try:
            (_, cloth_mask, model_image, model_parse, body_mask) = cmm.generate(
                model_path=api.decode_base64_to_image(data.image),
                category="fullbody",
            )
        except Exception as e:
            return HTTPException(status_code=500, detail=str("Get mask error: " + str(e)))
        # convert model_parse to model_mask
        model_np = np.array(model_parse.convert("RGB"))
        model_array = ~np.all(model_np == [0, 0, 0], axis=-1)

        cloth_mask, cloth_masked = get_masked_image(model_image, cloth_mask)
        body_mask, body_masked = get_masked_image(model_image, body_mask)

        logger.info(f"/sdapi/v2/ootd/getmask done in {(time.time() - t):.3f}")
        return [
            api.encode_pil_to_base64(cloth_mask),
            api.encode_pil_to_base64(cloth_masked),
            api.encode_pil_to_base64(body_mask),
            api.encode_pil_to_base64(body_masked),
        ]

    @app.post("/sdapi/v2/ootd/analyze-model")
    async def analyze_image(
        data: GetMaskRequest
    ):
        t = time.time()
        logger.info("/sdapi/v2/ootd/analyze-model start")
        try:
            (_, cloth_mask, model_image, model_parse, body_mask) = cmm.generate(
                model_path=api.decode_base64_to_image(data.image),
                category="fullbody",
            )
        except Exception as e:
            return HTTPException(status_code=500, detail=str("Get mask error: " + str(e)))
        # convert model_parse to model_mask
        model_np = np.array(model_parse.convert("RGB"))
        model_array = ~np.all(model_np == [0, 0, 0], axis=-1)

        cloth_mask, cloth_masked = get_masked_image(model_image, cloth_mask)
        body_mask, body_masked = get_masked_image(model_image, body_mask)
        
        logger.info(f"/sdapi/v2/ootd/analyze-model done in {(time.time() - t):.3f}")
        return {
            "body_ratio": calculate_body_ratio(body_mask),
            "blur_level": calculate_blur(body_masked),
            "images": [
            api.encode_pil_to_base64(cloth_mask),
            api.encode_pil_to_base64(cloth_masked),
            api.encode_pil_to_base64(body_mask),
            api.encode_pil_to_base64(body_masked),
            ]
        }
        
    @app.post("/sdapi/v2/ootd/try-outfit")
    async def tryOutfit(
        data: TryOutfitRequest
    ):
        t = time.time()
        logger.info("/sdapi/v2/ootd/try-outfit start")
        cmm = OOTDiffusionWithMaskModel(
            hg_root=str(DEFAULT_HG_ROOT),
            cache_dir=str(DEFAULT_CACHE),
        )
        start_time = time.perf_counter()
        cmm.load_pipe()
        end_time_load_model = time.perf_counter()
        print(
            f"Model loaded in {end_time_load_model - start_time:.2f} seconds.")

        try:
            start_generate_time = time.perf_counter()
            result_images = cmm.generate(
                model_path=api.decode_base64_to_image(data.model_image),
                cloth_path=api.decode_base64_to_image(data.cloth_image),
                model_mask_path=api.decode_base64_to_image(data.mask_image),
            )
            end_generate_time = time.perf_counter()
            print(
                f"Generated in {end_generate_time - start_generate_time:.2f} seconds.")
        except Exception as e:
            return HTTPException(status_code=500, detail=str("Generate error: " + str(e)))

        base64_images = []
        for i, result_image in enumerate(result_images):
            base64_images.append(api.encode_pil_to_base64(result_image))

        logger.info(
            f"/sdapi/v2/ootd/try-outfit done in {(time.time() - t):.3f}")
        return base64_images


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(ootd_api)
except:
    pass
