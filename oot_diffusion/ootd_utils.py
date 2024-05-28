from typing import Literal
import numpy as np
import cv2
from PIL import Image, ImageDraw

from oot_diffusion.humanparsing.utility import label_map, remove_outliers


def extend_arm_mask(wrist, elbow, scale):
    wrist = elbow + scale * (wrist - elbow)
    return wrist


def hole_fill_ootd(img: np.ndarray):
    img = np.pad(img[1:-1, 1:-1], pad_width=1, mode="constant", constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def refine_mask_ootd(mask):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask


def get_mask_location(
    model_type: Literal["hd", "dc"],
    category: Literal["upper_body", "lower_body", "dresses"],
    model_parse: Image.Image,
    keypoint: dict,
    width: int = 384,
    height: int = 512,
):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == "hd":
        arm_width = 30
    elif model_type == "dc":
        arm_width = 25
    else:
        raise ValueError("model_type must be 'hd' or 'dc'!")

    parse_head = (
        (parse_array == 1).astype(np.float32)
        + (parse_array == 3).astype(np.float32)
        + (parse_array == 11).astype(np.float32)
    )

    parser_mask_fixed = (
        (parse_array == label_map["left_shoe"]).astype(np.float32)
        + (parse_array == label_map["right_shoe"]).astype(np.float32)
        + (parse_array == label_map["hat"]).astype(np.float32)
        + (parse_array == label_map["sunglasses"]).astype(np.float32)
    )

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)
    arms = arms_left + arms_right

    if category == "dresses":
        parse_mask = (
            (parse_array == 7).astype(np.float32)
            + (parse_array == 4).astype(np.float32)
            + (parse_array == 5).astype(np.float32)
            + (parse_array == 6).astype(np.float32)
        )

        parser_mask_changeable += np.logical_and(
            parse_array, np.logical_not(parser_mask_fixed)
        )

    elif category == "upper_body":
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(
            np.float32
        )
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(
            np.float32
        ) + (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(
            parse_array, np.logical_not(parser_mask_fixed)
        )
    elif category == "lower_body":
        parse_mask = (
            (parse_array == 6).astype(np.float32)
            + (parse_array == 12).astype(np.float32)
            + (parse_array == 13).astype(np.float32)
            + (parse_array == 5).astype(np.float32)
        )
        parser_mask_fixed += (
            (parse_array == label_map["upper_clothes"]).astype(np.float32)
            + (parse_array == 14).astype(np.float32)
            + (parse_array == 15).astype(np.float32)
        )
        parser_mask_changeable += np.logical_and(
            parse_array, np.logical_not(parser_mask_fixed)
        )
    elif category == "full_body":
        parse_mask = (
            (parse_array == 7).astype(np.float32)
            + (parse_array == 4).astype(np.float32)
            + (parse_array == 5).astype(np.float32)
            + (parse_array == 6).astype(np.float32)
            + (parse_array == 8).astype(np.float32)
            + (parse_array == 16).astype(np.float32)
            + (parse_array == 17).astype(np.float32)
        )

        parser_mask_changeable += np.logical_and(
            parse_array, np.logical_not(parser_mask_fixed)
        )
    else:
        raise NotImplementedError

    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    # pose_data = np.array(pose_data)
    # pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new("L", (width, height))
    im_arms_right = Image.new("L", (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    if category == "dresses" or category == "upper_body" or category == "full_body":
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        shoulder_right = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[8][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[9][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[10][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [
            shoulder_left[0] - ARM_LINE_WIDTH // 2,
            shoulder_left[1] - ARM_LINE_WIDTH // 2,
            shoulder_left[0] + ARM_LINE_WIDTH // 2,
            shoulder_left[1] + ARM_LINE_WIDTH // 2,
        ]
        size_right = [
            shoulder_right[0] - ARM_LINE_WIDTH // 2,
            shoulder_right[1] - ARM_LINE_WIDTH // 2,
            shoulder_right[0] + ARM_LINE_WIDTH // 2,
            shoulder_right[1] + ARM_LINE_WIDTH // 2,
        ]

        if wrist_right[0] <= 1.0 and wrist_right[1] <= 1.0:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(
                np.concatenate((shoulder_right, elbow_right, wrist_right))
                .astype(np.uint16)
                .tolist(),
                "white",
                ARM_LINE_WIDTH,
                "curve",
            )
            arms_draw_right.arc(size_right, 0, 360, "white", ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1.0 and wrist_left[1] <= 1.0:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(
                np.concatenate((wrist_left, elbow_left, shoulder_left))
                .astype(np.uint16)
                .tolist(),
                "white",
                ARM_LINE_WIDTH,
                "curve",
            )
            arms_draw_left.arc(size_left, 0, 360, "white", ARM_LINE_WIDTH // 2)

        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    if category == "dresses" or category == "upper_body" or category == "full_body":
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(
            np.logical_or(im_arms_left, im_arms_right).astype("float32"),
            np.ones((5, 5), np.uint16),
            iterations=4,
        )
        parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill_ootd(img.astype(np.uint8))
    dst = remove_outliers(img.astype(np.uint8))
    dst = refine_mask_ootd(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray


def resize_crop_center(
    image: Image.Image, target_width: int, target_height: int
) -> Image.Image:
    # Calculate target aspect ratio
    target_aspect_ratio = target_width / target_height

    # Get current size
    orig_width, orig_height = image.size

    # Calculate current aspect ratio
    orig_aspect_ratio = orig_width / orig_height

    # Determine the dimensions for cropping
    if orig_aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio
        new_width = int(target_aspect_ratio * orig_height)
        new_height = orig_height
        left = (orig_width - new_width) / 2
        top = 0
        right = (orig_width + new_width) / 2
        bottom = orig_height
    else:
        # Image is taller than target aspect ratio
        new_width = orig_width
        new_height = int(orig_width / target_aspect_ratio)
        left = 0
        top = (orig_height - new_height) / 2
        right = orig_width
        bottom = (orig_height + new_height) / 2

    # Crop the image to the calculated dimensions
    cropped_image = image.crop((int(left), int(top), int(right), int(bottom)))

    # Resize the cropped image to the target size
    resized_image = cropped_image.resize((target_width, target_height), Image.LANCZOS)

    return resized_image
