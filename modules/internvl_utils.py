import copy
import base64
import requests
from PIL import Image
from io import BytesIO

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def extract_system_message(conversation_list):
    if conversation_list[0]["role"] == "system":
        if isinstance(conversation_list[0]["content"], list):
            return conversation_list[0]["content"][0]["text"]
        else:
            return conversation_list[0]["content"]

    return None


def process_conversation_list(conversation_list, system_message=None, image_newline=True):
    if system_message is not None:
        conversation_list = conversation_list[1:]

    processed_list = []

    for item in conversation_list:
        role = item["role"]
        content = item["content"]

        if isinstance(content, list):
            overall_str = ""
            for content_item in content:
                if content_item.get("type") == "image":
                    overall_str += "<image>" if not image_newline else "<image>\n"
                elif content_item.get("type") == "text":
                    overall_str += content_item.get("text")
                else:
                    raise ValueError(f"Unsupported content type: {type(content_item)}")
            processed_list.append(overall_str)
        elif isinstance(content, str):
            processed_list.append(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    return processed_list


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]

    if "role" in conversations[0][0]:
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele.get("type","") in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
    else:
        for message in conversations:
            if isinstance(message, list):
                for ele in message:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type","") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)

    return vision_infos


def fetch_image(ele: dict[str, str | Image.Image]) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")

    return image


def process_internvl_vision_info(conversations: list[dict] | list[list[dict]]):
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        else:
            raise ValueError("image or image_url should in content.")
    if len(image_inputs) == 0:
        image_inputs = None

    return image_inputs


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, max_num=max_num, image_size=input_size, use_thumbnail=True)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)

    return pixel_values
