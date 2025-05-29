import random
import string
import numpy as np
from PIL import Image

from utils.prompt import generate_grid_jigsaw_prompt, generate_box_jigsaw_prompt, generate_instruction


def crop_image(image, m, n):
    width, height = image.size

    left = 0
    top = 0
    right = (width // n) * n
    bottom = (height // m) * m

    image_cropped = image.crop((left, top, right, bottom))

    return image_cropped


def mask_image(image, m, n, mask_ratio):
    width, height = image.size
    patch_width = width // n
    patch_height = height // m

    image = np.array(image)
    for i in range(m):
        for j in range(n):
            left = j * patch_width
            top = i * patch_height
            right = left + patch_width
            bottom = top + patch_height

            if i == 0:
                border_top = top + int(patch_height * mask_ratio)
            else:
                border_top = top + int(patch_height * mask_ratio) // 2

            if i == m - 1:
                border_bottom = bottom - int(patch_height * mask_ratio)
            else:
                border_bottom = bottom - int(patch_height * mask_ratio) // 2

            if j == 0:
                border_left = left + int(patch_width * mask_ratio)
            else:
                border_left = left + int(patch_width * mask_ratio) // 2

            if j == n - 1:
                border_right = right - int(patch_width * mask_ratio)
            else:
                border_right = right - int(patch_width * mask_ratio) // 2

            image[top : border_top, :] = 0
            image[border_bottom : bottom, :] = 0
            image[:, left : border_left] = 0
            image[:, border_right : right] = 0

    image = Image.fromarray(image)

    return image


def create_grid_jigsaw_image(image, m, n):
    width, height = image.size

    assert (width % n == 0) and (height % m == 0), f"width: {width}, height: {height}, m: {m}, n: {n}"

    patch_width = width // n
    patch_height = height // m

    patches = []
    for i in range(m):
        for j in range(n):
            left = j * patch_width
            top = i * patch_height
            right = left + patch_width
            bottom = top + patch_height

            index = i * n + j
            patch = image.crop((left, top, right, bottom))
            patches.append((index, patch))

    random.shuffle(patches)

    jigsaw_image = Image.new("RGB", (width, height))
    patches_forward = [None] * (m * n)
    patches_backward = [None] * (m * n)

    for i in range(m):
        for j in range(n):
            left = j * patch_width
            top = i * patch_height
            right = left + patch_width
            bottom = top + patch_height

            index_new = i * n + j
            index_ori, patch = patches.pop()

            jigsaw_image.paste(patch, (left, top, right, bottom))
            patches_forward[index_new] = index_ori + 1
            patches_backward[index_ori] = index_new + 1

    return jigsaw_image, patches_forward, patches_backward


def create_box_jigsaw_image(
    image,
    m,
    n,
    k,
    mask_ratio,
    width_min,
    width_max,
    height_min,
    height_max,
):
    width, height = image.size

    assert (width % n == 0) and (height % m == 0), f"width: {width}, height: {height}, m: {m}, n: {n}"

    patch_width = width // n
    patch_height = height // m

    width_min = int(width_min * patch_width)
    width_max = int(width_max * patch_width)
    height_min = int(height_min * patch_height)
    height_max = int(height_max * patch_height)

    box_width = random.randint(width_min, width_max)
    box_height = random.randint(height_min, height_max)

    boxes = {}
    for i in range(m):
        for j in range(n):
            if i == 0:
                border_top = int(patch_height * mask_ratio)
            else:
                border_top = int(patch_height * mask_ratio) // 2

            if i == m - 1:
                border_bottom = int(patch_height * mask_ratio)
            else:
                border_bottom = int(patch_height * mask_ratio) // 2

            if j == 0:
                border_left = int(patch_width * mask_ratio)
            else:
                border_left = int(patch_width * mask_ratio) // 2

            if j == n - 1:
                border_right = int(patch_width * mask_ratio)
            else:
                border_right = int(patch_width * mask_ratio) // 2

            left = j * patch_width
            top = i * patch_height

            left_max = patch_width - box_width - border_right
            top_max = patch_height - box_height - border_bottom

            if left_max < border_left or top_max < border_top:
                raise RuntimeError("mask ratio is too large")

            x_min = left + random.randint(border_left, left_max)
            y_min = top + random.randint(border_top, top_max)

            x_max = x_min + box_width
            y_max = y_min + box_height

            index = i * n + j
            boxes[index] = [x_min, y_min, x_max, y_max]

    indexes = list(range(m * n))
    random.shuffle(indexes)

    k0 = k - 1
    if indexes[k0] == k0:
        choices = [i for i in range(m * n) if i != k0]
        c = random.choice(choices)
        indexes[k0], indexes[c] = indexes[c], indexes[k0]

    jigsaw_image = image.copy()

    for i in range(m * n):
        index_ori = i
        index_new = indexes[i]

        box_ori = boxes[index_ori]
        box_new = boxes[index_new]

        box_ori_cropped = image.crop(box_ori)
        jigsaw_image.paste(box_ori_cropped, box_new)

    #index_gt = indexes[k0]
    index_gt = indexes.index(k0)
    box_gt = boxes[index_gt]

    box_gt[0] /= width
    box_gt[2] /= width
    box_gt[1] /= height
    box_gt[3] /= height

    return jigsaw_image, box_gt


def create_jigsaw_image(
    image,
    m,
    n,
    k,
    mask_ratio,
    width_min,
    width_max,
    height_min,
    height_max,
    question_type
):
    image_cropped = crop_image(image, m, n)

    if question_type != "box":
        jigsaw_image, patches_forward, patches_backward = create_grid_jigsaw_image(image_cropped, m, n)

        if mask_ratio > 0:
            jigsaw_image = mask_image(jigsaw_image, m, n, mask_ratio)

        return jigsaw_image, patches_forward, patches_backward
    else:
        if mask_ratio > 0:
            image_cropped = mask_image(image_cropped, m, n, mask_ratio)

        jigsaw_image, box_gt = create_box_jigsaw_image(
            image_cropped,
            m,
            n,
            k,
            mask_ratio,
            width_min,
            width_max,
            height_min,
            height_max,
        )

        return jigsaw_image, box_gt


def create_grid_jigsaw_input(
    jigsaw_image,
    patches_forward,
    patches_backward,
    m,
    n,
    n_c,
    mask_ratio,
    forward,
    question_type,
    think,
    instruct_model
):
    if forward:
        patches_gt = patches_forward
    else:
        patches_gt = patches_backward

    if question_type == "full":
        n_c = pos = k = l = None
        solution = patches_gt
        answer_type = "grid"
    elif question_type == "multiple":
        pos = random.randint(0, n_c - 1)
        k = l = None
        solution = string.ascii_uppercase[pos]
        answer_type = "letter"
    elif question_type == "pair":
        if m == 1 or n == 1:
            n_c = 2
        else:
            n_c = 8
        pos = random.randint(0, n_c - 1)
        k, l = random.sample(range(1, m * n + 1), 2)
        solution = string.ascii_uppercase[pos]
        answer_type = "letter"
    elif question_type == "single":
        n_c = pos = l = None
        k = random.randint(1, m * n)
        solution = str(patches_gt[k - 1])
        answer_type = "digit"
    else:
        raise NotImplementedError

    question = generate_grid_jigsaw_prompt(m, n, n_c, pos, k, l, mask_ratio, patches_gt, forward, question_type)
    prompt = generate_instruction(question, m, n, answer_type, think, instruct_model)

    if instruct_model:
        prompt_input = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": jigsaw_image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
    else:
        prompt_input = [
            {"type": "image", "image": jigsaw_image},
            {"type": "text", "text": prompt}
        ]

    input = {
        "prompt": prompt_input,
        "solution": solution,
        "m": m,
        "n": n,
        "mask_ratio": mask_ratio,
        "forward": forward,
        "question_type": question_type,
        "answer_type": answer_type,
        "think": think
    }

    return input


def create_box_jigsaw_input(
    jigsaw_image,
    box_gt,
    m,
    n,
    k,
    mask_ratio,
    forward,
    think,
    instruct_model
):
    answer_type = "box"

    question = generate_box_jigsaw_prompt(m, n, k, mask_ratio, forward)
    prompt = generate_instruction(question, m, n, answer_type, think, instruct_model)

    if instruct_model:
        prompt_input = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": jigsaw_image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
    else:
        prompt_input = [
            {"type": "image", "image": jigsaw_image},
            {"type": "text", "text": prompt}
        ]

    input = {
        "prompt": prompt_input,
        "solution": box_gt,
        "m": m,
        "n": n,
        "mask_ratio": mask_ratio,
        "answer_type": answer_type,
        "think": think
    }

    return input
