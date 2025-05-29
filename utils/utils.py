import io
import os
import base64
import random
import string
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from utils.prompt import generate_instruction
from utils.jigsaw_utils import create_jigsaw_image, create_grid_jigsaw_input, create_box_jigsaw_input


def master_print(s):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if local_rank == 0:
        print(s)


def bytes_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def base64_to_bytes(image_base64):
    return base64.b64decode(image_base64)


def image_to_bytes(element):
    if isinstance(element, str) and os.path.exists(element):
        image = Image.open(element)
    elif isinstance(element, Image.Image):
        image = element
    else:
        raise TypeError(f"{type(element)} | {element}")

    bytes_io = io.BytesIO()
    image = image.convert("RGB")
    image.save(bytes_io, format="JPEG", quality=100)

    return bytes_io.getvalue()


def bytes_to_image(element):
    if isinstance(element, bytes):
        image = Image.open(io.BytesIO(element))
        return image
    elif isinstance(element, str):
        image_bytes = base64_to_bytes(element)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    elif isinstance(element, Image.Image):
        return element
    else:
        raise TypeError(f"{type(element)} | {element}")


def normalize_box(box, width, height):
    box[0] /= width
    box[2] /= width
    box[1] /= height
    box[3] /= height

    return box


def denormalize_box(box, width, height):
    box[0] *= width
    box[2] *= width
    box[1] *= height
    box[3] *= height

    return box


def create_jigsaw_inputs(
    dataset,
    dataset_size,
    jigsaw_seed,
    m,
    n,
    n_c,
    shuffle_mn,
    mask_ratio,
    width_min,
    width_max,
    height_min,
    height_max,
    forward,
    question_type,
    think,
    instruct_model
):
    random.seed(jigsaw_seed)

    results = []
    choices = [i for i in range(1, m * n + 1)]

    for i, example in tqdm(enumerate(dataset)):
        if dataset_size > 0 and i >= dataset_size:
            break

        image = example["image"]
        if isinstance(image, list):
            idx = random.randint(0, len(image) - 1)
            image = image[idx]
        image = bytes_to_image(image)

        if shuffle_mn and random.random() < 0.5:
            m, n = n, m

        k = random.choice(choices)

        result = create_jigsaw_image(
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
        )
        result += (m, n, k)
        results.append(result)

    inputs = []
    for i, example in enumerate(dataset):
        if dataset_size > 0 and i >= dataset_size:
            break

        if question_type != "box":
            image_jigsaw, patches_forward, patches_backward, m, n, k = results[i]
            input = create_grid_jigsaw_input(
                image_jigsaw,
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
            )
        else:
            jigsaw_image, box_gt, m, n, k = results[i]
            input = create_box_jigsaw_input(
                jigsaw_image,
                box_gt,
                m,
                n,
                k,
                mask_ratio,
                forward,
                think,
                instruct_model
            )

        if "image_id" in example:
            input["image_id"] = str(example["image_id"])
        elif "filename" in example:
            input["image_id"] = str(example["filename"])

        inputs.append(input)

    return inputs


def create_inputs(
    dataset,
    dataset_size,
    question_key,
    solution_key,
    answer_type,
    think,
    instruct_model
):
    inputs = []

    for i, example in tqdm(enumerate(dataset)):
        if dataset_size > 0 and i >= dataset_size:
            break

        if "answer_type" in example:
            answer_type = example["answer_type"]

        question = example[question_key]
        prompt = generate_instruction(question, answer_type=answer_type, think=think, instruct_model=instruct_model)
        solution = example[solution_key]
        image = example["image"]

        if isinstance(image, list) and len(image) == 1:
            image = image[0]

        content = []
        if isinstance(image, list) and len(image) > 1:
            for img in image:
                img = bytes_to_image(img)
                image_content = {"type": "image", "image": img}
                content.append(image_content)
        else:
            image = bytes_to_image(image)
            image_content = {"type": "image", "image": image}
            content.append(image_content)

        text_content = {"type": "text", "text": prompt}
        content.append(text_content)

        if answer_type == "letter":
            solution = solution.replace("(", "")[0]
            assert solution in string.ascii_uppercase

        if answer_type in ["box", "point"]:
            assert not isinstance(image, list)
            width, height = image.size
            solution = normalize_box(solution, width, height)

        if instruct_model:
            prompt_input = [{"role": "user", "content": content}]
        else:
            prompt_input = content

        input = {
            "prompt": prompt_input,
            "solution": solution,
            "answer_type": answer_type,
            "think": think
        }

        if "task" in example:
            input["task"] = example["task"]

        if "image_id" in example:
            input["image_id"] = str(example["image_id"])
        elif "filename" in example:
            input["image_id"] = str(example["filename"])

        inputs.append(input)

    return inputs


def create_input_helper(
    dataset_name,
    dataset_split,
    dataset_size,
    jigsaw,
    jigsaw_seed,
    m,
    n,
    n_c,
    shuffle_mn,
    mask_ratio,
    width_min,
    width_max,
    height_min,
    height_max,
    forward,
    question_type,
    think,
    instruct_model,
    **args
):
    assert dataset_name in [
        "coco",
        "cv_bench",
        "mmvp",
        "sat_static",
        "super_clevr",
        "screenspot",
    ]

    if dataset_name == "cv_bench":
        dataset = load_dataset("nyu-visionx/CV-Bench", split=dataset_split)
    else:
        dataset = load_dataset(f"jigsaw-r1/{dataset_name}", split=dataset_split)

    if jigsaw:
        assert m > 0 and n > 0 and m * n > 1
        assert n_c < len(string.ascii_uppercase)

        assert 0 < width_min < 1 and 0 < width_max < 1
        assert 0 < height_min < 1 and 0 < height_max < 1

        inputs = create_jigsaw_inputs(
            dataset,
            dataset_size,
            jigsaw_seed,
            m,
            n,
            n_c,
            shuffle_mn,
            mask_ratio,
            width_min,
            width_max,
            height_min,
            height_max,
            forward,
            question_type,
            think,
            instruct_model
        )
    else:
        if dataset_name == "cv_bench":
            question_key = "prompt"
            solution_key = "answer"
            answer_type = "letter"
        elif dataset_name == "mmvp":
            question_key = "prompt"
            solution_key = "answer"
            answer_type = "letter"
        elif dataset_name == "sat_static":
            question_key = "prompt"
            solution_key = "ground_truth"
            answer_type = "letter"
        elif dataset_name == "super_clevr":
            question_key = "question"
            solution_key = "ground_truth"
            answer_type = "digit"
        elif dataset_name == "screenspot":
            question_key = "prompt"
            solution_key = "solution"
            answer_type = "point"
        else:
            raise NotImplementedError

        inputs = create_inputs(
            dataset,
            dataset_size,
            question_key,
            solution_key,
            answer_type,
            think,
            instruct_model
        )

    return inputs
