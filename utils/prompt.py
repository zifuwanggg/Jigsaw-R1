import random
import string


def generate_jigsaw_string(m, n, patches):
    jigsaw_string = ""
    for i in range(m):
        for j in range(n):
            index = i * n + j
            patch = patches[index]
            jigsaw_string += f"{patch} "
        jigsaw_string = jigsaw_string[:-1] + "\n"

    jigsaw_string = jigsaw_string[:-1]

    return jigsaw_string


def generate_question_string_multiple(m, n, n_c, pos, gt_string):
    assert 0 <= pos <= n_c - 1

    count = 0
    choices = set()
    question_string = ""
    letters = string.ascii_uppercase

    while count < n_c:
        patches = list(range(1, m * n + 1))
        random.shuffle(patches)
        jigsaw_string = generate_jigsaw_string(m, n, patches)
        letter = letters[count]

        if count == pos:
            count += 1
            choices.add(gt_string)
            question_string += f"({letter})\n{gt_string}\n\n"
        elif jigsaw_string not in choices:
            count += 1
            choices.add(jigsaw_string)
            question_string += f"({letter})\n{jigsaw_string}\n\n"

    question_string = question_string[:-2]

    return question_string


def generate_question_string_pair(m, n, pos, k, l, patches_gt):
    choices = [
        f"{k} is directly to the left of {l}",
        f"{k} is directly to the right of {l}",
        f"{k} is directly above {l}",
        f"{k} is directly below {l}",
        f"{k} is on the upper left of {l}",
        f"{k} is on the upper right of {l}",
        f"{k} is on the lower left of {l}",
        f"{k} is on the lower right of {l}"
    ]

    if m == 1:
        choices = choices[:2]
    elif n == 1:
        choices = choices[2:4]

    assert 0 <= pos <= len(choices) - 1

    k_ori = patches_gt[k - 1]
    l_ori = patches_gt[l - 1]

    k_row = (k_ori - 1) // n
    k_col = (k_ori - 1) % n
    l_row = (l_ori - 1) // n
    l_col = (l_ori - 1) % n

    if k_row == l_row and k_col < l_col:
        gt = f"{k} is directly to the left of {l}"
    elif k_row == l_row and k_col > l_col:
        gt = f"{k} is directly to the right of {l}"
    elif k_row < l_row and k_col == l_col:
        gt = f"{k} is directly above {l}"
    elif k_row > l_row and k_col == l_col:
        gt = f"{k} is directly below {l}"
    elif k_row < l_row and k_col < l_col:
        gt = f"{k} is on the upper left of {l}"
    elif k_row < l_row and k_col > l_col:
        gt = f"{k} is on the upper right of {l}"
    elif k_row > l_row and k_col < l_col:
        gt = f"{k} is on the lower left of {l}"
    elif k_row > l_row and k_col > l_col:
        gt = f"{k} is on the lower right of {l}"

    random.shuffle(choices)
    idx = choices.index(gt)
    choices[idx], choices[pos] = choices[pos], choices[idx]

    letters = string.ascii_uppercase
    question_string = ""
    for i, choice in enumerate(choices):
        letter = letters[i]
        question_string += f"({letter}) {choice}\n"

    question_string = question_string[:-1]

    return question_string


def generate_grid_jigsaw_prompt(
    m,
    n,
    n_c=None,
    pos=None,
    k=None,
    l=None,
    mask_ratio=0,
    patches_gt=None,
    forward=True,
    question_type="full"
):
    if m == 1:
        directions = ["left", "right"]
    elif n == 1:
        directions = ["top", "bottom"]
    else:
        directions = ["top-left", "bottom-right"]

    if mask_ratio > 0:
        mask_info = " To increase the task difficulty, the borders of each patch have been masked out."
    else:
        mask_info = ""

    jigsaw_string = generate_jigsaw_string(m, n, range(1, m * n + 1))

    prompt = \
f"""The input image is divided into {m}x{n} patches that have been randomly permuted from their original positions. Your task is to solve this {m}x{n} jigsaw puzzle and reconstruct the original image.{mask_info}

Consider a {m}x{n} grid, where each number represents a position index ranging from 1 ({directions[0]}) to {m * n} ({directions[1]}):

```
{jigsaw_string}
```

"""

    if forward:
        if question_type == "full":
            prompt += \
f"""For each patch, determine its correct position index in the original image. If a patch currently at position X should belong at position Y, place "Y" at position X."""

        elif question_type == "multiple":
            gt_string = generate_jigsaw_string(m, n, patches_gt)
            question_string = generate_question_string_multiple(m, n, n_c, pos, gt_string)
            prompt += \
f"""For each patch, determine its correct position index in the original image. If a patch currently at position X should belong at position Y, place "Y" at position X.

Select the correct answer from the following {n_c} choices:

{question_string}"""

        elif question_type == "pair":
            question_string = generate_question_string_pair(m, n, pos, k, l, patches_gt)
            n_c = question_string.count("(")
            prompt += \
f"""For patches currently at positions {k} and {l}, determine their relative position in the original image.

Select the correct answer from the following {n_c} choices:

{question_string}"""

        elif question_type == "single":
            prompt += \
f"""For the patch currently at position {k}, determine its correct position index in the original image."""

        else:
            raise NotImplementedError

    else:
        if question_type == "full":
            prompt += \
f"""For each position, determine which patch should belong there. If a patch currently at position X should belong at position Y, place "X" at position Y."""

        elif question_type == "multiple":
            gt_string = generate_jigsaw_string(m, n, patches_gt)
            question_string = generate_question_string_multiple(m, n, n_c, pos, gt_string)
            prompt += \
f"""For each position, determine which patch should belong there. If a patch currently at position X should belong at position Y, place "X" at position Y.

Select the correct answer from the following {n_c} choices:

{question_string}"""

        elif question_type == "single":
            prompt += \
f"""Determine which patch belongs at position {k} in the original image."""

        else:
            raise NotImplementedError

    return prompt


def generate_box_jigsaw_prompt(m, n, k, mask_ratio, forward):
    if m == 1:
        directions = ["left", "right"]
    elif n == 1:
        directions = ["top", "bottom"]
    else:
        directions = ["top-left", "bottom-right"]

    if mask_ratio > 0:
        mask_info = f"The input image is divided into {m}x{n} regions and the borders of each region have been masked out."
    else:
        mask_info = f"The input image is divided into {m}x{n} regions."

    jigsaw_string = generate_jigsaw_string(m, n, range(1, m * n + 1))

    prompt = \
f"""{mask_info} Consider a {m}x{n} grid, where each number represents a region index ranging from 1 ({directions[0]}) to {m * n} ({directions[1]}):

```
{jigsaw_string}
```

"""

    if forward:
        prompt += f"""Some patches of equal size have been randomly swapped from their original positions, resulting in an unnatural appearance. Your task is to find the original location of the patch that currently belongs in region {k}."""
    else:
        raise NotImplementedError

    return prompt


def generate_instruction(question, m=None, n=None, answer_type="grid", think=True, instruct_model=True):
    if think:
        think_instruction = "First, output the thinking process within <think> </think> tags. Then, provide the final answer within <answer> </answer> tags. "
    else:
        think_instruction = "Directly output the final answer. "

    if answer_type in ["math", "direct"]:
        format_instruction = ""
    elif answer_type == "grid":
        format_instruction = f"The final answer should be the position indexes arranged in a {m}x{n} grid."
    elif answer_type == "digit":
        format_instruction = "The final answer should be a single number."
    elif answer_type == "letter":
        format_instruction = "The final answer should be a single letter."
    elif answer_type in ["box", "point"]:
        format_instruction = "The final answer should be bounding box coordinates, formatted as integers and separated by comma."
    else:
        raise NotImplementedError

    if instruct_model:
        prompt = question + "\n\n" + think_instruction + format_instruction
    else:
        background_instruction = "A conversation between User and Assistant. The user asks a question about the image, and the Assistant solves it.\n\nUser: "
        prompt = background_instruction + question + "\n\n" + think_instruction + format_instruction + "\n\nAssistant: "

        if think:
            prompt += "<think>"

    return prompt
