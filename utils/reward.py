import os
import re
from datetime import datetime
from math_verify import parse, verify

from utils.utils import denormalize_box


def process_jigsaw_string(m, n, jigsaw_string):
    for char in ["`", ",", ";", "\n"]:
        jigsaw_string = jigsaw_string.replace(char, " ")

    jigsaw_string = re.sub(r"\s{2,}", " ", jigsaw_string).strip()
    patches = jigsaw_string.split(" ")
    patches_pred = []

    for patch in patches:
        if re.fullmatch(r"[0-9]+", patch):
            patch = int(patch)
            if 1 <= patch <= m * n:
                patches_pred.append(patch)
            else:
                return
        else:
            return

    return patches_pred


def extract_answer(output, m=None, n=None, answer_type="grid", think=True):
    if think:
        answer_tag_pattern = r"<answer>(.*?)</answer>"
        answer_tag_match = re.search(answer_tag_pattern, output, re.DOTALL)

        if answer_tag_match:
            answer_content = answer_tag_match.group(1).strip()
        else:
            answer_content = output.strip()
    else:
        answer_content = output.strip()

    if answer_type in ["math", "direct", "grid"]:
        answer_pattern = r"(.*)"
    elif answer_type == "digit":
        answer_pattern = r"(\d+)"
    elif answer_type == "letter":
        answer_pattern = r"([A-Z])"
        answer_content = answer_content[::-1]
    elif answer_type in ["box", "point"]:
        answer_pattern = r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)"
    else:
        raise NotImplementedError

    answer_match = re.search(answer_pattern, answer_content, re.DOTALL)

    if answer_match:
        if answer_type == "grid":
            extracted_output = process_jigsaw_string(m, n, answer_match.group(1))
        elif answer_type in ["box", "point"]:
            extracted_output =  [
                float(answer_match.group(1)),
                float(answer_match.group(2)),
                float(answer_match.group(3)),
                float(answer_match.group(4))
            ]
        else:
            extracted_output = answer_match.group(1)
    else:
        extracted_output = answer_content

    return extracted_output


def score_boxes(box_pred, box_gt):
    if not isinstance(box_pred, list):
        return 0

    x1_min, y1_min, x1_max, y1_max = box_pred
    x2_min, y2_min, x2_max, y2_max = box_gt

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max - 1, x2_max - 1)
    inter_y_max = min(y1_max - 1, y2_max - 1)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        inter = (inter_x_max - inter_x_min + 1) * (inter_y_max - inter_y_min + 1)
    else:
        inter = 0

    size1 = (x1_max - x1_min) * (y1_max - y1_min)
    size2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = size1 + size2 - inter

    return float(inter) / union


def score_points(box_pred, box_gt):
    if not isinstance(box_pred, list):
        return 0.

    x = (box_pred[0] + box_pred[2]) / 2
    y = (box_pred[1] + box_pred[3]) / 2
    x_min, y_min, x_max, y_max = box_gt

    if (x_min <= x <= x_max) and \
        (y_min <= y <= y_max):
            return 1.
    else:
        return 0.


def verify_math(pred, gt):
    pred = parse(str(pred))
    gt = parse(str(gt))
    correct = float(verify(gt, pred))

    return correct


def score_jigsaw_patches(patches_pred, patches_gt):
    if patches_pred is None or len(patches_pred) != len(patches_gt):
        return 0

    score = 0
    for p_pred, p_gt in zip(patches_pred, patches_gt):
        if p_pred == p_gt:
            score += 1

    score /= len(patches_gt)

    return score


def score_jigsaw_connections(patches_pred, patches_gt, m, n, forward=True):
    if patches_pred is None or len(patches_pred) != len(patches_gt):
        return 0

    # Create inverse mapping to find the position of each patch in the predicted and ground truth
    pred_positions = {}
    gt_positions = {}

    if forward:
        # In forward format:
        # patches_pred[pos] = patch_id, meaning position 'pos' contains the patch with ID 'patch_id'
        # We create a mapping: patch_id -> position
        for pos, patch_id in enumerate(patches_pred):
            pred_positions[patch_id] = pos

        for pos, patch_id in enumerate(patches_gt):
            gt_positions[patch_id] = pos
    else:
        # In backward format:
        # patches_gt[patch_id-1] = pos, meaning the patch with ID 'patch_id' is at position 'pos'
        # We create a mapping: patch_id -> position
        for patch_id, pos in enumerate(patches_gt):
            gt_positions[patch_id + 1] = pos - 1  # Adjust for 1-indexing

        for patch_id, pos in enumerate(patches_pred):
            pred_positions[patch_id + 1] = pos - 1  # Adjust for 1-indexing

    # Count correct connections
    correct_connections = 0
    total_internal_connections = 2 * m * n - m - n  # Horizontal + vertical internal connections

    # Check horizontal connections
    for i in range(m):
        for j in range(n-1):
            pos1 = i * n + j
            pos2 = i * n + j + 1

            # Get the patch numbers at these positions
            patch1_pred = patches_pred[pos1]
            patch2_pred = patches_pred[pos2]

            # Find positions of these patches in the ground truth
            gt_pos1 = gt_positions[patch1_pred]
            gt_pos2 = gt_positions[patch2_pred]

            # Check if these patches are also horizontally adjacent in the ground truth
            if (gt_pos1 // n == gt_pos2 // n) and (gt_pos2 % n - gt_pos1 % n == 1):
                correct_connections += 1

    # Check vertical connections
    for i in range(m-1):
        for j in range(n):
            pos1 = i * n + j
            pos2 = (i+1) * n + j

            # Get the patch numbers at these positions
            patch1_pred = patches_pred[pos1]
            patch2_pred = patches_pred[pos2]

            # Find positions of these patches in the ground truth
            gt_pos1 = gt_positions[patch1_pred]
            gt_pos2 = gt_positions[patch2_pred]

            # Check if these patches are also vertically adjacent in the ground truth
            if (gt_pos1 % n == gt_pos2 % n) and (gt_pos2 // n - gt_pos1 // n == 1):
                correct_connections += 1

    # Calculate the final score
    if total_internal_connections > 0:
        score = correct_connections / total_internal_connections
    else:
        score = 0

    return score


def accuracy_reward_helper(
    pred,
    gt,
    answer_type,
    use_jigsaw_connections=False,
    m=None,
    n=None,
    forward=True
):
    if answer_type == "grid":
        if use_jigsaw_connections:
            score = score_jigsaw_connections(patches_pred=pred, patches_gt=gt, m=m, n=n, forward=forward)
        else:
            score = score_jigsaw_patches(patches_pred=pred, patches_gt=gt)
        correct = float(pred == gt)
    elif answer_type in ["math", "digit"]:
        score = 0.
        correct = float(verify_math(pred=pred, gt=gt))
    elif answer_type in ["direct", "letter"]:
        score = 0.
        correct = float(str(pred).lower() == str(gt).lower())
    elif answer_type == "box":
        score = score_boxes(box_pred=pred, box_gt=gt)
        correct = float(score > 0.5)
    elif answer_type == "point":
        score = score_boxes(box_pred=pred, box_gt=gt)
        correct = score_points(box_pred=pred, box_gt=gt)
    else:
        raise NotImplementedError(f"answer_type: {answer_type}")

    return score, correct


def accuracy_reward(completions, **args):
    current_time = datetime.now().strftime("%m-%d %H:%M:%S:%f")

    thinks = args.get("think")
    if isinstance(completions[0], str):
        outputs = []
        for i in range(len(completions)):
            think = thinks[i]
            completion = completions[i]
            if think:
                outputs.append("<think>" + completion)
            else:
                outputs.append(completion)
    else:
        outputs = [completion[0]["content"] for completion in completions]

    prompts = args.get("prompt")
    solutions = args.get("solution")
    ms = args.get("m", [None] * len(outputs))
    ns = args.get("n", [None] * len(outputs))
    forwards = args.get("forward", [True] * len(outputs))
    answer_types = args.get("answer_type")
    widths = args.get("width")
    heights = args.get("height")
    image_ids = args.get("image_id", [None] * len(outputs))
    use_jigsaw_connections = args.get("score_jigsaw_connections")

    rewards = []
    for output, prompt, solution, m, n, forward, answer_type, think, width, height, image_id in zip(
        outputs,
        prompts,
        solutions,
        ms,
        ns,
        forwards,
        answer_types,
        thinks,
        widths,
        heights,
        image_ids
    ):
        if answer_type in ["box", "point"]:
            solution = denormalize_box(solution, width, height)

        output_extracted = extract_answer(output, m, n, answer_type, think)
        score, correct = accuracy_reward_helper(
            output_extracted,
            solution,
            answer_type,
            use_jigsaw_connections,
            m,
            n,
            forward
        )

        if answer_type in ["grid", "box", "point"]:
            reward = score
        else:
            reward = correct

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")

            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} -------------\n")
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Output: {output}\n")
                f.write(f"Solution: {solution}\n")
                f.write(f"Accuracy Reward: {reward}\n")

    return rewards


def format_reward(completions, **args):
    current_time = datetime.now().strftime("%m-%d %H:%M:%S:%f")

    thinks = args.get("think")
    if isinstance(completions[0], str):
        outputs = []
        for i in range(len(completions)):
            think = thinks[i]
            completion = completions[i]
            if think:
                outputs.append("<think>" + completion)
            else:
                outputs.append(completion)
    else:
        outputs = [completion[0]["content"] for completion in completions]

    ms = args.get("m", [None] * len(outputs))
    ns = args.get("n", [None] * len(outputs))
    answer_types = args.get("answer_type")
    image_ids = args.get("image_id", [None] * len(outputs))

    rewards = []
    for output, m, n, answer_type, think, image_id in zip(
        outputs,
        ms,
        ns,
        answer_types,
        thinks,
        image_ids
    ):
        reward = 0.

        if think:
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
            match = re.search(pattern, output, re.DOTALL)
            if not match or match.group(0) != output:
                rewards.append(0.)
                continue

        if not think:
            if answer_type == "digit":
                if output.isdigit():
                    rewards.append(1.)
                    continue
                else:
                    rewards.append(0.)
                    continue
            elif answer_type == "letter":
                if len(output) == 1 and output[0].isupper():
                    rewards.append(1.)
                    continue
                else:
                    rewards.append(0.)
                    continue

        output_extracted = extract_answer(output, m, n, answer_type, think)

        if answer_type == "math":
            reward = 1.0
        elif answer_type == "grid":
            if isinstance(output_extracted, list) and len(output_extracted) == m * n:
                reward = 1.0
        elif think and answer_type == "digit":
            if output_extracted.isdigit():
                reward = 1.0
        elif think and answer_type == "letter":
            if len(output_extracted) == 1 and output_extracted[0].isupper():
                reward = 1.0
        elif answer_type in ["box", "point"]:
            if isinstance(output_extracted, list) and len(output_extracted) == 4:
                reward = 1.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")

            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} -------------\n")
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Output: {output}\n")
                f.write(f"Format Reward: {reward}\n")

    return rewards
