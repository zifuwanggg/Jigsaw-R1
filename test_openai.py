import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from test import test_jigsaw, test_outputs
from models.openai_model import OpenAIModel
from utils.utils import create_input_helper


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions")
    parser.add_argument("--batch_index", default=0, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--temperature", default=0., type=float)
    parser.add_argument("--think", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--instruct_model", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_name", default="coco", type=str)
    parser.add_argument("--dataset_split", default="test", type=str)
    parser.add_argument("--dataset_size", default=-1, type=int)
    parser.add_argument("--output_dir", default="outputs", type=str)

    parser.add_argument("--jigsaw", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--jigsaw_seed", default=0, type=int)
    parser.add_argument("--m", default=2, type=int)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--n_c", default=4, type=int)
    parser.add_argument("--shuffle_mn", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mask_ratio", default=0, type=float)
    parser.add_argument("--forward", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--question_type", default="pair", type=str)
    parser.add_argument("--width_min", default=0.1, type=float)
    parser.add_argument("--width_max", default=0.6, type=float)
    parser.add_argument("--height_min", default=0.1, type=float)
    parser.add_argument("--height_max", default=0.6, type=float)

    return parser.parse_args()


def split_inputs(inputs, batch_size):
    n = len(inputs) // batch_size + min(1, len(inputs) % batch_size)

    sub_inputs = []
    for i in range(n):
        start = i * batch_size
        end = (i + 1) * batch_size
        sub_inputs.append(inputs[start : end])

    return sub_inputs


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    output_dir = f"{args.output_dir}/{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    if args.jigsaw:
        job_name = f"{args.dataset_split}_{args.model_name}_{args.think}_s{args.jigsaw_seed}_m{args.m}n{args.n}nc{args.n_c}_{args.forward}_{args.question_type}"
    else:
        job_name = f"{args.dataset_split}_{args.model_name}_{args.think}"

    jsonl_path = f"{output_dir}/{job_name}_{args.batch_index}.jsonl"
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            outputs = [json.loads(line) for line in f]
            image_ids = [list(line.keys())[0] for line in outputs]
            image_ids = set(image_ids)
    else:
        image_ids = {}

    model = OpenAIModel(args.model_name, args.base_url, args.endpoint)

    inputs = create_input_helper(**vars(args))
    sub_inputs = split_inputs(inputs, args.batch_size)
    sub_input = sub_inputs[args.batch_index]

    if len(image_ids) < len(sub_input):
        with open(jsonl_path, "a") as f:
            for input in tqdm(sub_input):
                image_id = input["image_id"]
                if image_id not in image_ids:
                    prompt = input["prompt"]
                    prompt = model.process_image_prompt(prompt)

                    output = model.generate(prompt, args.temperature)
                    output = {f"{image_id}": output}

                    f.write(json.dumps(output) + "\n")

    results = []
    metrics = defaultdict(float)
    counter = defaultdict(int)

    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            outputs = [json.loads(line) for line in f]
            outputs = {k: v for d in outputs for k, v in d.items()}

    for input in sub_input:
        image_id = input["image_id"]
        prompt = input["prompt"]

        if prompt[0]["content"][0]["type"] == "image_url":
            prompt = model.process_base64_prompt(prompt)

        image = prompt[0]["content"][0]["image"]
        output = outputs[image_id]

        if args.jigsaw:
            result = test_jigsaw(input, image, output, metrics, args.instruct_model)
        else:
            result = test_outputs(input, image, output, metrics, counter, args.instruct_model)

        results.append(result)

    output_dict = {
        "results": results,
        "args": vars(args),
        "length": metrics["length"] / len(inputs),
        "score": metrics["score"] / len(inputs),
        "correct": metrics["correct"] / len(inputs),
        "#example": len(inputs),
    }

    if len(counter) > 0:
        for task in counter.keys():
            output_dict[task] = metrics[task] / counter[task]

    output_path = f"{output_dir}/{job_name}_{args.batch_index}.json"
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)
