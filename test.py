import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict

from modules.qwenvl_module import QwenVLModule
from modules.internvl_module import InternVLModule
from utils.reward import extract_answer, accuracy_reward_helper
from utils.utils import create_input_helper, denormalize_box


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--max_pixels", default=12845056, type=int)
    parser.add_argument("--min_pixels", default=3136, type=int)
    parser.add_argument("--max_anyres_num", default=6, type=int)
    parser.add_argument("--think", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--instruct_model", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_name", default="coco", type=str)
    parser.add_argument("--dataset_split", default="test", type=str)
    parser.add_argument("--dataset_size", default=-1, type=int)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--print_result", default=True, action=argparse.BooleanOptionalAction)

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


def test_jigsaw(input, image_input, output, metrics, instruct_model):
    if "image_id" in input:
        image_id = input["image_id"]
    else:
        image_id = 0

    if instruct_model:
        question = input["prompt"][0]["content"][-1]["text"]
    else:
        question = input["prompt"][-1]["text"]

    solution = input["solution"]
    m = input["m"]
    n = input["n"]
    answer_type = input["answer_type"]
    think = input["think"]

    if answer_type in ["box"]:
        width, height = image_input.size
        solution = denormalize_box(solution, width, height)

    length = len(output)
    output_extracted = extract_answer(output, m, n, answer_type, think)
    score, correct = accuracy_reward_helper(output_extracted, solution, answer_type)

    metrics["length"] += length
    metrics["score"] += score
    metrics["correct"] += correct

    result = {
        "image_id": image_id,
        "question": question,
        "solution": solution,
        "output": output,
        "output_extracted": output_extracted,
        "length": length,
        "score": score,
        "correct": correct
    }

    return result


def test_outputs(input, image_input, output, metrics, counter, instruct_model):
    if "image_id" in input:
        image_id = input["image_id"]
    else:
        image_id = 0

    if instruct_model:
        question = input["prompt"][0]["content"][-1]["text"]
    else:
        question = input["prompt"][-1]["text"]

    solution = input["solution"]
    answer_type = input["answer_type"]
    think = input["think"]

    if answer_type in ["box", "point"]:
        width, height = image_input.size
        solution = denormalize_box(solution, width, height)

    length = len(output)
    output_extracted = extract_answer(output, answer_type=answer_type, think=think)
    score, correct = accuracy_reward_helper(output_extracted, solution, answer_type)

    metrics["length"] += length
    metrics["score"] += score
    metrics["correct"] += correct

    if "task" in input:
        task = input["task"]
        counter[task] += 1
        metrics[task] += correct

    result = {
        "image_id": image_id,
        "question": question,
        "solution": solution,
        "output": output,
        "output_extracted": output_extracted,
        "length": length,
        "score": score,
        "correct": correct
    }

    return result


if __name__ == "__main__":
    args = parse_arguments()
    args.model_name = "_".join(args.model_path.split("/")[-2:])
    print(args)

    if "instruct" in args.model_path.lower():
        assert args.instruct_model == True

    output_dir = f"{args.output_dir}/{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    if "qwen" in args.model_path.lower():
        mllm_module = QwenVLModule()
    elif "intern" in args.model_path.lower():
        mllm_module = InternVLModule()
    else:
        raise NotImplementedError(args.model_path)

    model_init_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    mllm_module.pre_model_init(model_init_kwargs)
    model_cls = mllm_module.get_model_class(args.model_path)
    model = model_cls.from_pretrained(args.model_path, **model_init_kwargs)
    model.eval()

    processing_cls = mllm_module.get_processing_class(args.model_path)
    processor = processing_cls.from_pretrained(
        args.model_path,
        trust_remote_code=model_init_kwargs.get("trust_remote_code", None)
    )

    for component, processing_keyword in mllm_module.get_custom_processing_keywords():
        if processing_keyword in args:
            processing_component = getattr(processor, component, processor)
            setattr(processing_component, processing_keyword, getattr(args, processing_keyword))

    mllm_module.post_model_init(model, processor)

    inputs = create_input_helper(**vars(args))

    results = []
    metrics = defaultdict(float)
    counter = defaultdict(int)

    for i in tqdm(range(0, len(inputs), args.batch_size)):
        batch_inputs = inputs[i : i + args.batch_size]
        prompts = [x["prompt"] for x in batch_inputs]

        text_inputs = mllm_module.prepare_text_inputs(processor, prompts)
        image_inputs = mllm_module.prepare_image_inputs(prompts)
        prompt_inputs = mllm_module.prepare_prompt_inputs(
            processor,
            text_inputs,
            image_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = prompt_inputs.to("cuda")

        with torch.no_grad():
            if isinstance(mllm_module, QwenVLModule):
                generated_ids = model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in mllm_module.get_non_generate_params()},
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(prompt_inputs.input_ids, generated_ids)]
            elif isinstance(mllm_module, InternVLModule):
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].to(torch.bfloat16)
                generated_ids = model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in mllm_module.get_non_generate_params()},
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.eos_token_id
                )

        batch_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for input, image_input, output in zip(batch_inputs, image_inputs, batch_outputs):
            if args.jigsaw:
                result = test_jigsaw(input, image_input, output, metrics, args.instruct_model)
            else:
                result = test_outputs(input, image_input, output, metrics, counter, args.instruct_model)

            results.append(result)

            if args.print_result:
                print(result)

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

    if args.jigsaw:
        output_path = f"{output_dir}/{args.dataset_split}_{args.model_name}_{args.think}_s{args.jigsaw_seed}_m{args.m}n{args.n}nc{args.n_c}_{args.forward}_{args.question_type}.json"
    else:
        output_path = f"{output_dir}/{args.dataset_split}_{args.model_name}_{args.think}.json"

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)
