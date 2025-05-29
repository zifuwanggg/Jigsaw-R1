# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Optional
from dataclasses import dataclass, field

from trainer.grpo_config import GRPOConfig
from trainer.grpo_trainer import MLLMGRPOTrainer
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from modules.qwenvl_module import QwenVLModule
from modules.internvl_module import InternVLModule
from utils.utils import create_input_helper
from utils.jigsaw_dataset import JigsawDataset
from utils.reward import accuracy_reward, format_reward


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


@dataclass
class GRPOScriptArguments(ScriptArguments):
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    max_anyres_num: Optional[int] = field(default=6)
    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format"])
    reward_weights: list[float] = field(default_factory=lambda: [1., 0.5])
    reward_reduction: str = field(default="sum")
    load_parquet: bool = field(default=False)
    data_dir: str = field(default="")
    dataset_name: str = field(default="coco")
    data_splits: list[str] = field(default_factory=lambda: ["train2014"])
    dataset_size: int = field(default=-1)
    jigsaw: bool = field(default=True)
    jigsaw_seed: int = field(default=0)
    score_jigsaw_connections: bool = field(default=False)
    stage: list[int] = field(default_factory=lambda: [6400, 99999999])
    m: list[int] = field(default_factory=lambda: [2, 3])
    n: list[int] = field(default_factory=lambda: [1, 1])
    n_c: list[int] = field(default_factory=lambda: [4, 4])
    shuffle_mn: bool = field(default=True)
    mask_ratio: float = field(default=0)
    forward: bool = field(default=True)
    question_type: str = field(default="full")
    width_min: float = field(default=0.1)
    width_max: float = field(default=0.6)
    height_min: float = field(default=0.1)
    height_max: float = field(default=0.6)
    max_attempts: int = field(default=10)
    think: bool = field(default=True)
    instruct_model: bool = field(default=False)


def main(script_args, training_args, model_args):
    if "instruct" in model_args.model_name_or_path.lower():
        assert script_args.instruct_model == True

    trainer_cls = MLLMGRPOTrainer

    if "qwen" in model_args.model_name_or_path.lower():
        mllm_module = QwenVLModule()
    elif "intern" in model_args.model_name_or_path.lower():
        mllm_module = InternVLModule()
    else:
        raise NotImplementedError

    assert len(script_args.reward_funcs) == len(script_args.reward_weights)

    reward_funcs_registry = {
        "accuracy": accuracy_reward,
        "format": format_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.load_parquet:
        script_args.dataset_split = script_args.dataset_train_split
        script_args.m = script_args.m[0]
        script_args.n = script_args.n[0]
        script_args.n_c = script_args.n_c[0]
        jigsaw_dataset = create_input_helper(**vars(script_args))
    else:
        image_paths = []
        for data_split in script_args.data_splits:
            json_path = f"{script_args.data_dir}/{script_args.dataset_name}/{data_split}.json"
            with open(json_path, "r") as f:
                data_list = json.load(f)

            for image_id in data_list:
                image_path = f"{script_args.data_dir}/{script_args.dataset_name}/{image_id}"
                image_paths.append(image_path)

        jigsaw_dataset = JigsawDataset(
            image_paths,
            script_args.stage,
            script_args.m,
            script_args.n,
            script_args.n_c,
            script_args.shuffle_mn,
            script_args.mask_ratio,
            script_args.width_min,
            script_args.width_max,
            script_args.height_min,
            script_args.height_max,
            script_args.forward,
            script_args.question_type,
            script_args.think,
            script_args.instruct_model
        )

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        reward_weights=script_args.reward_weights,
        args=training_args,
        mllm_module=mllm_module,
        train_dataset=jigsaw_dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
        reward_reduction=script_args.reward_reduction,
        score_jigsaw_connections=script_args.score_jigsaw_connections
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    print(training_args)
    print(model_args)
    main(script_args, training_args, model_args)
