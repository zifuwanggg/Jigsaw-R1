import torch
from typing import Any, Union

from modules.qwenvl_utils import process_vision_info
from models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


class QwenVLModule():
    def __init__(self):
        super().__init__()

    def get_model_class(self, model_id):
        if "Qwen2.5-VL" in model_id or "qwen25" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif "Qwen2-VL" in model_id or "qwen2" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls

    def get_processing_class(self, model_id):
        if "Qwen2.5-VL" in model_id or "qwen25" in model_id:
            processing_cls = Qwen2_5_VLProcessor
        elif "Qwen2-VL" in model_id or "qwen2" in model_id:
            processing_cls = Qwen2VLProcessor
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return processing_cls

    def pre_model_init(self, model_init_kwargs):
        pass

    def post_model_init(self, model, processor):
        pass

    def get_vision_modules_keywords(self):
        return ['visual']

    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []

    def is_embeds_input(self):
        return False

    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]

    def prepare_text_inputs(self, processor, inputs: dict[str, Union[torch.Tensor, Any]]):
        text_inputs = [
            processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in inputs
        ]

        return text_inputs

    def prepare_image_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]):
        image_inputs, video_inputs = process_vision_info(inputs)
        return image_inputs

    def prepare_prompt_inputs(
        self,
        processor,
        text_inputs,
        image_inputs,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False
    ):
        if len(image_inputs) > 0:
            prompt_inputs = processor(
                text=text_inputs,
                images=image_inputs,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )
        else:
            prompt_inputs = processor(
                text=text_inputs,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )

        return prompt_inputs
