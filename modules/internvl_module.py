import torch
from typing import Any, Union
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.feature_extraction_sequence_utils import BatchFeature

from modules.internvl_utils import (
    load_image,
    extract_system_message,
    process_conversation_list,
    process_internvl_vision_info,
)


IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'


class InternVLModule():
    def __init__(self):
        super().__init__()
        self.conv_template = None
        self.num_image_token = None

    def get_model_class(self, model_id):
        assert "intern" in model_id.lower(), f"model_id must contain 'intern', but got {model_id}"

        self.model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # The model class of InternVL when being mapped has been determined by its config
        model_cls = AutoModel

        return model_cls

    def get_processing_class(self, model_id):
        return AutoTokenizer

    def pre_model_init(self, model_init_kwargs):
         # InternVL should be inputted with "trust_remote_code=True"
        model_init_kwargs["trust_remote_code"] = True

        # "use_cache" should be removed
        model_init_kwargs.pop("use_cache", None)

        # "flash_attention_2" should be modified to "use_flash_attn" in InternVL
        if "flash_attention_2" in model_init_kwargs.get("attn_implementation", ""):
            model_init_kwargs["use_flash_attn"] = True
            model_init_kwargs.pop("attn_implementation")

    def post_model_init(self, model, processor):
        self.conv_template = model.conv_template if self.conv_template is None else self.conv_template
        self.num_image_token = model.num_image_token if self.num_image_token is None else self.num_image_token
        img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id

    def get_eos_token_id(self, processor):
        eos_token_id = processor.convert_tokens_to_ids(self.conv_template.sep.strip())
        return eos_token_id

    def get_vision_modules_keywords(self):
        return ['vision_model']

    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_flags']

    def get_non_generate_params(self):
        return ['image_flags']

    def get_custom_processing_keywords(self):
        return [('None', 'max_anyres_num')]

    def is_embeds_input(self):
        return True

    def prepare_text_inputs(self, processor, inputs: dict[str, Union[torch.Tensor, Any]]):
        text_inputs = []
        for conversation_list in inputs:
            template = self.conv_template.copy()
            system_message = extract_system_message(conversation_list)

            if system_message is not None:
                template.system_message = system_message

            processed_list = process_conversation_list(conversation_list, system_message)
            for i, processed_item in enumerate(processed_list):
                if i % 2 == 0:
                    template.append_message(template.roles[0], processed_item)
                else:
                    template.append_message(template.roles[1], processed_item)

            if len(processed_list) % 2 == 1:
                template.append_message(template.roles[1], None)

            query = template.get_prompt()
            text_inputs.append(query)

        return text_inputs

    def prepare_image_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]):
        image_inputs = process_internvl_vision_info(inputs)
        return image_inputs

    def prepare_prompt_inputs(
        self,
        processor,
        text_inputs,
        images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False
    ):
        # Process images
        full_pixel_values = []
        num_patches_list = []
        for img in images:
            pixel_values = load_image(
                img,
                input_size=self.model_config.vision_config.image_size,
                max_num=processor.max_anyres_num
            )
            full_pixel_values.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        full_pixel_values = torch.cat(full_pixel_values, dim=0)

        # Process prompts
        queries = []
        image_idx = 0
        for query in text_inputs:
            while "<image>" in query:
                num_patches = num_patches_list[image_idx]
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)
                image_idx += 1
            queries.append(query)

        assert image_idx == len(num_patches_list)

        model_inputs = processor(
            queries,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )
        model_inputs["pixel_values"] = full_pixel_values
        # Only support pure-image data currently (each sample should contain the image)
        model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)
        model_inputs = BatchFeature(data=model_inputs)

        return model_inputs
