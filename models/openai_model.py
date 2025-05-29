import os
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils import image_to_bytes, bytes_to_base64, bytes_to_image


class OpenAIModel():
    def __init__(
        self,
        model_name,
        base_url="https://api.openai.com/v1",
        endpoint="/v1/chat/completions",
        api_key=None
    ):
        super().__init__()

        self.model_name = model_name
        self.endpoint = endpoint
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url
        )


    def process_image_prompt(
        self,
        prompt,
    ):
        assert prompt[0]["content"][0]["type"] == "image"

        image = prompt[0]["content"][0]["image"]
        image_bytes = image_to_bytes(image)
        image_base64 = bytes_to_base64(image_bytes)

        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
            }
        }
        prompt[0]["content"][0] = image_content

        return prompt


    def process_base64_prompt(
        self,
        prompt,
    ):
        assert prompt[0]["content"][0]["type"] == "image_url"

        prefix = "data:image/jpeg;base64,"
        image_base64 = prompt[0]["content"][0]["image_url"]["url"]
        image_base64 = image_base64[len(prefix):]
        image = bytes_to_image(image_base64)

        image_content = {"type": "image", "image": image}
        prompt[0]["content"][0] = image_content

        return prompt


    def create_input(
        self,
        custom_id,
        prompt,
        temperature,
    ):
        if self.model_name in ["o3", "o4-mini", "o3-mini", "o1"]:
            temperature = 1

        messages = self.process_image_prompt(prompt)
        input = {
            "custom_id": str(custom_id),
            "method": "POST",
            "url": self.endpoint,
            "body": {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
            }
        }

        return input


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def generate(
        self,
        message,
        temperature,
    ):
        if self.model_name in ["o3", "o4-mini", "o3-mini", "o1"]:
            temperature = 1

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=temperature,
        )
        output = completion.choices[0].message.content

        return output
