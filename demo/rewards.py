from utils import retry
import inspect
import re
from typing import List
import torch
from PIL import Image
import asyncio
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import ServerError, TooManyRequests

class GeminiQuestion():
    def __init__(self):
        pass

    @staticmethod
    @retry(times=10, failed_return=None, exceptions=(ServerError, TooManyRequests, ValueError))
    async def get_one_score_async(model, contents, retry_attempt):
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        temperature = 0 if retry_attempt == 0 else 0.1
        generation_config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=500)

        response = await model.generate_content_async(contents, safety_settings=safety_settings, generation_config=generation_config, request_options={"timeout": 300})
        text = response.text.strip()
        match = re.search(r"Score=(\d+)", text)
        if match is None:
            raise ValueError(f"Invalid response: {text}")
        score = int(match.group(1))

        return (text, score)

    @staticmethod
    async def get_scores_async(contents_list):
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-lite-001")
        
        tasks = [GeminiQuestion.get_one_score_async(model, contents) for contents in contents_list]
        outputs = await asyncio.gather(*tasks)
        return outputs

    # higher is better
    def __call__(self, images: List[Image.Image], target_prompt):

        query = inspect.cleandoc(f"""
            Does the prompt '{target_prompt}' accurately describe the image, why? Rate from 1 (inaccurate) to 5 (accurate).
            Answer in the format: Reason=reason, Score=score.
        """)

        if asyncio.get_event_loop_policy()._local._loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            loop = asyncio.get_event_loop()
        
        contents = []
        for image in images:
            if isinstance(image, Image.Image):
                contents.append([image, query])
            elif isinstance(image, List):
                contents.append(image + [query])
            else:
                raise ValueError(f"Invalid image type: {type(image)}")
        outputs = loop.run_until_complete(self.get_scores_async(contents))
        # Eg outputs = [("Reason=xxx, Score=5", "5"), ...]

        texts = []
        scores = []
        for text, score in outputs:
            texts.append(text)
            scores.append(score)
        scores = torch.as_tensor(scores) / 5.0

        return scores, texts