from PIL import Image
import torch
import einops
import io
import requests
from openai import OpenAI
import base64
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio
import inspect
import re
from typing import List
from utils.utils import retry
from google.api_core.exceptions import InternalServerError

class RewardBase:
    def __init__(self):
        pass

    def __call__(self, images: Image.Image, target_prompt: str, query_prompt: str):
        return
    
    def to(self, device):
        self.device = device
        return self

class GeminiQuestion(RewardBase):
    def __init__(self):
        self.target_embedding_cache = {}
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_target_embedding(self, target_prompt):
        if target_prompt in self.target_embedding_cache:
            return self.target_embedding_cache[target_prompt]
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=target_prompt,
        )["embedding"]

        embedding = torch.as_tensor(embedding)
                
        self.target_embedding_cache[target_prompt] = embedding
        return embedding

    @staticmethod
    @retry(times=10, failed_return=None, exceptions=(InternalServerError,ValueError))
    async def generate_content_async(contents_list):
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        generation_config = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=100)
        tasks = [model.generate_content_async(contents, safety_settings=safety_settings, generation_config=generation_config, request_options={"timeout": 300}) for contents in contents_list]
        responses = await asyncio.gather(*tasks)
        texts = [r.text.strip() for r in responses]
        return texts

    def __call__(self, images: List[Image.Image], target_prompt, query_prompt, max_reward=5.0):

        if not query_prompt:
            question_query = inspect.cleandoc(f"""
                Does the prompt '{target_prompt}' accurately describe the image? Rate from 1 (inaccurate) to 5 (accurate).
                Answer in the format: Score=(score), Reason=(reason).
            """)
        else:
            question_query = query_prompt.format(target_prompt=target_prompt)

        if asyncio.get_event_loop_policy()._local._loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            loop = asyncio.get_event_loop()
        
        contents = []
        for image in images:
            if isinstance(image, Image.Image):
                contents.append([image, question_query])
            elif isinstance(image, List):
                contents.append(image + [question_query])
            else:
                raise ValueError(f"Invalid image type: {type(image)}")
        responses = loop.run_until_complete(self.generate_content_async(contents))
        # Eg responses = ["Score=5, Reason=xxx.", ...]

        if responses is None:
            responses = [""] * len(images)

        scores = []
        for response in responses:
            match = re.search(r"Score=(\d+)", response)
            score = int(match.group(1)) if match else 0
            scores.append(score)
        scores = torch.as_tensor(scores, device=self.device) / max_reward

        return scores, responses