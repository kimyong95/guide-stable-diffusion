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
from google.api_core.exceptions import ServerError, TooManyRequests

class RewardBase:
    def __init__(self):
        pass

    def __call__(self, images: List[Image.Image], **kwargs):
        return
    
    def to(self, device):
        self.device = device
        return self


from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)
class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load("related_works/d3po/d3po_pytorch/assets/sac+logos+ava1-l14-linearMSE.pth")
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
class Aesthetic(RewardBase):
    def __init__(self):
        super().__init__()
        self.aesthetic_scorer = AestheticScorer(dtype=torch.float32)
    
    def to(self, device):
        self.device = device
        self.aesthetic_scorer.to(device)
        return self

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], **kwargs):
        rewards = self.aesthetic_scorer(images)
        rewards = rewards / 10
        return rewards, [""] * len(images)

class Compressibility(RewardBase):
    def __init__(self):
        super().__init__()

    # higher is better
    def __call__(self, images: List[Image.Image], **kwargs):

        MAX_SIZE = 1e6
        
        sizes = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            sizes.append(buffer.tell() / MAX_SIZE)
        sizes = torch.as_tensor(sizes, device=self.device)

        scores = -sizes
        dummy_responses = [""] * len(images)

        return scores, dummy_responses

class InCompressibility(Compressibility):
    
    # higher is better
    def __call__(self, images: List[Image.Image], **kwargs):
        scores, dummy_responses = super().__call__(images, **kwargs)
        scores = scores * -1
        return scores, dummy_responses

class GeminiQuestion(RewardBase):
    def __init__(self):
        self.target_embedding_cache = {}
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")

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
    @retry(times=10, failed_return=None, exceptions=(ServerError, TooManyRequests, ValueError))
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

    # higher is better
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