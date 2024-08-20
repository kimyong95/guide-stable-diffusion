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


class LLaVA(RewardBase):
    def __init__(self):
        self.url = "http://localhost:8000/reward"

    def __call__(self, images: Image.Image, target_prompt, query_prompt):
        buffers = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffers.append(buffer)
        files = [("files", ( buf.getvalue() )) for buf in buffers]
        data = {"target_prompt": target_prompt, "query_prompt": query_prompt}
        response = requests.post(self.url, files=files, data=data)
        response.raise_for_status()
        recalls = torch.as_tensor(response.json()["recalls"], device=self.device)
        texts = response.json()["texts"]

        return recalls, texts
class Gpt(RewardBase):
    def __init__(self):
        self.client = OpenAI()
        self.target_embedding_cache = {}
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_message(self, image_base64, query_prompt):
        return {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query_prompt,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "low",
                }
                }
            ]
        }

    def get_target_embedding(self, target_prompt):
        if target_prompt in self.target_embedding_cache:
            return self.target_embedding_cache[target_prompt]
        response = self.client.embeddings.create(input = [target_prompt], model="text-embedding-3-small")
        embedding = torch.as_tensor(response.data[0].embedding)
        self.target_embedding_cache[target_prompt] = embedding
        return embedding

    def __call__(self, images: Image.Image, target_prompt, query_prompt):
        buffers = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffers.append(buffer)
        base64s = [base64.b64encode(buf.getvalue()).decode('utf-8') for buf in buffers]
        
        responses = []
        for base64_ in base64s:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[self.get_message(base64_, query_prompt)],
                max_tokens=100,
            )
            responses.append(response.choices[0].message.content.strip())

        embeddings = self.client.embeddings.create(input = responses, model="text-embedding-3-small")
        embeddings = [data.embedding for data in embeddings.data]
        embeddings = torch.as_tensor(embeddings)

        similarity = self.cos_sim(embeddings, self.get_target_embedding(target_prompt)[None,:]).to(self.device)

        return similarity, responses
    
class Gemini(RewardBase):
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
    async def generate_content_async(contents_list):
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        generation_config = genai.types.GenerationConfig(temperature=0.0)
        tasks = [model.generate_content_async(contents, safety_settings=safety_settings, generation_config=generation_config) for contents in contents_list]
        responses = await asyncio.gather(*tasks)
        texts = [r.text.strip() for r in responses]
        return texts

    def __call__(self, images: Image.Image, target_prompt, query_prompt):
        
        contents_list = [[image, query_prompt] for image in images]

        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(self.generate_content_async(contents_list))

        embeddings = genai.embed_content(
            model="models/text-embedding-004",
            content=responses,
        )["embedding"]

        embeddings = torch.as_tensor(embeddings)

        similarity = self.cos_sim(embeddings, self.get_target_embedding(target_prompt)[None,:]).to(self.device)

        return similarity, responses
    
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
    @retry(times=3, exceptions=(InternalServerError,))
    async def generate_content_async(contents_list):
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        generation_config = genai.types.GenerationConfig(temperature=0.0)
        tasks = [model.generate_content_async(contents, safety_settings=safety_settings, generation_config=generation_config) for contents in contents_list]
        responses = await asyncio.gather(*tasks)
        texts = [r.text.strip() for r in responses]
        return texts

    def __call__(self, images: List[Image.Image], target_prompt, query_prompt):
        
        question_query = f"""Does the prompt '{target_prompt}' accurately describe the image? Rate from 1 (inaccurate) to 5 (accurate).
        Answer in the format: Score=(score), Reason=(reason).
        """

        loop = asyncio.get_event_loop()
        
        contents = [[image, question_query] for image in images]
        responses = loop.run_until_complete(self.generate_content_async(contents))
        # Eg responses = ["Score=5, Reason=xxx.", ...]
    
        scores = []
        for response in responses:
            match = re.search(r"Score=(\d+)", response)
            score = int(match.group(1)) if match else 0
            scores.append(score)
        scores = torch.as_tensor(scores, device=self.device) / 5.0

        return scores, responses