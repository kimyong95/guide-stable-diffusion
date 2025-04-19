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
    def __call__(self, images: List[Image.Image], target_prompt, query_prompt, max_reward=5.0):

        if not query_prompt:
            question_query = inspect.cleandoc(f"""
                Does the prompt '{target_prompt}' accurately describe the image? Rate from 1 (inaccurate) to 5 (accurate).
                Answer in the format: Score=score, Reason=reason.
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
        outputs = loop.run_until_complete(self.get_scores_async(contents))
        # Eg outputs = [("Score=5, Reason=xxx.", "5"), ...]

        texts = []
        scores = []
        for text, score in outputs:
            texts.append(text)
            scores.append(score)
        scores = torch.as_tensor(scores, device=self.device) / max_reward

        return scores, texts

class LlamaQuestion(RewardBase):
    
    def __init__(self):
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def to(self,device):
        super().to(device)
        self.model = self.model.to(device)
        return self

    # higher is better
    def __call__(self, images: List[Image.Image], target_prompt, query_prompt, max_reward=5.0):

        if not query_prompt:
            question_query = inspect.cleandoc(f"""
                Does the prompt '{target_prompt}' accurately describe the image? Rate from 1 (inaccurate) to 5 (accurate).
                Answer in the format: Score=score, Reason=reason.
            """)
        else:
            question_query = query_prompt.format(target_prompt=target_prompt)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question_query}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        scores = []
        output_texts = []

        for image in images:
            input_tokens = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
            input_len = input_tokens.input_ids.shape[1]

            RETRY_TIMES = 10
            for i in range(RETRY_TIMES):
                output_tokens = self.model.generate(**input_tokens, max_new_tokens=200)
                output_text = self.processor.decode(output_tokens[0][input_len:])
                
                match = re.search(r"Score=(\d+)", output_text)
                if match is None:
                    print(f"Invalid output text: {output_text}, retry {i+1}/{RETRY_TIMES}")
                else:
                    break

            score = int(match.group(1))
            scores.append(score)
            output_texts.append(output_text)
        
        scores = torch.as_tensor(scores, device=self.device) / max_reward

        return scores, output_texts


class GemmaQuestion(RewardBase):
    def __init__(self):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        model_id = "google/gemma-3-12b-it"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def to(self,device):
        super().to(device)
        self.model = self.model.to(device)
        return self

    # higher is better
    def __call__(self, images: List[Image.Image], target_prompt, query_prompt, max_reward=5.0):

        if not query_prompt:
            question_query = inspect.cleandoc(f"""
                Does the prompt '{target_prompt}' accurately describe the image? Rate from 1 (inaccurate) to 5 (accurate).
                Answer in the format: Score=score, Reason=reason.
            """)
        else:
            question_query = query_prompt.format(target_prompt=target_prompt)

        scores = []
        output_texts = []

        for image in images:

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_query}
                ]}
            ]

            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            
            RETRY_TIMES = 10
            
            for i in range(RETRY_TIMES):
                do_sample = ( i != 0 )
                with torch.inference_mode():
                    generation = self.model.generate(**inputs, max_new_tokens=1000, do_sample=do_sample)
                    generation = generation[0][input_len:]
                output_text = self.processor.decode(generation, skip_special_tokens=True).replace("\n", "")
                match = re.search(r"Score=(\d+)", output_text)
                if match is None:
                    print(f"Invalid output text: {output_text}, retry {i+1}/{RETRY_TIMES}")
                else:
                    break

            score = int(match.group(1))
            scores.append(score)
            output_texts.append(output_text)
        
        scores = torch.as_tensor(scores, device=self.device) / max_reward

        return scores, output_texts