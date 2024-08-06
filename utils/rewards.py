from PIL import Image
import torch
import einops
import io
import requests
from openai import OpenAI
import base64

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
        self.client = OpenAI(api_key="sk-proj-KYh7prhCAul7nm8BQ0m6T3BlbkFJbzQOpJrTALeqTr3uLwkT")
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
            responses.append(response.choices[0].message.content)

        embeddings = self.client.embeddings.create(input = responses, model="text-embedding-3-small")
        embeddings = [data.embedding for data in embeddings.data]
        embeddings = torch.as_tensor(embeddings)

        similarity = self.cos_sim(embeddings, self.get_target_embedding(target_prompt)[None,:]).to(self.device)

        return similarity, responses