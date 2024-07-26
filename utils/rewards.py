from PIL import Image
import torch
import einops
import io
import requests

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