# Execute:
# CUDA_VISIBLE_DEVICES=0 uvicorn llava_server.api:app --timeout-keep-alive 600 --log-level debug

from typing import List
from fastapi import FastAPI, UploadFile, File, Body, Form
from PIL import Image
from llava_server.core import LLaVA

import torch
import einops
from pydantic import BaseModel
import uvicorn

model_path="liuhaotian/llava-v1.5-7b"
assert torch.cuda.is_available()
device = torch.device("cuda")

llava = LLaVA()
llava = llava.to(device)

app = FastAPI()

class Response(BaseModel):
    recalls: List[float]
    texts: List[str]

@app.post("/reward")
async def llava_reward(
    files: List[UploadFile] = File(...),
    target_prompt: str = Form(...),
    query_prompt: str = Form(...)
):

    images = [ Image.open(file.file) for file in files ]
    
    recalls, outputs = llava(images, target_prompt, query_prompt)

    response = Response(recalls=recalls.tolist(), texts=outputs)
    
    return response
