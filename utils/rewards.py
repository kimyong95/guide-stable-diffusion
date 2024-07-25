import open_clip
from PIL import Image
import torch
import einops
import io
import requests

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, load_images
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from bert_score import BERTScorer

class RewardBase:
    def __init__(self):
        pass

    def __call__(self, images: Image.Image, target_prompt: str, query_prompt: str):
        return
    
    def to(self, device):
        self.device = device
        return self

class LLaVA_Web(RewardBase):
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

class LLaVA(RewardBase):
    def __init__(self, model_path = "liuhaotian/llava-v1.5-7b"):
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
        )

        self.device = torch.device("cpu")

        self.bert_scorer = BERTScorer("microsoft/deberta-xlarge-mnli", use_fast_tokenizer=True, device="cpu")

    def process_query(self, query="Describe this image concisely."):
        query = f"{DEFAULT_IMAGE_TOKEN}\n{query}"
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        processed_query = conv.get_prompt()
        return processed_query

    # maximize
    @torch.inference_mode()
    def __call__(self, images, target_prompt, query_prompt):

        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(self.process_query(query_prompt), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.device)

        input_ids = einops.repeat(input_ids, "S -> B S", B=len(images))

        output_ids = self.model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs = [output.replace("\n", " ") for output in outputs]

        precision, recall, f1 = self.bert_scorer.score(outputs, [target_prompt]*len(images))

        return recall.to(self.device), outputs

    def to(self, device):
        super().to(device)
        self.model = self.model.to(device)
        self.bert_scorer._model = self.bert_scorer._model.to(device)
        self.bert_scorer.device = device
        return self