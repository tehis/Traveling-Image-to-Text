import io
import os

import gdown
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import psutil
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPConfig,
    CLIPModel,
    CLIPVisionModel,
    default_data_collator,
)


# Download test_set_general.csv
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1v5G6du9Lq9RPk0n6lterAiKkqaVKQ-qG",
    "/var/lib/data/",
)
# Download text_embeddings_specific.pt
gdown.download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1kCdvvY60S_FSvLPOLsfcqb8ZpvSn6J-8",
    "/var/lib/data/",
)


MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
IMAGE_SIZE = 224
MAX_LEN = 80
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


test_df_general = pd.read_csv("/var/lib/data/test_set_general.csv")


class VisionDataset(Dataset):
    preprocess = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    def __init__(self, image_paths: list):
        self.image_paths = image_paths

    def __getitem__(self, idx):
        return self.preprocess(Image.open(self.image_paths[idx]).convert("RGB"))

    def __len__(self):
        return len(self.image_paths)


class TextDataset(Dataset):
    def __init__(self, text: list, tokenizer, max_len):
        self.len = len(text)
        self.tokens = tokenizer(
            text, padding="max_length", max_length=max_len, truncation=True
        )

    def __getitem__(self, idx):
        token = self.tokens[idx]
        return {"input_ids": token.ids, "attention_mask": token.attention_mask}

    def __len__(self):
        return self.len


class CLIPDemo:
    def __init__(
        self,
        vision_encoder,
        text_encoder,
        tokenizer,
        batch_size: int = 32,
        max_len: int = 32,
        device="cpu",
    ):
        self.vision_encoder = vision_encoder.eval().to(device)
        self.text_encoder = text_encoder.eval().to(device)
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_embeddings = None
        self.image_embeddings = None

    def compute_image_embeddings(self, image_paths: list):
        self.image_paths = image_paths
        datalodear = DataLoader(
            VisionDataset(image_paths=image_paths), batch_size=self.batch_size
        )
        embeddings = []
        with torch.no_grad():
            for images in tqdm(datalodear, desc="computing image embeddings"):
                image_embedding = self.vision_encoder(
                    pixel_values=images.to(self.device)
                ).pooler_output
                embeddings.append(image_embedding)
        self.image_embeddings = torch.cat(embeddings)

    def compute_text_embeddings(self, text: list):
        self.text = text
        dataloader = DataLoader(
            TextDataset(text=text, tokenizer=self.tokenizer,
                        max_len=self.max_len),
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
        )
        embeddings = []
        with torch.no_grad():
            for tokens in tqdm(dataloader, desc="computing text embeddings"):
                image_embedding = self.text_encoder(
                    input_ids=tokens["input_ids"].to(self.device),
                    attention_mask=tokens["attention_mask"].to(self.device),
                ).pooler_output
                embeddings.append(image_embedding)
        self.text_embeddings = torch.cat(embeddings)

    def image_query_embedding(self, image):
        image = VisionDataset.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_embedding = self.vision_encoder(
                image.to(self.device)).pooler_output
        return image_embedding

    def most_similars(self, embeddings_1, embeddings_2):
        values, indices = torch.cosine_similarity(embeddings_1, embeddings_2).sort(
            descending=True
        )
        return values.cpu(), indices.cpu()

    def zero_shot(self, image_path: str):
        top_k = 5
        output_num = 5
        """ Zero shot image classification with label list 
            Args:
                image_path (str): target image path that is going to be classified
                class_list (list[str]): list of candidate classes 
        """
        image = Image.open(image_path)
        image_embedding = self.image_query_embedding(image)
        values, indices = self.most_similars(
            image_embedding, self.text_embeddings)
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run()
        for i, sim in zip(indices, torch.softmax(values, dim=0)):
            print(f"Probability : {float(sim)}")
            print(f"label: {self.text[i]}")
            print("_________________________")
            top_k -= 1
            metric_name = (
                "top" + str(output_num) + "_zeroshot_" +
                str((output_num - top_k))
            )
            mlflow.log_metrics(
                {
                    metric_name: float(sim),
                }
            )
            if top_k == 0:
                mlflow.end_run()
                break
        plt.imshow(image)
        plt.axis("off")

    def caption_search(self, image_path: str):
        base_image = Image.open(image_path)
        image_embedding = self.image_query_embedding(base_image)
        values, indices = self.most_similars(
            self.text_embeddings, image_embedding)

        return values, indices

    def monitor_hardware(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        return cpu_percent , memory_percent

    def predict(self, image, use_case):
        top_k = 5
        output_num = 5
        output_dict = {}

        cpu_percent, memory_percent = self.monitor_hardware()
        start_time = time.time()

        image_embedding = self.image_query_embedding(image)
        values, indices = self.most_similars(
            image_embedding, self.text_embeddings)
        latency = time.time() - start_time
        params = {
            "top_k_predictor"+"_"+ str(use_case): top_k,
            "batch_size"+"_"+ str(use_case): self.batch_size,
            "device"+"_"+ str(use_case): self.device,
        }
        mlflow.log_params(params)
        for i, sim in zip(indices, torch.softmax(values, dim=0)):
            output_dict[f'Rank-{abs(top_k - output_num) + 1}'] = {
                'Probability': float(f"{float(sim)*100:.4f}"),
                'label': self.text[i]
            }
            top_k -= 1
            metric_name = "top" + str(output_num) +"_" + str((output_num - top_k))+"_"+ str(use_case)
            mlflow.log_metrics({
                metric_name: float(sim)*100,
            })

            if top_k == 0:
                mlflow.log_metric("CPU Usage"+"_"+ str(use_case), cpu_percent)
                mlflow.log_metric("Memory Usage"+"_"+ str(use_case), memory_percent)
                mlflow.log_metric("Latency"+"_"+ str(use_case), latency)
                break
        return output_dict

def clip_wraper_creator():
    """create a dummy CLIPModel to wrap text and vision encoders in order to use CLIPTrainer"""
    config = {'num_hidden_layers': 0,
              'max_position_embeddings': 0,
              'vocab_size': 0,
              'hidden_size': 1,
              'patch_size': 1,
              }
    DUMMY_CONFIG = CLIPConfig(text_config_dict=config,
                              vision_config_dict=config)
    clip = CLIPModel(config=DUMMY_CONFIG)
    # convert projectors to Identity
    clip.text_projection = nn.Identity()
    clip.visual_projection = nn.Identity()
    return clip

TEXT_MODEL = 'roberta-base'
IMAGE_MODEL = 'openai/clip-vit-base-patch32'
clip_raw = clip_wraper_creator()
vision_encoder_raw = CLIPVisionModel.from_pretrained(IMAGE_MODEL)
text_encoder_raw = AutoModel.from_pretrained(TEXT_MODEL)

clip_raw.text_model = text_encoder_raw
clip_raw.vision_model = vision_encoder_raw


search_demo_raw = CLIPDemo(clip_raw.vision_model, clip_raw.text_model, tokenizer)
search_demo_raw.text_embeddings = torch.load(
    '/var/lib/data/text_embeddings_general_no_finetune.pt')


app = FastAPI()


EXPERIMENT_NAME = "shadow expriment"
# EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
RUN_NAME_General = "shadow runs"
mlflow.set_experiment(EXPERIMENT_NAME)
run_General = mlflow.start_run(run_name=RUN_NAME_General)
# run_General = mlflow.start_run(run_name=RUN_NAME_General)
RUN_ID_General = run_General.info.run_id

# EXPERIMENT_NAME_Specific = "Specific_Label"
# EXPERIMENT_ID_Specific = mlflow.create_experiment(EXPERIMENT_NAME_Specific)
# RUN_NAME_Specific = "Specific_Label"
# mlflow.set_experiment(EXPERIMENT_NAME)
# run_Specific = mlflow.start_run(run_name=RUN_NAME_Specific)
# run_Specific = mlflow.start_run(run_name=RUN_NAME_Specific)
# RUN_ID_Specific = run_Specific.info.run_id

origins = ["http://localhost", "http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def upload(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict")
def prediction_api(request: Request, image: UploadFile = File(...)):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    output_general = search_demo_raw.predict(image.copy(), "General_Label")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "rank_1_general": output_general["Rank-1"],
            "rank_2_general": output_general["Rank-2"],
            "rank_3_general": output_general["Rank-3"],
            "rank_4_general": output_general["Rank-4"],
            "rank_5_general": output_general["Rank-5"]
        },
    )
