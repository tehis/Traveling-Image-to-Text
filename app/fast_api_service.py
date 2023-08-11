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
# Download supervised_test_set.csv
gdown.download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1NoAluvhq4Z8A0m9TV1gfpxBq8sPgNap7",
    "/var/lib/data/",
)

# Download text_model folder content
if not os.path.exists("/var/lib/data/text_model_general_label"):
    os.mkdir("/var/lib/data/text_model_general_label")
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1-6ThDz5S7GZeTtP74c7B4TkZ1vKS2sP6",
    "/var/lib/data/text_model_general_label/config.json",
)
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1-5L29XnzokoHMfMEvw7wZb6fGc5j1O6p",
    "/var/lib/data/text_model_general_label/pytorch_model.bin",
)


# Download vision_model folder content
if not os.path.exists("/var/lib/data/vision_model_general_label"):
    os.mkdir("/var/lib/data/vision_model_general_label")
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1--Akn08LVreaaInW6Dsa8hw6FEF7GWFP",
    "/var/lib/data/vision_model_general_label/config.json",
)
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1--eKcoWllY3pNdJckVLaGyuSmRn-KrI-",
    "/var/lib/data/vision_model_general_label/pytorch_model.bin",
)

# Download text_model folder content for specific label
if not os.path.exists("/var/lib/data/text_model_specific_label"):
    os.mkdir("/var/lib/data/text_model_specific_label")
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1-0P8fKqWHFQfolSeR0XlCk9wtBK_DBvC",
    "/var/lib/data/text_model_specific_label/config.json",
)
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1-33EKuSDdU6Y4RDMg07r6SThmThtAp62",
    "/var/lib/data/text_model_specific_label/pytorch_model.bin",
)


# Download vision_model folder content for specific label
if not os.path.exists("/var/lib/data/vision_model_specific_label"):
    os.mkdir("/var/lib/data/vision_model_specific_label")
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1-9YYdlvkMP1dx8Ps6y1IpwnFBDiclIUY",
    "/var/lib/data/vision_model_specific_label/config.json",
)
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1-ECAI3nwRL78YVd-6d7EECKepwes59KB",
    "/var/lib/data/vision_model_specific_label/pytorch_model.bin",
)

# Download text_embeddings_general.pt
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1isO8_XFYvKvvJh8adeGRO7tGlWDKHpzS",
    "/var/lib/data/",
)

# Download text_embeddings_specific.pt
gdown.cached_download(
    "https://drive.google.com/uc?export=download&confirm=pbef&id=1yfrAp9dHgvbaVUxjsRagGayq4zhvLvsi",
    "/var/lib/data/",
)


vision_model_general = CLIPVisionModel.from_pretrained(
    "/var/lib/data/vision_model_general_label", local_files_only=True
)
text_model_general = AutoModel.from_pretrained(
    "/var/lib/data/text_model_general_label", local_files_only=True
)

vision_model_specific = CLIPVisionModel.from_pretrained(
    "/var/lib/data/vision_model_specific_label", local_files_only=True
)
text_model_specific = AutoModel.from_pretrained(
    "/var/lib/data/text_model_specific_label", local_files_only=True
)

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
IMAGE_SIZE = 224
MAX_LEN = 80
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


test_df_general = pd.read_csv("/var/lib/data/test_set_general.csv")
test_df_specific = pd.read_csv("/var/lib/data/supervised_test_set.csv")


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
        # mlflow: stop active runs if any
        if mlflow.active_run():
            mlflow.end_run()
        # mlflow:track run
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
                # mlflow: end tracking
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

    def predict(self, image, use_case):
        top_k = 5
        output_num = 5
        output_dict = {}
        # mlflow: stop active runs if any
        if mlflow.active_run():
            mlflow.end_run()
        # mlflow:track run
        # tracking_uri = ""
        # mlflow.set_tracking_uri(tracking_uri)
        # experiment_name = ""
        # mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        # monitor hardware usage
        cpu_percent, memory_percent = self.monitor_hardware()
        start_time = time.time()

        image_embedding = self.image_query_embedding(image)
        values, indices = self.most_similars(
            image_embedding, self.text_embeddings)
        latency = time.time() - start_time
        params = {
            "top_k_predictor": top_k,
            "batch_size": self.batch_size,
            "device": self.device,
            "use_case": use_case
        }
        mlflow.log_params(params)
        # top5_1 = Gauge('top5_1', 'top5 1')
        for i, sim in zip(indices, torch.softmax(values, dim=0)):
            output_dict[f'Rank-{abs(top_k - output_num) + 1}'] = {
                'Probability': float(sim),
                'label': self.text[i]
            }
            top_k -= 1
            metric_name = "top" + str(output_num) + \
                "_" + str((output_num - top_k))
            # if((output_num - top_k)==1):
            #     top5_1.set(float(sim)*1000)
            mlflow.log_metrics({
                metric_name: float(sim)*1000,
            })

            if top_k == 0:
                mlflow.log_metric("CPU Usage", cpu_percent)
                mlflow.log_metric("Memory Usage", memory_percent)
                mlflow.log_metric("Latency", latency)
                # mlflow: end tracking
                # mlflow.pytorch.log_model(self.vision_encoder, "vision_encoder")
                # mlflow.pytorch.log_model(self.vision_encoder, "vision_encoder")

                # mlflow.pytorch.log_model(
                #     pytorch_model= self.vision_encoder,
                #     registered_model_name="vision_encoder",
                #     artifact_path="vision_encoder"
                # )
                # mlflow.pytorch.log_model(
                #     pytorch_model= self.text_encoder,
                #     registered_model_name="text_encoder",
                #     artifact_path="text_encoder"
                # )
                mlflow.end_run()
                break
        return output_dict


search_demo_general = CLIPDemo(
    vision_model_general, text_model_general, tokenizer)
search_demo_specific = CLIPDemo(
    vision_model_specific, text_model_specific, tokenizer)
# search_demo_general.compute_text_embeddings(test_df_general.label.tolist())
# search_demo_specific.compute_text_embeddings(test_df_specific.label.tolist())

search_demo_general.text = test_df_general.label.tolist()
search_demo_general.text_embeddings = torch.load(
    '/var/lib/data/text_embeddings_general.pt')

search_demo_specific.text = test_df_specific.label.tolist()
search_demo_specific.text_embeddings = torch.load(
    '/var/lib/data/text_embeddings_specific.pt')


app = FastAPI()
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
    output_general = search_demo_general.predict(image.copy(), "General_Label")
    output_specific = search_demo_specific.predict(image.copy(), "Specific_Label")

    return templates.TemplateResponse(
        # "result.html", {
        #     "request": request,
        #     "rank_1": {"Probability": 1, "label": "salgijfgg "},
        #     "rank_2": {"Probability": 1.23, "label": "sgdghsalgijfgg "},
        #     "rank_3": {"Probability": 0.245, "label": "salg dfg ggdhijfgg "},
        #     "rank_4": {"Probability": 0.34545645, "label": "salgijdfghgh dgsh fgg dlfgje gmoejqojeogmoe;mg joetjmgo;wtjot tjii "},
        #     "rank_5": {"Probability": 0.0000035, "label": "salgijfglegkjds t jeg;ojh;iogsj orwhjio;THkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk PJRTHJ RTHJ'WHJO'HJRG J;ORTIJHROIRGJHOJ jmrojphtrg dfsh gfh dsfh fsh \n adryyt "},
        # }
        "result.html",
        {
            "request": request,
            "rank_1_general": output_general["Rank-1"],
            "rank_2_general": output_general["Rank-2"],
            "rank_3_general": output_general["Rank-3"],
            "rank_4_general": output_general["Rank-4"],
            "rank_5_general": output_general["Rank-5"],
            "rank_1_specific": output_specific["Rank-1"],
            "rank_2_specific": output_specific["Rank-2"],
            "rank_3_specific": output_specific["Rank-3"],
            "rank_4_specific": output_specific["Rank-4"],
            "rank_5_specific": output_specific["Rank-5"],
        },
    )
