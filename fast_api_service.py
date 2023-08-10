import gdown
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator, CLIPModel, CLIPConfig, CLIPVisionModel, AutoModel
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
import io
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tarfile
import os
# import mlflow


## Download test_set_general.csv
gdown.download('https://drive.google.com/uc?export=download&confirm=pbef&id=1v5G6du9Lq9RPk0n6lterAiKkqaVKQ-qG', '/var/lib/data')

# all_images.tar
# gdown.download('https://drive.google.com/uc?export=download&confirm=pbef&id=1iiF-UQx-kGIG54jS791ze3dAceqEvUsx', '/var/lib/data')

## Download text_model folder content
os.mkdir('/var/lib/datatext_model_general_label')
gdown.download('https://drive.google.com/uc?export=download&confirm=pbef&id=1-6ThDz5S7GZeTtP74c7B4TkZ1vKS2sP6',
               '/var/lib/datatext_model_general_label/config.json')
gdown.download('https://drive.google.com/uc?export=download&confirm=pbef&id=1-5L29XnzokoHMfMEvw7wZb6fGc5j1O6p',
               '/var/lib/datatext_model_general_label/pytorch_model.bin')

## Download vision_model folder content
os.mkdir('/var/lib/datavision_model_general_label')
gdown.download('https://drive.google.com/uc?export=download&confirm=pbef&id=1--Akn08LVreaaInW6Dsa8hw6FEF7GWFP',
               '/var/lib/datavision_model_general_label/config.json')
gdown.download('https://drive.google.com/uc?export=download&confirm=pbef&id=1--eKcoWllY3pNdJckVLaGyuSmRn-KrI-',
               '/var/lib/datavision_model_general_label/pytorch_model.bin')


vision_model = CLIPVisionModel.from_pretrained('/var/lib/datavision_model_general_label', local_files_only=True)
text_model = AutoModel.from_pretrained('/var/lib/datatext_model_general_label', local_files_only=True)

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
IMAGE_SIZE = 224
MAX_LEN = 80
tokenizer = AutoTokenizer.from_pretrained('roberta-base')


test_df = pd.read_csv('/var/lib/datatest_set_general.csv')


class VisionDataset(Dataset):
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    def __init__(self, image_paths: list):
        self.image_paths = image_paths

    def __getitem__(self, idx):
        return self.preprocess(Image.open(self.image_paths[idx]).convert('RGB'))

    def __len__(self):
        return len(self.image_paths)


class TextDataset(Dataset):
    def __init__(self, text: list, tokenizer, max_len):
        self.len = len(text)
        self.tokens = tokenizer(text, padding='max_length',
                                max_length=max_len, truncation=True)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        return {'input_ids': token.ids, 'attention_mask': token.attention_mask}

    def __len__(self):
        return self.len


class CLIPDemo:
    def __init__(self, vision_encoder, text_encoder, tokenizer,
                 batch_size: int = 32, max_len: int = 32, device='cpu'):
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
        datalodear = DataLoader(VisionDataset(
            image_paths=image_paths), batch_size=self.batch_size)
        embeddings = []
        with torch.no_grad():
            for images in tqdm(datalodear, desc='computing image embeddings'):
                image_embedding = self.vision_encoder(
                    pixel_values=images.to(self.device)).pooler_output
                embeddings.append(image_embedding)
        self.image_embeddings =  torch.cat(embeddings)

    def compute_text_embeddings(self, text: list):
        self.text = text
        dataloader = DataLoader(TextDataset(text=text, tokenizer=self.tokenizer, max_len=self.max_len),
                                batch_size=self.batch_size, collate_fn=default_data_collator)
        embeddings = []
        with torch.no_grad():
            for tokens in tqdm(dataloader, desc='computing text embeddings'):
                image_embedding = self.text_encoder(input_ids=tokens["input_ids"].to(self.device),
                                                    attention_mask=tokens["attention_mask"].to(self.device)).pooler_output
                embeddings.append(image_embedding)
        self.text_embeddings = torch.cat(embeddings)

    def image_query_embedding(self, image):
        image = VisionDataset.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_embedding = self.vision_encoder(
                image.to(self.device)).pooler_output
        return image_embedding

    def most_similars(self, embeddings_1, embeddings_2):
        values, indices = torch.cosine_similarity(
            embeddings_1, embeddings_2).sort(descending=True)
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
        values, indices = self.most_similars(image_embedding, self.text_embeddings)
        # mlflow: stop active runs if any
        # if mlflow.active_run():
        #     mlflow.end_run()
        # mlflow:track run
        # mlflow.start_run()
        for i, sim in zip(indices, torch.softmax(values, dim=0)):
            print(f'Probability : {float(sim)}')
            print(
                f'label: {self.text[i]}')
            print('_________________________')
            top_k -= 1
            metric_name = "top" + str(output_num) + "_zeroshot_" + str((output_num - top_k))
            # mlflow.log_metrics({
                # metric_name: float(sim),
            # })
            if top_k == 0:
                # mlflow: end tracking
                # mlflow.end_run()
                break
        plt.imshow(image)
        plt.axis('off')

    def caption_search(self, image_path: str):
        base_image = Image.open(image_path)
        image_embedding = self.image_query_embedding(base_image)
        values , indices = self.most_similars(
            self.text_embeddings, image_embedding)

        return values , indices

    def predict(self, image):
        top_k = 5
        output_num = 5
        output_dict = {}
        image_bytes = image.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_embedding = self.image_query_embedding(image)
        values, indices = self.most_similars(image_embedding, self.text_embeddings)
        # mlflow: stop active runs if any
        # if mlflow.active_run():
        #     mlflow.end_run()
        # mlflow:track run
        # mlflow.start_run()
        for i, sim in zip(indices, torch.softmax(values, dim=0)):
            output_dict[f'Rank-{abs(top_k - output_num) + 1}'] = {
              'Probability':float(sim),
              'label':self.text[i]
            }
            top_k -= 1
            metric_name = "top" + str(output_num) + "_" + str((output_num - top_k))
            # mlflow.log_metrics({
            #     metric_name: float(sim),
            # })

            if top_k == 0:
                # mlflow: end tracking
                # mlflow.end_run()
                break
        return output_dict

search_demo = CLIPDemo(vision_model, text_model, tokenizer)
search_demo.compute_text_embeddings(test_df.label.tolist())


app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def prediction_api(image: UploadFile = File (...)):
  output =  search_demo.predict(image)
  return output

# uvicorn.run(app , host="0.0.0.0", port=8000)
