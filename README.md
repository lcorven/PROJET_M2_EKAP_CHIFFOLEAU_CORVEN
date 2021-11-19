# PROJET_M2_EKAP_CHIFFOLEAU_CORVEN
 
 
 Description : 

A picture is worth a thousand words, yet sometimes a few will do. We all rely on online images for knowledge sharing, learning, and understanding. Even the largest websites are missing visual content and metadata to pair with their images. Captions and “alt text” increase accessibility and enable better search. The majority of images on Wikipedia articles, for example, don't have any written context connected to the image. Open models could help anyone improve accessibility and learning for all.

Current solutions rely on simple methods based on translations or page interlinks, which have limited coverage. Even the most advanced computer vision image captioning isn't suitable for images with complex semantics.


 
Code projet interessant : https://www.kaggle.com/hijest/wikipedia-image-caption-matching-starter-eda
 
Initialisation : 

import os
import requests

General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import PIL.Image

from IPython.display import Image, display
import warnings
warnings.filterwarnings("ignore")




#inportation des packages et librairies nécessaire à la realisation du projet. 
! pip install timm
! pip install --upgrade wandb
! pip install transformers
! pip install --user albumentations

Requirement already satisfied: timm in c:\users\loicc\anaconda3\lib\site-packages (0.4.12)
Requirement already satisfied: torchvision in c:\users\loicc\appdata\roaming\python\python38\site-packages (from timm) (0.11.1)
Requirement already satisfied: torch>=1.4 in c:\users\loicc\appdata\roaming\python\python38\site-packages (from timm) (1.10.0)
Requirement already satisfied: typing-extensions in c:\users\loicc\anaconda3\lib\site-packages (from torch>=1.4->timm) (3.7.4.3)
Requirement already satisfied: numpy in c:\users\loicc\anaconda3\lib\site-packages (from torchvision->timm) (1.20.1)
Requirement already satisfied: pillow!=8.3.0,>=5.3.0 in c:\users\loicc\anaconda3\lib\site-packages (from torchvision->timm) (8.2.0)
Requirement already satisfied: wandb in c:\users\loicc\anaconda3\lib\site-packages (0.12.6)
Requirement already satisfied: docker-pycreds>=0.4.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (0.4.0)
Requirement already satisfied: yaspin>=1.0.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (2.1.0)
Requirement already satisfied: promise<3,>=2.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (2.3)
Requirement already satisfied: requests<3,>=2.0.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (2.25.1)
Requirement already satisfied: psutil>=5.0.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (5.8.0)
Requirement already satisfied: protobuf>=3.12.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (3.19.1)
Requirement already satisfied: GitPython>=1.0.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (3.1.24)
Requirement already satisfied: subprocess32>=3.5.3 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (3.5.4)
Requirement already satisfied: PyYAML in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (5.4.1)
Requirement already satisfied: pathtools in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (0.1.2)
Requirement already satisfied: sentry-sdk>=1.0.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (1.5.0)
Requirement already satisfied: Click!=8.0.0,>=7.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (7.1.2)
Requirement already satisfied: six>=1.13.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (1.15.0)
Requirement already satisfied: shortuuid>=0.5.0 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (1.0.8)
Requirement already satisfied: python-dateutil>=2.6.1 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (2.8.1)
Requirement already satisfied: configparser>=3.8.1 in c:\users\loicc\anaconda3\lib\site-packages (from wandb) (5.1.0)
Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\users\loicc\anaconda3\lib\site-packages (from GitPython>=1.0.0->wandb) (3.7.4.3)
Requirement already satisfied: gitdb<5,>=4.0.1 in c:\users\loicc\anaconda3\lib\site-packages (from GitPython>=1.0.0->wandb) (4.0.9)
Requirement already satisfied: chardet<5,>=3.0.2 in c:\users\loicc\anaconda3\lib\site-packages (from requests<3,>=2.0.0->wandb) (4.0.0)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\loicc\anaconda3\lib\site-packages (from requests<3,>=2.0.0->wandb) (1.26.4)
Requirement already satisfied: idna<3,>=2.5 in c:\users\loicc\anaconda3\lib\site-packages (from requests<3,>=2.0.0->wandb) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\loicc\anaconda3\lib\site-packages (from requests<3,>=2.0.0->wandb) (2020.12.5)
Requirement already satisfied: termcolor<2.0.0,>=1.1.0 in c:\users\loicc\anaconda3\lib\site-packages (from yaspin>=1.0.0->wandb) (1.1.0)
Requirement already satisfied: smmap<6,>=3.0.1 in c:\users\loicc\anaconda3\lib\site-packages (from gitdb<5,>=4.0.1->GitPython>=1.0.0->wandb) (5.0.0)
Requirement already satisfied: transformers in c:\users\loicc\anaconda3\lib\site-packages (4.12.5)
Requirement already satisfied: numpy>=1.17 in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (1.20.1)
Requirement already satisfied: pyyaml>=5.1 in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (5.4.1)
Requirement already satisfied: huggingface-hub<1.0,>=0.1.0 in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (0.1.2)
Requirement already satisfied: tqdm>=4.27 in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (4.59.0)
Requirement already satisfied: packaging>=20.0 in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (20.9)
Requirement already satisfied: sacremoses in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (0.0.46)
Requirement already satisfied: tokenizers<0.11,>=0.10.1 in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (0.10.3)
Requirement already satisfied: filelock in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (3.0.12)
Requirement already satisfied: regex!=2019.12.17 in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (2021.4.4)
Requirement already satisfied: requests in c:\users\loicc\anaconda3\lib\site-packages (from transformers) (2.25.1)
Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\users\loicc\anaconda3\lib\site-packages (from huggingface-hub<1.0,>=0.1.0->transformers) (3.7.4.3)
Requirement already satisfied: pyparsing>=2.0.2 in c:\users\loicc\anaconda3\lib\site-packages (from packaging>=20.0->transformers) (2.4.7)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\loicc\anaconda3\lib\site-packages (from requests->transformers) (1.26.4)
Requirement already satisfied: idna<3,>=2.5 in c:\users\loicc\anaconda3\lib\site-packages (from requests->transformers) (2.10)
Requirement already satisfied: chardet<5,>=3.0.2 in c:\users\loicc\anaconda3\lib\site-packages (from requests->transformers) (4.0.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\loicc\anaconda3\lib\site-packages (from requests->transformers) (2020.12.5)
Requirement already satisfied: joblib in c:\users\loicc\anaconda3\lib\site-packages (from sacremoses->transformers) (1.0.1)
Requirement already satisfied: click in c:\users\loicc\anaconda3\lib\site-packages (from sacremoses->transformers) (7.1.2)
Requirement already satisfied: six in c:\users\loicc\anaconda3\lib\site-packages (from sacremoses->transformers) (1.15.0)
Requirement already satisfied: albumentations in c:\users\loicc\appdata\roaming\python\python38\site-packages (1.1.0)
Requirement already satisfied: scikit-image>=0.16.1 in c:\users\loicc\anaconda3\lib\site-packages (from albumentations) (0.18.1)
Requirement already satisfied: PyYAML in c:\users\loicc\anaconda3\lib\site-packages (from albumentations) (5.4.1)
Requirement already satisfied: opencv-python-headless>=4.1.1 in c:\users\loicc\appdata\roaming\python\python38\site-packages (from albumentations) (4.5.4.58)
Requirement already satisfied: numpy>=1.11.1 in c:\users\loicc\anaconda3\lib\site-packages (from albumentations) (1.20.1)
Requirement already satisfied: scipy in c:\users\loicc\anaconda3\lib\site-packages (from albumentations) (1.6.2)
Requirement already satisfied: qudida>=0.0.4 in c:\users\loicc\appdata\roaming\python\python38\site-packages (from albumentations) (0.0.4)
Requirement already satisfied: typing-extensions in c:\users\loicc\anaconda3\lib\site-packages (from qudida>=0.0.4->albumentations) (3.7.4.3)
Requirement already satisfied: scikit-learn>=0.19.1 in c:\users\loicc\anaconda3\lib\site-packages (from qudida>=0.0.4->albumentations) (0.24.1)
Requirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in c:\users\loicc\anaconda3\lib\site-packages (from scikit-image>=0.16.1->albumentations) (3.3.4)
Requirement already satisfied: networkx>=2.0 in c:\users\loicc\anaconda3\lib\site-packages (from scikit-image>=0.16.1->albumentations) (2.5)
Requirement already satisfied: pillow!=7.1.0,!=7.1.1,>=4.3.0 in c:\users\loicc\anaconda3\lib\site-packages (from scikit-image>=0.16.1->albumentations) (8.2.0)
Requirement already satisfied: imageio>=2.3.0 in c:\users\loicc\anaconda3\lib\site-packages (from scikit-image>=0.16.1->albumentations) (2.9.0)
Requirement already satisfied: tifffile>=2019.7.26 in c:\users\loicc\anaconda3\lib\site-packages (from scikit-image>=0.16.1->albumentations) (2021.4.8)
Requirement already satisfied: PyWavelets>=1.1.1 in c:\users\loicc\anaconda3\lib\site-packages (from scikit-image>=0.16.1->albumentations) (1.1.1)
Requirement already satisfied: cycler>=0.10 in c:\users\loicc\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (0.10.0)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\loicc\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (2.4.7)
Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\loicc\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (1.3.1)
Requirement already satisfied: python-dateutil>=2.1 in c:\users\loicc\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (2.8.1)
Requirement already satisfied: decorator>=4.3.0 in c:\users\loicc\anaconda3\lib\site-packages (from networkx>=2.0->scikit-image>=0.16.1->albumentations) (5.0.6)
Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\loicc\anaconda3\lib\site-packages (from scikit-learn>=0.19.1->qudida>=0.0.4->albumentations) (2.1.0)
Requirement already satisfied: joblib>=0.11 in c:\users\loicc\anaconda3\lib\site-packages (from scikit-learn>=0.19.1->qudida>=0.0.4->albumentations) (1.0.1)
Requirement already satisfied: six in c:\users\loicc\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (1.15.0)

#Appel des packages et librairies necessaire à la realisation du porjet
import os
import gc
import cv2
import copy
import time
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import base64
import pickle

#librairie qui vont servire à  télécherger les images
from io import BytesIO

# Libraries qui vont permettre de manipuler les images
import numpy as np
import pandas as pd

# Libraries Pytorch
# Pytorch est un package qui a deux spcécifsité interessante. 
# il va permettre de mettre en place des réseaux de neuronnes profonds construit
# sur un systéme d'autogradation des bandes. Ce package ce base sur le GPU
# La pluspart des frameworks comme TensorFlows sont basé sur des dynamique statsique
# PyTorch utilise quant aà lui une technique d'auto-différenciation qui va permettre de rendre 
# dynamique le reseaux de neuronnes en le modifiantarbitrairement. 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

import joblib
from tqdm import tqdm
from collections import defaultdict

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

import timm

# Permet de transdormé les models
import transformers
from transformers import AutoTokenizer, AutoModel

import albumentations as A
from albumentations.pytorch import ToTensorV2

# package de misen en forme texteuelle 
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# Permet de decrire les message d'erreur. 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

                # Weights & Biases 
    # Weights & Biases  est un outil d'apprentissage automatique 
    # qui permet de creer des modèle plus rapidement
    # Généralement il est préférable de devellopper et d'evalué rapidement un modèle 
    # c'est particuliérement le cas avec notre concour qui necessite un traitement de beaucoup de données volumineuse. 
    # Dans notre cas nous utliserons Weights & Biases  particuliémrent pour stocker versionner les données. 
    

import wandb

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
     #utilisation d'une API 
    api_key = user_secrets.get_secret("wandb_api")
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"

#Training Configuration

#COnfiguration du modèle cnsidéré comme la plus pertinente. 
CONFIG = {"seed": 2021,
          "epochs": 3,
          "img_size": 256,
          "image_model_name": "tf_efficientnet_b0",
          "text_model_name": "xlm-roberta-base",
          "embedding_size": 256,
          "train_batch_size": 10,
          "valid_batch_size": 10,
          "learning_rate": 0.10,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 0.50,
          "T_max": 50,
          "weight_decay": 0.10,
          "max_length": 32,
          "n_accumulate": 1,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          }

CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['text_model_name'])

CONFIG

{'seed': 2021,
 'epochs': 3,
 'img_size': 256,
 'image_model_name': 'tf_efficientnet_b0',
 'text_model_name': 'xlm-roberta-base',
 'embedding_size': 256,
 'train_batch_size': 10,
 'valid_batch_size': 10,
 'learning_rate': 0.1,
 'scheduler': 'CosineAnnealingLR',
 'min_lr': 0.5,
 'T_max': 50,
 'weight_decay': 0.1,
 'max_length': 32,
 'n_accumulate': 1,
 'device': device(type='cpu'),
 'tokenizer': PreTrainedTokenizerFast(name_or_path='xlm-roberta-base', vocab_size=250002, model_max_len=512, is_fast=True, padding_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False)})}

                                            #Set Seed for Reproducibility

#On définit iciun noyaux de base qui va eléminer le facteur aléatoire des
# modèle afin que les resultat soirent les mêmes à chaque fois que le script sera
# utlisé. Anisi le modèle aura une meuilleur reproductibilité. Cepedant on pert un 
# peu en représentativité.
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
# définition des options torch, qui utilise cuDNN de NVIDIA 
# cuDNN va permettre de faire des reseaux de neuronnes profonds en optimisaznt les performance du CPU
#Or cuDNN avec des convolutions CUDA peu etre sources de non-déterminisme sur plusieur exécution.
#Quand une convolution cuDNN est applelé avec un nouvelle esenmble de paramétre de taille,
# une fonctinnlaité peut venir exécuter plusieur algorothme de convolution en les comparant pour trouver les plus perfromants. 
#En raison du bruit de l'analyse comparative, mais également du matériel utilisé, 
# l'analyse comparative peut selectoinné des modèle différent. Ainsi ici La désactivation de la fonction d'analyse comparative a
# amène cuDNN à sélectionner de manière déterministe un algorithme, mais au prix de performances réduites. fonction : torch.backends.cudnn.benchmark = False
#
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])

                                            #Data importation 

import json

run = wandb.init(project="Wikipedia", 
                 anonymous="must")
artifact = run.use_artifact('dchanda/Wikipedia/Wiki-data:latest', type='dataset')
artifact_dir = artifact.download()
run.finish()

for file in os.listdir(artifact_dir):
    filepath = os.path.join(artifact_dir, file)
    with open(filepath, "rb") as fp:
        data = pickle.load(fp)
        data=data[:200]

wandb: Currently logged in as: anony-mouse-158971 (use `wandb login --relogin` to force relogin)

Syncing run solar-field-21 to Weights & Biases (docs).

wandb: Downloading large artifact Wiki-data:latest, 2168.54MB. 1 files... Done. 0:0:0


Waiting for W&B process to finish, PID 7420... (success).
Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Synced solar-field-21: https://wandb.ai/anony-mouse-158971/Wikipedia/runs/1hxxhlgs?apiKey=530a6b96a52d389faa7ce76e4b44825adc3e368d
Find logs at: .\wandb\run-20211119_152203-1hxxhlgs\logs

random.shuffle(data)

train_data = data[:45000]
valid_data = data[45000:]
print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(valid_data)}")

Number of training samples: 200
Number of validation samples: 0

                                            #Visualize Images

 #run = wandb.init(project='Wikipedia',
  #                job_type='Visualization',
   #               anonymous='must')

# preview_table = wandb.Table(columns=['Image', 'Captions'])
 #for content in json_content[:1000]:
  #   out = base64.b64decode(content['b64_bytes'])
   #  img = Image.open(BytesIO(out)).convert("RGB")
    # preview_table.add_data(wandb.Image(img), 
     #                       content['caption_title_and_reference_description'])

 #wandb.log({'Visualization': preview_table})
 #run.finish()

#Dataset Class

class WikipediaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, transforms=None):
        self.data = data
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_bytes = base64.b64decode(self.data[index]["b64_bytes"])
        img = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
        caption = random.choice(self.data[index]["caption_title_and_reference_description"])
        caption = caption.replace("[SEP]", "</s>") # sep token for xlm-roberta
        inputs = self.tokenizer.encode_plus(
                caption,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length'
            )
        target = self.data[index]['target']
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'image': img,
            'target': torch.tensor(target, dtype=torch.long)
        }

data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}

train_dataset = WikipediaDataset(train_data, CONFIG["tokenizer"], CONFIG["max_length"], 
                                 transforms=data_transforms["train"])
train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                          num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

valid_dataset = WikipediaDataset(valid_data, CONFIG["tokenizer"], CONFIG["max_length"], 
                                 transforms=data_transforms["valid"])
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                          num_workers=4, shuffle=False, pin_memory=True)

                                           #Create Model

class WikipediaModel(nn.Module):
    def __init__(self, image_model, text_model, embedding_size):
        super(WikipediaModel, self).__init__()
        self.image_model = timm.create_model(image_model, pretrained=True)
        self.n_features = self.image_model.classifier.in_features
        self.image_model.reset_classifier(0)
        self.image_drop = nn.Dropout(p=0.2)
        self.image_fc = nn.Linear(self.n_features, embedding_size)
        
        self.text_model = AutoModel.from_pretrained(text_model)
        self.text_drop = nn.Dropout(p=0.2)
        self.text_fc = nn.Linear(768, embedding_size)
        
        self.freeze_backbone()
        
    def forward(self, images, ids, mask):
        image_features = self.image_model(images)
        image_embeddings = self.image_fc(self.image_drop(image_features))
        
        out = self.text_model(input_ids=ids,attention_mask=mask,
                              output_hidden_states=False)
        out = self.text_drop(out[1])
        text_embeddings = self.text_fc(out)

        return image_embeddings, text_embeddings
    
    def freeze_backbone(self):
        for params in self.image_model.parameters():
            params.requires_grad = False
        # Only finetune final layer
        self.image_fc.weight.requires_grad = True
        self.image_fc.bias.requires_grad = True
        
        for params in self.text_model.parameters():
            params.requires_grad = False
        # Only finetune final layer
        self.text_fc.weight.requires_grad = True
        self.text_fc.bias.requires_grad = True
    

model = WikipediaModel(CONFIG['image_model_name'], CONFIG['text_model_name'], CONFIG['embedding_size'])
model.to(CONFIG['device']);

Some weights of the model checkpoint at xlm-roberta-base were not used when initializing XLMRobertaModel: ['lm_head.decoder.weight', 'lm_head.layer_norm.weight', 'lm_head.dense.weight', 'lm_head.bias', 'lm_head.layer_norm.bias', 'lm_head.dense.bias']
- This IS expected if you are initializing XLMRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

                                         #Loss Function

def criterion(outputs1, outputs2, targets):
    return nn.CosineEmbeddingLoss()(outputs1, outputs2, targets)

                                       #Training Function

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)

        image_outputs, text_outputs = model(images, ids, mask)
        loss = criterion(image_outputs, text_outputs, targets)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss

                                         #Validation Function

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)
        
        image_outputs, text_outputs = model(images, ids, mask)
        loss = criterion(image_outputs, text_outputs, targets)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return epoch_loss

                                            #Run Training

def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        
        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            run.summary["Best Loss"] = best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "Loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])
scheduler = fetch_scheduler(optimizer)

run = wandb.init(project='Wikipedia', 
                 config=CONFIG,
                 job_type='Train',
                 anonymous='must')

Syncing run fragrant-forest-22 to Weights & Biases (docs).

model, history = run_training(model, optimizer , scheduler, device ="GPU",
                              num_epochs=2)

run.finish()



  
  
 
 
