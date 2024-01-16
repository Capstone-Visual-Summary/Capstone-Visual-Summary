from typing import Union
from Grand_Parent import GrandParent
from PIL import Image
from IMG2VEC_class import Img2Vec
from torch import torch
from pathlib import Path
from torchvision import models
import torch.nn as nn
import csv
import ast
from torchvision import transforms

class EmbeddingParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Embedding"
        self.children: dict[str, dict[str, Union[str, EmbeddingParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, **kwargs):
        version = kwargs['embedding_version'] if 'embedding_version' in kwargs else -1
        
        return super().run(version, **kwargs)


class EmbeddingResNet(EmbeddingParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "EmbeddingResNet 1.0"

    def Image2Vec_embedder_ResNet50(self, image) -> torch.Tensor:
        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        img2vec = Img2Vec(cuda=use_cuda, model='resnet50', layer='default', layer_output_size=2048, gpu=0)
        # layer = 'layer_name' For advanced users, which layer of the model to extract the output from.   default: 'avgpool'
        img = Image.open(image).convert('RGB')
        vec = torch.tensor(img2vec.get_vec(img))
        return vec

    def Image2Vec_embedder_ResNet152(self, image) -> torch.Tensor:
        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        img2vec = Img2Vec(cuda=use_cuda, model='resnet152', layer='default', layer_output_size=2048, gpu=0)
        # layer = 'layer_name' For advanced users, which layer of the model to extract the output from.   default: 'avgpool'
        img = Image.open(image).convert('RGB')
        vec = torch.tensor(img2vec.get_vec(img))
        return vec
    
    def run(self, **kwargs):
        if 'file_name' in kwargs:
            file_name = f"Embedding Files/kwargs['file_name']"
        else:
            version_split = str(self.version).split('.')
            file_name = f'Embedding Files/Embeddings_{version_split[0]}_{version_split[1]}_0.csv'

        if not hasattr(self, 'image_embeddings'):
            try:
                with open(file_name, mode='r', newline='') as csvfile:
                    temp = csv.DictReader(csvfile)
                    
                self.image_embeddings = dict()
                
                for row in temp:
                    self.image_embeddings[row['image_id']] = torch.Tensor(ast.literal_eval(row['tensor']))
            except:
                self.image_embeddings = dict()
            
        if 'rerun' in kwargs and kwargs['rerun']:
                self.image_embeddings = dict()
        
        if str(kwargs['image_id']) in self.image_embeddings:
            return self.image_embeddings[str(kwargs['image_id'])]
        
        path = 'U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + kwargs['img_path']
        
        image_embedding = self.Image2Vec_embedder_ResNet152(path)
        image_embedding = self.Image2Vec_embedder_ResNet152(path)

        self.image_embeddings[str(kwargs['image_id'])] = image_embedding

        with open(file_name, mode='w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['image_id', 'tensor'])

            if csvfile.tell() == 0:
                csv_writer.writeheader()

            for image_id, tensor in self.image_embeddings.items():
                csv_writer.writerow({'image_id': image_id, 'tensor': tensor.tolist()})
            
        return image_embedding
     

class EmbeddingResNet_2_0(EmbeddingParent):
    def __init__(self) -> None:
        self.version: float | str = '2.0 WIP'
        self.name: str = "EmbeddingResNet 2.0" 

    def Image2Vec_embedder_ResNet152(self, image) -> torch.Tensor:
        resnet152 = models.resnet152(pretrained=True)
        resnet152.eval()

        # Define the transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        modules = list(resnet152.children())[:-2] #-2 without avgpool and fc
        resnet152 = nn.Sequential(*modules)
        
        img = Image.open(image).convert('RGB')
        vec = torch.tensor(img2vec.get_vec(img))
        return vec
        
    def run(self, **kwargs):
        file_name = 'Embedding Files/Embeddings_1_0_0.csv'

        if not hasattr(self, 'image_embeddings'):
            try:
                with open(file_name, mode='r', newline='') as csvfile:
                    temp = csv.DictReader(csvfile)
                    
                self.image_embeddings = dict()
                
                for row in temp:
                    self.image_embeddings[row['image_id']] = torch.Tensor(ast.literal_eval(row['tensor']))
            except:
                self.image_embeddings = dict()
        
        if str(kwargs['image_id']) in self.image_embeddings:
            return self.image_embeddings[str(kwargs['image_id'])]
        
        path = 'U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + kwargs['img_path']
        
        if kwargs['resnet'] == 50:
            image_embedding = self.Image2Vec_embedder_ResNet50(path)
        else:
            image_embedding = self.Image2Vec_embedder_ResNet152(path)

        self.image_embeddings[str(kwargs['image_id'])] = image_embedding

        with open(file_name, mode='w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['image_id', 'tensor'])

            if csvfile.tell() == 0:
                csv_writer.writeheader()

            for image_id, tensor in self.image_embeddings.items():
                csv_writer.writerow({'image_id': image_id, 'tensor': tensor.tolist()})
            
        return image_embedding
