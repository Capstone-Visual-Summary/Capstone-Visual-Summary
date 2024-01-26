from typing import Union
from Grand_Parent import GrandParent
from PIL import Image
from IMG2VEC_class import Img2Vec
from torch import torch
from torchvision import models
import torch.nn as nn
import csv
import ast
import os
from torchvision import transforms
from torchvision.models.resnet import ResNet152_Weights
from typing import List, Dict, Any

class EmbeddingParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Embedding"
        self.children: dict[str, dict[str, Union[str, EmbeddingParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, **kwargs):
        version = kwargs['embedding_version'] if 'embedding_version' in kwargs else -1
        return super().run(version, **kwargs)

    def get_file_path(self, version, img_id_com) -> str:
        """
        Returns the file path corresponding to the given version and image ID combination.

        Parameters:
        version (str): The version of the file.
        img_id_com (int): The combined image ID.

        Returns:
        str: The file path.

        Example:
        >>> get_file_path('1.0', 100)
        'Embedding Files/v1_0_100_0.npy'
        """
        version_split = str(version).split('.')
        version_str = f'v{version_split[0]}_{version_split[1]}'

        files = ['Embedding Files/' + f for f in os.listdir('Embedding Files') if version_str in f]
        files = sorted(files, key=lambda x: int(x.split('_')[3].split('_')[0]))

        return files[int(img_id_com) // 2392]


class EmbeddingResNet(EmbeddingParent):
    def __init__(self) -> None:
        """
        Initializes an instance of the EmbeddingResNet class.

        Attributes:
        - version (float | str): The version number of the embedding class.
        - name (str): The name of the embedding class.
        - files_in_memory (list): A list to store files in memory.
        - image_embeddings (dict): A dictionary to store image embeddings.
        """
        self.version: float | str = 1.0
        self.name: str = "EmbeddingResNet 1.0"
        self.files_in_memory: List[str] = []
        self.image_embeddings: Dict[str, Any] = dict()

    def Image2Vec_embedder_ResNet152(self, image) -> torch.Tensor:
        """
        Embeds an image using the ResNet152 model.

        Args:
            image (str): The path to the image file.

        Returns:
            torch.Tensor: The embedded representation of the image.
        """
        use_cuda = torch.cuda.is_available()
        img2vec = Img2Vec(cuda=use_cuda, model='resnet152', layer='default', layer_output_size=2048, gpu=0)
        img = Image.open(image).convert('RGB')
        vec = torch.tensor(img2vec.get_vec(img))
        return vec
    
    def run(self, **kwargs):
        """
        Runs the embedding process for a given image ID.

        Args:
            **kwargs: Additional keyword arguments.
                - file_name (str): The name of the file containing the image embeddings.
                - image_id (int): The ID of the image to retrieve the embedding for.
                - max_files (int): The maximum number of files to keep in memory.

        Returns:
            torch.Tensor: The embedding tensor for the specified image ID.

        Raises:
            FileNotFoundError: If the specified file cannot be found.
            ValueError: If the embedding for the specified image ID is not found in the file.
        """
        if 'file_name' in kwargs and kwargs['file_name'] != '':
            file_name = f"Embedding Files/" + kwargs['file_name']
        else:
            file_name = self.get_file_path(self.version, kwargs['image_id'])

        if file_name in self.files_in_memory:
            return self.image_embeddings[str(kwargs['image_id'])]

        max_files = int(kwargs['max_files']) if 'max_files' in kwargs and int(kwargs['max_files']) >= 1 else 100

        if len(self.files_in_memory) == max_files:
            lower_bound = int(self.files_in_memory[0].split('_')[3].split('_')[0])
            upper_bound = int(self.files_in_memory[0].split('_')[4].split('.')[0]) + 1

            for i in range(lower_bound, upper_bound):
                del self.image_embeddings[int(i)]

            del self.files_in_memory[0]

            self.files_in_memory.append(file_name)
        else:
            self.files_in_memory.append(file_name)

        try:
            with open(file_name, mode='r', newline='', encoding='utf-8') as csvfile:
                temp = csv.DictReader(csvfile, delimiter=';')

                for row in temp:
                    self.image_embeddings[row['image_id']] = torch.Tensor(ast.literal_eval(row['tensor']))
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find file: {file_name}, please check if you have it downloaded')

        if str(kwargs['image_id']) in self.image_embeddings:
            return self.image_embeddings[str(kwargs['image_id'])]

        if str(kwargs['image_id']) not in self.image_embeddings:
            raise ValueError(f"Embedding of {kwargs['image_id']} not found. Please check {file_name} to see if it is included")
     

class EmbeddingResNet_2_0(EmbeddingParent):
    """
    Class representing the EmbeddingResNet 2.0 model, without the last two layers.

    Attributes:
        version (float | str): The version of the model.
        name (str): The name of the model.

    Methods:
        Image2Vec_embedder_ResNet152(image) -> torch.Tensor:
            Converts an image to a vector embedding using the ResNet152 model, without the last 2 layers.

        run(**kwargs) -> torch.Tensor:
            Runs the model to generate image embeddings.

    """
    def __init__(self) -> None:
        self.version: float | str = '2.0 WIP'
        self.name: str = "EmbeddingResNet 2.0" 

    def Image2Vec_embedder_ResNet152(self, image) -> torch.Tensor:
        """
        Converts an image to a vector embedding using the ResNet152 model, without last 2 layers
        without the average pooling layer and without the fully connected layer.

        Args:
            image: The path to the image file.

        Returns:
            torch.Tensor: The vector embedding of the image.

        """
        resnet152 = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
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
        img = transform(img)
        vec = torch.tensor(resnet152(img.unsqueeze(0))) 
        return vec
        
    def run(self, **kwargs):
        """
        Runs the model to generate image embeddings.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The image embedding.

        """
        if 'file_name' in kwargs and kwargs['file_name'] != '':
            file_name = f"Embedding Files/" + kwargs['file_name']
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

class EmbeddingResNet_2_1(EmbeddingParent):
    """
    EmbeddingResNet_2_1 class represents a specific version of the embedding model
    that uses ResNet152 architecture with triplet loss.

    This class provides functionality to convert an image into a vector representation
    using the ResNet152 model. It also supports saving and loading image embeddings
    to/from a CSV file.

    Attributes:
        version (float | str): The version of the embedding model.
        name (str): The name of the embedding model.

    Methods:
        Image2Vec_embedder_ResNet152: Converts an image into a vector representation using ResNet152.
        run: Runs the embedding process and returns the image embedding vector.
    """
    def __init__(self) -> None:
        self.version: float | str = '2.1 WIP'
        self.name: str = "EmbeddingResNet 2.1 with triplet loss" 
    
    def Image2Vec_embedder_ResNet152(self, image) -> torch.Tensor:
        """
        Converts an image into a vector representation using the ResNet152 model.
        Implement own model with triplet loss. At resnet152 variable
        Args:
            image: The path to the image file.

        Returns:
            torch.Tensor: The vector representation of the image.
        """
        resnet152 = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
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
        img = transform(img)
        vec = torch.tensor(resnet152(img.unsqueeze(0))) 
        return vec
        
    def run(self, **kwargs):
        """
        Runs the embedding process and returns the image embedding vector.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The image embedding vector.
        """
        if 'file_name' in kwargs and kwargs['file_name'] != '':
            file_name = f"Embedding Files/" + kwargs['file_name']
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
        
        if str(kwargs['image_id']) in self.image_embeddings:
            return self.image_embeddings[str(kwargs['image_id'])]
        
        path = 'U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + kwargs['img_path']
        
        image_embedding = self.Image2Vec_embedder_ResNet152(path)

        self.image_embeddings[str(kwargs['image_id'])] = image_embedding

        with open(file_name, mode='w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['image_id', 'tensor'])

            if csvfile.tell() == 0:
                csv_writer.writeheader()

            for image_id, tensor in self.image_embeddings.items():
                csv_writer.writerow({'image_id': image_id, 'tensor': tensor.tolist()})
            
        return image_embedding