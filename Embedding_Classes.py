from typing import Union
from Parent import GrandParent
from PIL import Image
from IMG2VEC_class import Img2Vec
from torch import torch
from pathlib import Path

class EmbeddingParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Embedding"
        self.children: dict[str, dict[str, Union[str, EmbeddingParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)


class EmbeddingResNet(EmbeddingParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "EmbeddingResNet"

    def Image2Vec_embedder_ResNet50(self, image) -> torch.Tensor:
        img2vec = Img2Vec(cuda=False, model='resnet50', layer='default', layer_output_size=2048, gpu=0)
        # layer = 'layer_name' For advanced users, which layer of the model to extract the output from.   default: 'avgpool'
        img = Image.open(image).convert('RGB')
        vec = torch.tensor(img2vec.get_vec(img))
        return vec

    def Image2Vec_embedder_ResNet152(self, image) -> torch.Tensor:
        img2vec = Img2Vec(cuda=False, model='resnet152', layer='default', layer_output_size=2048, gpu=0)
        # layer = 'layer_name' For advanced users, which layer of the model to extract the output from.   default: 'avgpool'
        img = Image.open(image).convert('RGB')
        vec = torch.tensor(img2vec.get_vec(img))
        return vec
    
    def run(self, **kwargs):
         # if not hasattr(self, 'image_embeddings'):
        #     with open():
        #         pass
        #     self.image_embeddings = 1
        path = 'U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + kwargs['img_path']

        if kwargs['resnet'] == 50:
            return self.Image2Vec_embedder_ResNet50(path)
        else:
            return self.Image2Vec_embedder_ResNet152(path)
        
