from typing import Union
from Parent import GrandParent
from PIL import Image
from IMG2VEC_class import Img2Vec


class EmbeddingParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Embedding"
        self.children: dict[str, dict[str, Union[str, EmbeddingParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)


class EmbeddingADDMETHODNAME(EmbeddingParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "ADD METHOD NAME"

    def Image2Vec_embedder_ResNet50(image) -> torch.Tensor:
        img2vec = Img2Vec(cuda=True, model='resnet50', layer='default', layer_output_size=2048, gpu=0)
        # layer = 'layer_name' For advanced users, which layer of the model to extract the output from.   default: 'avgpool'
        img = Image.open(image).convert('RGB')
        vec = torch.tensor(img2vec.get_vec(img))
        return vec

    def Image2Vec_embedder_ResNet152(image) -> torch.Tensor:
        img2vec = Img2Vec(cuda=True, model='resnet152', layer='default', layer_output_size=2048, gpu=0)
        # layer = 'layer_name' For advanced users, which layer of the model to extract the output from.   default: 'avgpool'
        img = Image.open(image).convert('RGB')
        vec = torch.tensor(img2vec.get_vec(img))
        return vec
