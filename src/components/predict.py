import os, sys
from PIL import Image
import io

from from_root import from_root
import torch
from torch import nn
from torchvision import transforms
import numpy as np

from src.components.model import NeuralNet
from src.components.custom_ann import CustomAnnoy
from src.components.storage import StorageConnection
from src.entity.config import PredictConfig
from src.logger import logging
from src.exception import CustomException

class Prediction:
    '''
    Prediction class Prepares the model endpoint
    '''

    def __init__(self):
        self.config: PredictConfig = PredictConfig()

        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        
        self.ann = CustomAnnoy(self.config.EMBEDDING_SIZE,
                               self.config.SEARCH_MATRIX)
        self.ann.load(self.config.MODEL_PATHS[1][0])
        self.estimator: torch.nn.Module = self.load_model()
        self.estimator.eval()
        self.estimator = self.estimator.to(self.device)
        self.transforms = self.transformations()
    
    def transformations(self) -> transforms.Compose:
        '''
        Transformation Method Provides transforms.Compose object. Its pytorch's transformation class to apply on images.
        return: transforms.Compose object
        '''
        TRANSFORM_OBJ = transforms.Compose(
            [
                transforms.Resize(self.config.IMAGE_SIZE),
                transforms.CenterCrop(self.config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.MEAN,
                                    std=self.config.STD)
            ]
        )

        return TRANSFORM_OBJ

    def load_model(self) -> torch.nn.Module:
        model = NeuralNet()
        model.load_state_dict(torch.load(self.config.MODEL_PATHS[2][0], map_location=self.device))
        return nn.Sequential(*list(model.children())[:-1])

    @staticmethod
    def setup():
        if not os.path.exists(os.path.join(from_root(), "artifacts")):
            os.makedirs(os.path.join(from_root(), "artifacts"))
            connection = StorageConnection()
            connection.get_package()
    
    def generate_embeddings(self, image: torch.tensor):
        image = image.to(self.device)
        with torch.inference_mode():
            embeddings = self.estimator(image.to(self.device))
        embeddings = embeddings.detach().cpu().numpy()
        return embeddings
    
    def generate_links(self, embeddings):
        return self.ann.get_nns_by_vector(embeddings, self.config.NUM_PREDICTIONS)
    

    def run_prediction(self, image):
        image = Image.open(io.BytesIO(image))
        if len(image.getbands()) < 3:
            image = image.convert('RGB')
        image = torch.from_numpy(np.array(self.transforms(image)))
        image = image.reshape(1, 3, 256, 256)
        embedding = self.generate_embeddings(image)
        return self.generate_links(embedding[0])
