from from_root import from_root
import os
from dotenv import load_dotenv

load_dotenv()


class s3Config:
    def __init__(self):
        self.ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
        self.SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
        self.REGION = os.environ['AWS_REGION']
        self.BUCKET = os.environ['AWS_BUCKET_NAME']
        self.KEY = "model"
        self.ARTIFACTS_ROOT = os.path.join(from_root(), "artifacts")
        self.ZIP_PATHS = ["embeddings.json",
                          "embeddings.ann",
                          "model.pth"]
        
    def get_s3config(self):
        return self.__dict__
    
class PredictConfig:
    def __init__(self):
        self.NUM_LABEL = 101
        self.REPO = "pytorch/vision:v0.13.0"
        self.BASE_MODEL = "resnet18"
        self.PRETRAINED = "ResNet18_Weights.DEFAULT"
        self.IMAGE_SIZE = 256
        self.EMBEDDING_SIZE = 256
        self.SEARCH_MATRIX = 'euclidean'
        self.NUM_PREDICTIONS = 20
        self.PATH = os.path.join(from_root(), "artifacts")
        self.MODEL_PATHS = [(os.path.join(from_root(), "artifacts", "embeddings.json"), "embeddings.json"),
                          (os.path.join(from_root(), "artifacts", "embeddings.ann"), "embeddings.ann"),
                          (os.path.join(from_root(), "model", "model.pth"), "model.pth")]
    
    def get_predict_config(self):
        return self.__dict__
