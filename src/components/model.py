import os

from torch import nn
import torch

from src.entity.config import PredictConfig
from torchsummary import summary

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config: PredictConfig = PredictConfig()
        self.basemodel = self.get_model()
        self.conv1 = nn.Conv2d(512, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.dropout2 = nn.Dropout2d(p=0.3)
        self.conv3 = nn.Conv2d(16, 4, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.dropout3 = nn.Dropout2d(p=0.3)
        self.flatten = nn.Flatten()
        self.final = nn.Linear(4 * 8 * 8, self.config.NUM_LABEL)

    

    def get_model(self) -> torch.nn.Sequential:
        torch.hub.set_dir(self.config.PATH) 
        model = torch.hub.load(
            self.config.REPO,
            self.config.BASE_MODEL,
            weights=self.config.PRETRAINED
        )
        return nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        x = self.basemodel(x)
        x = self.dropout1(self.conv1(x))
        x = self.dropout2(self.conv2(x))
        x = self.dropout3(self.conv3(x))
        return self.final(self.flatten(x))


if __name__ == "__main__":
    device = "cpu"
    model = NeuralNet()
    model.to(device)
    print(summary(model, input_size=(3,256,256), device=device))
    