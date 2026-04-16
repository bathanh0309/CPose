import torch.nn as nn
from torchvision import models

class ResNetExtractor(nn.Module):
    """
    ResNet backbone for person ReID research.
    """
    def __init__(self, model_name='resnet50', pretrained=True):
        super(ResNetExtractor, self).__init__()
        base_model = getattr(models, model_name)(pretrained=pretrained)
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_dim = 2048

    def forward(self, x):
        features = self.base(x)
        features = features.view(features.size(0), -1)
        return features
