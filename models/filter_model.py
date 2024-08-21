import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ClassificationFilterModel(nn.Module):

    def __init__(self, num_classes: int):
        super(ClassificationFilterModel, self).__init__()

        self.image_processor = None
        self.num_classes = nn.Parameter(torch.tensor(num_classes, dtype=torch.float32, requires_grad=False))
        self.temperature = nn.Parameter(torch.ones(1))

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.out = nn.Linear(2048, num_classes)

    def forward(self, image):
        x = image
        with torch.no_grad():
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)

        return self.out(x)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
