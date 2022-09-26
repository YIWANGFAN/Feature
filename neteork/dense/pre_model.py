import torch.nn as nn
from torchvision import models


def model(name,num_class):
    if name == 'densenet121':
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_class)
        return model
    if name == 'densenet169':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_class)
        return model
    if name == 'densenet201':
        model = models.densenet201(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_class)
        return model
    if name == 'densenet161':
        model = models.densenet161(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_class)
        return model
if __name__ == '__main__':
    model = model('densenet121',4)
    print(model)
