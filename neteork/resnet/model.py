import torch.nn as nn
from torchvision import models

def getmodel(modelname,num_class,pre):
    if modelname=='resnet18':
        model = models.resnet18(pretrained=pre)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)
        return model
    if modelname=='resnet50':
        model = models.resnet50(pretrained=pre)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)
        return model
    if modelname=='resnet101':
        model = models.resnet101(pretrained=pre)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)
        return model
    if modelname=='resnet152':
        model = models.resnet152(pretrained=pre)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)
        return model
if __name__ == '__main__':
    net = getmodel('resnet152',num_class=20,pre=True)
    print(net)