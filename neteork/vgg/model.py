import torch.nn as nn
from torchvision import models


def getmodel(modelname,num_class,pre):
    if modelname == 'vgg11':
        model = models.vgg11(pretrained=pre)

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_class)
        return model
    if modelname == 'vgg16':
        model = models.vgg16(pretrained=pre)
        print(model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_class)
        return model
    if modelname == 'vgg13':
        model = models.vgg13(pretrained=pre)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_class)
        return model
    if modelname == 'vgg19':
        model = models.vgg19(pretrained=pre)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_class)
        return model

# if __name__ == '__main__':
#     net = getmodel('vgg16',20,True)
#     print(net)