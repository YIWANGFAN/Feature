from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from network.dataload_tea import MyDataset

import torch
import torch.nn as nn


class Extract(nn.Module):
    def __init__(self):
        super(Extract, self).__init__()
        model = torch.load('/home/daip/share/old_share/wxf/feature/network/vgg/tea-cake/vgg16_tea.pth')
        self.features = model.features
        self.avgpool = model.avgpool

        fc1 = nn.Linear(25088, 4096)
        fc1.bias = model.classifier[0].bias
        fc1.weight = model.classifier[0].weight

        fc2 = nn.Linear(4096, 4096)
        fc2.bias = model.classifier[3].bias
        fc2.weight = model.classifier[3].weight

        self.classifier = nn.Sequential(
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            fc2,
            )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 加載模型
    model = Extract()
    model.to(device)
    model.eval()
    print(model)


    root = '/home/daip/share/old_share/wxf/datasets/tea-cake/'
    traindata = MyDataset(txt=root + 'teatrain.txt', transform=transforms.ToTensor())
    train_dataload = DataLoader(traindata, batch_size=1, shuffle=True)
    test_data = MyDataset(txt=root + 'teatest.txt', transform=transforms.ToTensor())
    test_dataload = DataLoader(test_data, batch_size=1, shuffle=True)
    trainsize = len(traindata)
    test_size = len(test_data)
    batch_size = 1
    print(trainsize, test_size)

    train_data = []
    train_lable = []
    train_save_path = '/home/daip/share/old_share/wxf/feature/feature/tea/vgg/vggFeature_tea_train_pre.npz'
    for step, data in enumerate(train_dataload, start=0):
        images, labels = data
        labels = labels-1
        labels = labels.to(device)
        outputs = model(images.to(device)).cpu()
        outputs = torch.squeeze(outputs)
        outputs = outputs.data.numpy()
        outputs = outputs.reshape(-1)
        labels = labels.cpu()
        labels = labels.data.numpy()
        labels = labels.reshape(-1)
        # print(outputs.shape)
        # print(labels)
        train_data.append(outputs)
        train_lable.append(labels)
        # lable.reshape(-1)
    train_lable = np.array(train_lable)
    train_lable = train_lable.reshape(-1)
    print(len(train_lable))
    np.savez(train_save_path, vector=train_data, utt=train_lable)

    test_data = []
    test_lable = []
    test_save_path = '/home/daip/share/old_share/wxf/feature/feature/tea/vgg/vggFeature_tea_test_pre.npz'
    for step, data in enumerate(test_dataload, start=0):
        images, labels = data
        labels = labels-1
        labels = labels.to(device)
        outputs = model(images.to(device)).cpu()
        outputs = torch.squeeze(outputs)
        outputs = outputs.data.numpy()
        outputs = outputs.reshape(-1)
        labels = labels.cpu()
        labels = labels.data.numpy()
        labels = labels.reshape(-1)
        # print(len(outputs))
        # print(labels)
        test_data.append(outputs)
        test_lable.append(labels)
        # lable.reshape(-1)
    test_lable = np.array(test_lable)
    test_lable = test_lable.reshape(-1)
    print(len(test_data))
    np.savez(test_save_path, vector=test_data, utt=test_lable)
