from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from network.dataload_tea import MyDataset

import torch
import torch.nn as nn


class Extract(nn.Module):
    def __init__(self):
        super(Extract, self).__init__()
        model = torch.load('/home/daip/share/old_share/wxf/feature/network/resnet/tea-cake/resnet50_tea.pth')
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.avgpool(x)
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
    train_save_path = '/home/daip/share/old_share/wxf/feature/feature/tea/resnet/resFeature_tea_train_pre.npz'
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
    test_save_path = '/home/daip/share/old_share/wxf/feature/feature/tea/resnet/resFeature_tea_test_pre.npz'
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
